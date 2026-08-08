# Plan de integración RL: brownian ↔ PyTorch

Convertir el sandbox en un entorno de aprendizaje por refuerzo multi-agente
vectorizado, con observaciones como tensor `[N, C, H, W]` y transferencia
zero-copy hasta la GPU.

Estado de partida (commit `cbfab94`):

| | |
|---|---|
| `ThermalField` | `Array2<f32>` row-major `[H, W]`, doble buffer, 512×288 |
| Agentes | entidades Bevy con marker `Agent`, cuerpos Rapier dinámicos |
| Acoplamiento | `deposit_heat → heat_brush → step_field → apply_brownian_impulse` |
| Reloj | `Time` de Bevy, `dt` variable acotado por `MAX_SIM_DT` |
| GPU | compute shader sobre el `RenderDevice` de Bevy, `poll` bloqueante por frame |

---

## 1. Arquitectura objetivo

### 1.1 Invertir la dependencia: la física debajo de Bevy, no dentro

Hoy la simulación vive **dentro** de Bevy: el estado son `Resource`s, el avance
son sistemas del `Schedule`, el `dt` lo dicta el renderer. Un entorno de
entrenamiento necesita lo contrario: una función `step(estado, acciones)` que se
pueda llamar millones de veces, replicar K veces en un pool de hebras y ejecutar
sin ventana ni GPU.

```
brownian-core/     lib pura: rapier2d, ndarray, glam. Cero Bevy, cero render.
                   Define Env::step(). Es lo que consume Python.
brownian-app/      binario actual: Bevy + wgpu. Consume core y lo dibuja.
                   Aquí siguen el HUD, el ratón y el backend GPU del campo.
brownian-py/       cdylib PyO3. Consume core. Expone VecEnv a Python.
```

### 1.2 El motor de física: `rapier2d` oficial, sin reimplementar nada

**Decisión: `brownian-core` depende directamente del crate oficial de Dimforge
`rapier2d = "0.33.0-alpha"`, usando su API pública documentada.**

No se reescribe ni se reimplementa `bevy_rapier2d`. Se baja un nivel, al motor
que `bevy_rapier2d` ya usa por dentro. Verificado en `Cargo.lock`:

```
bevy_rapier2d 0.35.0  →  rapier2d 0.33.0-alpha  →  parry2d 0.27.0
```

Al ser **la misma versión que ya está en el lock**, cargo no duplica el crate:
`core` y `app` comparten los mismos tipos, y `bevy_rapier2d` sigue funcionando
en la app sin conflictos.

La API oficial es la del user guide de Dimforge, y es la que usa
`bevy_rapier2d/src/plugin/context/mod.rs:763`:

```rust
use rapier2d::prelude::*;

pipeline.step(
    gravity, &integration_parameters,
    &mut islands, &mut broad_phase, &mut narrow_phase,
    &mut bodies, &mut colliders,
    &mut impulse_joints, &mut multibody_joints, &mut ccd_solver,
    &(), &(),                 // PhysicsHooks, EventHandler
);
```

Equivalencias con lo que hoy hace el plugin, todas oficiales:

| Hoy (`bevy_rapier2d`) | En `brownian-core` (`rapier2d`) |
|---|---|
| `RapierPhysicsPlugin::pixels_per_meter(100.0)` | `IntegrationParameters::length_unit = 100.0` |
| `RapierConfiguration::gravity` | argumento `gravity` de `step()` |
| `TimestepMode::Fixed { dt, substeps }` | `IntegrationParameters::dt` |
| `Collider::ball(r)` / `cuboid(hx, hy)` | `ColliderBuilder::ball(r)` / `cuboid(hx, hy)` |
| `RigidBody::Dynamic` + `Damping` | `RigidBodyBuilder::dynamic().linear_damping(..)` |
| `ExternalImpulse` | `RigidBody::apply_impulse(v, true)` |
| `ReadMassProperties` | `RigidBody::mass()` |
| `Ccd::enabled()` | `RigidBodyBuilder::ccd_enabled(true)` |
| `Sleeping::disabled()` | `RigidBodyBuilder::can_sleep(false)` |
| `context.intersect_point(..)` | `BroadPhaseBvh` (la query pipeline vive ahí en 0.33) |

**Hallazgo al implementarlo:** desde la 0.33 rapier trabaja en **glam**, no en
nalgebra — su alias `Vector` es literalmente `glam::Vec2`, vía `glamx 0.2` sobre
`glam 0.32.1`, *la misma versión exacta que usa Bevy*. Resultado: no hay una sola
conversión de vectores entre `core`, `app` y rapier. Los `vector![x, y]` de la
guía (que producen tipos de nalgebra) no compilan aquí; se usa `Vec2` directo.

Feature relevante para RL: `rapier2d/enhanced-determinism` fuerza `libm` en las
funciones trascendentes y garantiza reproducibilidad entre máquinas IEEE-754.
Es incompatible con `simd-stable`. Se evalúa midiendo el coste.

### 1.3 Transporte Rust ↔ Python

| Opción | Latencia/step | Zero-copy | Veredicto |
|---|---|---|---|
| **PyO3 + rust-numpy** (Rust como `.pyd`) | ~µs | Sí, buffer protocol | **Elegida** |
| Proceso aparte + memoria compartida | ~decenas de µs | Sí | Modo espectador opcional |
| gRPC / ZeroMQ / msgpack | ~ms + serialización | No | Descartada |

El descarte de la tercera es aritmético: el campo completo son 512·288·4 = 590 KB.
Serializarlo por env y por step, a 60 Hz y K=32, son ~1.1 GB/s de tráfico que no
existe si la memoria es la misma.

El segundo motivo, más importante: con Python como proceso host, **Python conduce
el reloj**. Eso disuelve el problema de "Rust va a 144 FPS y Python a 30" durante
el entrenamiento — no hay dos relojes, hay una llamada a función.

### 1.4 Vectorización en Rust, no en Python

Nada de `SubprocVecEnv`. Un solo `step()` avanza **K entornos en paralelo dentro
de Rust** con rayon y escribe en un único tensor: se eliminan K procesos, K
pickles por step y K copias.

Corolario: Rapier va **sin** la feature `parallel`. Y hay que quitarla de
`bevy_rapier2d`, no solo omitirla en `core`: **cargo compila una sola copia de
`rapier2d` para todo el workspace y unifica sus features**, así que
`bevy_rapier2d[parallel]` se la activaba también a `core`.

Verificado con `cargo tree --workspace -e features -i rapier2d`. El detalle
traicionero es que los tests de determinismo pasaban igualmente: con
`min_island_size = 128`, Rapier no reparte entre hebras hasta tener ~128 cuerpos
en una isla, de modo que a la escala de un test el determinismo salía por tamaño
y no por construcción — y se habría roto justo al escalar.

---

## 2. Plan por fases

### Fase 0 — Determinismo (prerrequisito) ✅

Sin esto nada de lo demás se puede depurar.

1. **`dt` fijo.** `config::SIM_DT = 1/60` sustituye a `MAX_SIM_DT`. El tope
   actual protege contra la espiral de sub-pasos, pero hace que la trayectoria
   dependa de la carga de la máquina.
2. **Rapier con `TimestepMode::Fixed { dt: SIM_DT, substeps: 1 }`** (es un
   `Resource` en bevy_rapier2d 0.35).
3. **RNG indexado, no secuencial.** `apply_brownian_impulse` consumía el
   `ChaCha8Rng` en el orden que devolvía la `Query`, que el ECS no garantiza. Se
   sustituye por ruido *counter-based*: `hash(episode_seed, step, slot)`. Así el
   ruido no depende del orden de iteración ni del paralelismo — es lo que
   permitirá luego rasterizar y sacudir con rayon sin cambiar la trayectoria.
4. **Identidad estable**: `Agent` pasa de marker a `Agent { slot: u32 }`.

> **Criterio:** el ruido de un agente depende solo de `(seed, step, slot)` y es
> invariante al orden de iteración. Verificado en `sim::tests`.

**Limitación conocida que resuelve la Fase 2:** `deposit_heat` suma en `f32`
sobre el campo en el orden de la `Query`. Con dos agentes en la misma celda, un
reordenamiento de arquetipos cambiaría el último bit. Se cierra cuando
`AgentTable` imponga el orden por slot.

### Fase 1a — Extraer `brownian-core` ✅

Workspace de tres crates. `ThermalField`, `DiffusionParams`, `diffuse_row` y los
backends serial/rayon se mueven casi tal cual (quitando `derive(Resource)` y
cambiando `bevy::math::Vec2` por `glam::Vec2`, que es el mismo tipo).

El **backend GPU se queda en `brownian-app`**: su `device.poll(wait_indefinitely())`
por frame es exactamente lo que no se quiere en un bucle de entrenamiento.

```rust
pub struct Env { /* §3 */ }

impl Env {
    pub fn new(cfg: &EnvConfig) -> Self;
    pub fn reset(&mut self, seed: u64);
    /// Avanza un paso. `actions` es [N, 2] en C-order.
    pub fn step(&mut self, actions: &[f32]);
    /// Rasteriza la observación en un buffer ajeno. No asigna.
    pub fn observe(&self, out: &mut [f32]);
    pub fn rewards(&self) -> &[f32];
    pub fn terminated(&self) -> bool;
}
```

Tres decisiones al repartir el código:

- **`config` partido.** `core` se queda lo que decide cómo se mueve el mundo; la
  app conserva un `config` propio con lo visual que hace
  `pub use brownian_core::config::*`, así ningún `config::X` de la app cambia.
- **El parseo de `--grid` / `--backend` baja a la app.** Leer la línea de órdenes
  no es cosa de una librería de simulación; `ThermalField` gana `with_grid()`.
- **Envoltorios en `app/src/res.rs`.** `core` no puede derivar `Resource` (regla
  del huérfano, y sería justo la dependencia que la separación evita), así que la
  app envuelve `ThermalField`/`SimClock`/`BackendKind` con `Deref`.

> **Criterio, revisado:** `cargo test --workspace` verde (17 tests) y la app
> arranca y se comporta igual.
>
> El criterio original decía «una trayectoria de 10 000 pasos coincide entre
> `core` headless y la app». **No se implementó**, por dos motivos: exige montar
> un `App` de Bevy headless dentro de un test, y sobre todo deja de tener sentido
> en cuanto la Fase 1b haga que la app *consuma* `core` — no habrá dos
> implementaciones que comparar. En su lugar, `env::tests` fija cinco propiedades
> del mundo, entre ellas que la velocidad va como `sqrt(T/m)`
> (`small_bodies_shake_more_than_large_ones`), que es la comprobación de que el
> acoplamiento sobrevivió al cambio de motor.

### Fase 1b — La app consume `core` ✅

`bevy_rapier2d` fuera del árbol de dependencias. Los cuerpos viven en `Env` y
Bevy solo dibuja: una entidad es malla, material y `Transform`, atada a su cuerpo
por el `slot` que guarda el componente `Agent`. `spawn.rs::sync_transforms` cierra
el paso llevando las poses del `Env` a los `Transform`.

Lo que daba el plugin, rehecho:

| Antes | Ahora |
|---|---|
| `RapierDebugRenderPlugin` (F3) | `debug.rs`, gizmos sobre `Env::agent_poses()` |
| `context.intersect_point` | `Env::despawn_at` |
| `Velocity` en el HUD | `Env::mean_speed()` |
| Colliders de `bevy_rapier2d` | `ShapeKind::collider()` en `core` |

Los contornos del debug salen de los mismos parámetros que el collider — el del
triángulo, literalmente de los mismos vértices — así que no pueden desviarse de
lo que colisiona.

**Bug encontrado por el test, no por la vista:** consultar el broad-phase para
borrar por punto falla con los cuerpos recién creados. Su BVH no conoce un cuerpo
hasta que pasa por `PhysicsPipeline::step`, de modo que un cuerpo recién soltado
con el ratón era inmune al clic derecho hasta el siguiente tick. `despawn_at`
recorre los agentes y comprueba `contains_point`: O(n) sobre decenas de cuerpos,
una vez por clic, y siempre correcto. Fijado en
`a_body_can_be_removed_before_the_first_step`.

> **Criterio:** `cargo test --workspace` verde (22 tests), clippy sin avisos, la
> app arranca, y `cargo tree` confirma que `bevy_rapier2d` ya no está y que
> `brownian-core` no depende de Bevy.

### Fase 1c — Pendiente

`Env` no expone todavía nada para *actuar* sobre los cuerpos desde fuera: hace
falta el espacio de acción de la Fase 2 antes de que una política pueda tocar
nada.

### Fase 2 — Agentes y espacio de acción

- `AgentTable` SoA con slots reciclables (§3.2).
- Acción continua `Box(-1, 1, shape=(N, 2))`: un impulso dirigido que **se suma**
  al browniano en vez de sustituirlo. Mantiene la física intacta y obliga a la
  política a trabajar contra el ruido, que es lo interesante del dominio.
- Recompensa calculada en Rust, dentro del step.

### Fase 3 — Observaciones (§3.3)

Rasterizado egocéntrico con rayon sobre agentes, sin asignaciones.

> **Criterio:** `observe()` con N=16, crop 32 → < 100 µs y cero allocs
> (verificable con un allocador contador en el test).

### Fase 4 — `brownian-py`

PyO3 + rust-numpy + maturin. `VecEnv` envolviendo `Vec<Env>`.

> **Criterio:** `pip install -e .` y 1000 steps con acciones aleatorias, midiendo
> steps/s.

### Fase 5 — Solapamiento asíncrono (§4.2)

`send()` / `recv()` con doble buffer. Optimización, no funcionalidad: hacerla
solo si el profiling muestra la CPU ociosa durante el forward.

### Fase 6 — Gymnasium + PPO

Wrapper `VectorEnv`, autoreset con `final_observation` (§3.5), PPO con parameter
sharing entre agentes.

### Fase 7 — Espectador

`brownian-app --policy modelo.pt`, con la política a menor frecuencia que el
render (§4.3).

---

## 3. Estructuras de datos

### 3.1 El principio: Python asigna, Rust rellena

La tentación es que Rust sea dueño del `Vec<f32>` y lo exponga como array numpy
con `borrow_from_array`. **No**: obliga a garantizar a mano que Python no
conserva la vista mientras Rust muta. El patrón correcto es el de `torch.*(out=)`:

> Python asigna los buffers **una vez**, al construir el env, y se los presta a
> Rust en cada `step`. Rust escribe dentro y no es dueño de nada.

Cero allocations por step, cero copias, y el lifetime lo resuelve el borrow
checker sin `unsafe`.

### 3.2 Lado Rust: SoA denso, no ECS

Todo lo que corre por agente y por paso lee de una tabla columnar que se
sincroniza una vez por step, en vez de hacer acceso aleatorio desde una query:

```rust
pub struct AgentTable {
    pub handle: Vec<RigidBodyHandle>,
    pub pos:    Vec<Vec2>,
    pub vel:    Vec<Vec2>,
    pub mass:   Vec<f32>,
    /// Celda del campo, resuelta una sola vez por paso. Hoy la recalculan por
    /// separado `deposit_heat` y `apply_brownian_impulse`.
    pub cell:   Vec<[u32; 2]>,
    pub alive:  Vec<bool>,
    free: Vec<u32>,   // slots reutilizables: el indice que ve la politica
}                     // tiene que sobrevivir a que muera un agente
```

`brownian-app` conserva entidades Bevy solo para malla y material; el `Transform`
se copia desde `AgentTable.pos` al final del frame.

### 3.3 El tensor de observación

**Decisión: observación egocéntrica recortada, no campo global.** La aritmética
lo impone — con el campo completo (512×288), C=4, N=64:

```
64 · 4 · 288 · 512 · 4 B = 151 MB por env y por step   ❌
```

Con un recorte `Hc×Wc` centrado en la celda del agente:

```
bytes_por_step = K · N · C · Hc · Wc · 4
```

| K envs | N agentes | C | crop | por step | rollout T=128 |
|---:|---:|---:|---:|---:|---:|
| 32 | 64 | 5 | 64 | 168 MB | 21 GB ❌ |
| 64 | 16 | 4 | 32 | 16.8 MB | 2.1 GB ⚠️ |
| **32** | **16** | **4** | **32** | **8.4 MB** | **1.07 GB** ✅ |

Empezar por la última fila. Esta tabla es el contrato: cualquier subida de
`crop`, `N` o `C` se valida contra ella antes de codificarla.

El recorte no es solo ahorro — hace la política invariante a traslación, lo que
permite **compartir parámetros entre los N agentes**: una red, N muestras/step.

Canales (C=4):

| Canal | Contenido |
|---|---|
| 0 | Temperatura normalizada, `(T − ambient) / T_scale` |
| 1 | Densidad de masa de otros agentes, rasterizada |
| 2–3 | Velocidad media rasterizada, `(vx, vy)` |

Salida en `[K·N, C, Hc, Wc]` C-order, que es NCHW nativo de `conv2d`. Coincidencia
afortunada: `Array2<f32>` ya es row-major `[fila, columna]`, el layout exacto de
un plano `[Hc, Wc]` — **copiar una fila del recorte es un `memcpy`**.

```rust
pub struct ObsView<'a> { data: &'a mut [f32], spec: ObsSpec }

impl<'a> ObsView<'a> {
    /// Reparte el tensor en un bloque por agente. Son disjuntos por
    /// construccion, asi que rayon los escribe a la vez sin `unsafe`.
    pub fn par_agents(&mut self) -> impl IndexedParallelIterator<Item = &mut [f32]> {
        self.data.par_chunks_mut(self.spec.per_agent())
    }
}
```

### 3.4 Lado Python: la cadena sin copias hasta la GPU

```python
# pin_memory=True hace la transferencia a GPU un DMA asincrono. .numpy() da una
# vista del MISMO buffer, que es el que Rust rellenara: numpy y torch no se
# copian entre si, y Rust escribe directamente en memoria pinneada.
self._obs_t = torch.empty((K * N, C, crop, crop), dtype=torch.float32, pin_memory=True)
self.obs = self._obs_t.numpy()
self._raw = _core.VecEnv(K, N, crop, C, seed)

def step(self, actions_t):
    self._act_t.copy_(actions_t)                              # unica copia: GPU -> pinned
    self._raw.step(self.act, self.obs, self.rew, self.done)   # Rust escribe in-place
    return self._obs_t.to('cuda', non_blocking=True), self._rew_t, self._done_t
```

Binding:

```rust
fn step(
    &mut self, py: Python<'_>,
    actions: PyReadonlyArray2<'_, f32>,
    mut obs: PyReadwriteArray4<'_, f32>,
    mut rew: PyReadwriteArray1<'_, f32>,
    mut done: PyReadwriteArray1<'_, bool>,
) -> PyResult<()> {
    // `as_slice` falla si el array no es contiguo: mejor un error claro aqui
    // que una observacion silenciosamente transpuesta.
    let acts = actions.as_slice()?;
    let (obs, rew, done) = (obs.as_slice_mut()?, rew.as_slice_mut()?, done.as_slice_mut()?);
    // Sin objetos de Python vivos dentro, se puede soltar el GIL.
    py.allow_threads(|| self.inner.step_all(acts, obs, rew, done));
    Ok(())
}
```

`allow_threads` exige que el closure sea `Ungil`; `&mut [f32]` lo es. Versiones:
PyO3 ≥ 0.23 con el crate `numpy` a juego (rust-numpy sigue la numeración de PyO3).

### 3.5 Autoreset

Devolver la observación terminal como si fuera la del siguiente paso hace que el
crítico haga bootstrap sobre un estado que ya no existe. Convención de
Gymnasium, implementada **en Rust dentro del step**:

```rust
if env.terminated() {
    env.write_obs(&mut final_obs[slot]);   // buffer aparte, para el bootstrap
    env.reset(next_seed);
    env.write_obs(&mut obs[slot]);         // la que ve la politica
    done[slot] = true;
}
```

---

## 4. Concurrencia

### 4.1 Entrenamiento: control invertido (por defecto)

Python es el hilo principal y llama `step()`. Rust no tiene bucle propio ni FPS.
**Cero locks, cero canales, cero desincronía.** Dentro, rayon reparte los K envs:

```rust
self.envs.par_iter_mut()
    .zip(acts.par_chunks(per_a))
    .zip(obs.par_chunks_mut(per_o))
    .zip(rew.par_chunks_mut(self.agents))
    .zip(done.par_iter_mut())
    .for_each(|((((env, a), o), r), d)| {
        env.step(a); env.observe(o);
        r.copy_from_slice(env.rewards()); *d = env.terminated();
    });
```

Cada env es un dominio de memoria disjunto: no comparten nada, ni siquiera el
RNG (counter-based por env, Fase 0). Escalado lineal y determinismo por env
independiente del número de hebras.

### 4.2 Solapar simulación y aprendizaje

El cuello real no es que Rust y Python vayan a distinta velocidad, es que
**mientras la GPU hace forward+backward, los núcleos están parados**. Patrón
EnvPool: partir el pool en dos mitades con doble buffer.

```python
env.send(actions_a)                 # no bloquea: Rust arranca la mitad A
logits = policy(obs_b)              # la GPU trabaja sobre la mitad B
obs_a, rew_a, done_a = env.recv()   # sincroniza
```

Dos buffers alternos garantizan que Rust nunca escribe sobre el que Python está
leyendo: la disjunción es temporal, no protegida por locks.

### 4.3 Modo interactivo: aquí sí hay dos relojes

Único régimen con desincronía real (Bevy a 60–144 FPS, política a 10–30 Hz).
Tres mecanismos, ninguno con `Mutex`:

1. **Action repeat.** La política decide cada `k` frames y la acción se mantiene.
   Estándar (Atari usa 4); cubre el 90% del caso por sí solo.
2. **Publicación sin bloqueo.** El hilo de simulación publica el último snapshot
   con `ArcSwap<Snapshot>`; el de inferencia hace `load()` sin detener a nadie.
3. **Retorno con descarte.** Las acciones vuelven por un canal `crossbeam` de
   capacidad 1 con `try_send`: si está lleno se descarta la vieja. Siempre gana
   la más fresca.

La propiedad que da el diseño: bajo carga la degradación es *"la política actúa a
menos Hz"*, no *"la simulación se atasca"*. Un `Mutex<State>` daría lo contrario.

---

## 5. Riesgos

| Riesgo | Mitigación |
|---|---|
| Determinismo entre máquinas | `enhanced-determinism` de rapier2d si el coste lo permite; si no, garantizar determinismo por máquina y semilla, que es lo que necesita depurar RL |
| `rapier2d 0.33` es **alpha** | Es la versión que ya arrastra `bevy_rapier2d 0.35`; fijarla exacta en el workspace para que app y core no diverjan |
| Presupuesto de memoria del rollout | La tabla de §3.3 es el contrato |
| `as_slice_mut()` con arrays no contiguos | El `?` propaga error claro; documentar que los buffers no se pueden *slicear* en Python |
