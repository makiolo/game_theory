# brownian

Sandbox de agentes con física de contacto sobre un medio térmico, en Rust con
Bevy y Rapier.

Conviven dos simulaciones acopladas:

- **Cuerpos rígidos** (Rapier, CPU): colisiones, rebotes y apilamiento de las
  formas que crea el usuario.
- **Campo de temperatura** (rejilla 2D): un medio continuo que se difunde cada
  paso, con tres backends intercambiables —serie, rayon y GPU— para comparar en
  caliente lo que aporta paralelizarlo.

El acoplamiento va en los dos sentidos: los cuerpos al moverse calientan el
medio, y el medio caliente los sacude con impulsos aleatorios. La magnitud del
impulso sigue la ecuación de Langevin (`sqrt(m·T)`), así que la velocidad
resultante va como `sqrt(T/m)`: las formas pequeñas vibran mucho y las grandes
apenas se inmutan, que es la firma visual del movimiento browniano.

## Ejecutar

```sh
cargo run                                  # 512x288, backend serie
cargo run -- --backend gpu                 # arranca en un backend concreto
cargo run -- --backend rayon --grid 2048x1152
cargo run --release                        # para medir de verdad
```

| Opción | Valores | Por defecto |
|---|---|---|
| `--backend` | `serial`, `rayon`, `gpu` | `serial` |
| `--grid` | `<ancho>x<alto>` | `512x288` |

## Controles

| | |
|---|---|
| Ratón izquierdo | crear la forma activa (mantener = chorro) |
| Ratón derecho | borrar el cuerpo bajo el cursor |
| Ratón central | inyectar calor en el medio |
| ← → | cambiar de forma (círculo, cuadrado, triángulo, cápsula) |
| ↑ ↓ | cambiar el tamaño |
| B | ciclar el backend del campo |
| R | reiniciar: vacía el recinto y enfría el medio |
| F3 | dibujar los colliders de Rapier |

## Rendimiento

El campo se difunde con un esquema explícito, que obliga a trocear el paso en
sub-pasos para no salirse del límite de estabilidad. Cuanto más fina es la
rejilla, más sub-pasos por frame — y ahí es donde se separan los backends.

Medida de un sub-paso aislado, en release (i7-4790K, 4 núcleos):

| Rejilla | serie | rayon |
|---|---|---|
| 512×288 (147 k celdas) | 0.09 ms | 0.06 ms |
| 1024×576 (590 k) | 0.44 ms | 0.16 ms |
| 2048×1152 (2.4 M) | 2.98 ms | 1.92 ms |

```sh
cargo test --release -- --ignored --nocapture bench
```

El paso completo de un frame a 2048×1152 son ~112 sub-pasos, y ahí la GPU se
despega porque los encadena todos en la tarjeta con una única subida y bajada de
datos (medido en debug, con una GTX 970):

| backend | paso del campo |
|---|---|
| serie | 1213 ms |
| rayon | 538 ms |
| gpu | 28 ms |

En la rejilla pequeña por defecto la GPU **no** gana: a 147 k celdas el cómputo
es tan corto que domina el coste de mover los datos, y los tres backends quedan
en torno a 1 ms.

## Estructura

```
src/
  config.rs      constantes de la simulación, todas juntas
  physics.rs     recinto, gravedad y cuerpos iniciales
  shapes.rs      formas disponibles → collider y malla
  spawn.rs       ratón y teclado, previsualización, crear y borrar
  field/
    mod.rs       ThermalField (ndarray) + trait FieldBackend
    serial.rs    backend de referencia, una hebra
    parallel.rs  backend rayon, filas repartidas entre hebras
    gpu.rs       backend de compute shader
    diffuse.wgsl el kernel, espejo del núcleo de CPU
  coupling.rs    el puente entre Rapier y el campo
  viz.rs         el campo como textura de fondo
  hud.rs         panel de estado
```

## Tests

```sh
cargo test
```

Cubren lo que puede romperse en silencio: que la difusión conserve el calor con
bordes reflectantes, que relaje hacia el baño térmico, que serie y rayon
coincidan **bit a bit** (cualquier divergencia delataría una carrera), que el
troceado en sub-pasos evite que el campo diverja con un `dt` grande, y que el
mapeo mundo↔celda sea correcto en los bordes.

## Notas de implementación

- **`wgpu` no es una dependencia directa.** Bevy usa wgpu 29 por dentro y
  crates.io va por la 30; declararlo aparte crearía un segundo `Device` con
  tipos incompatibles. El backend de GPU usa el `RenderDevice` que Bevy ya tiene
  abierto.
- **Rapier resuelve la física en CPU.** No hay backend de GPU para su
  broad-phase ni su solver; lo que se paraleliza aquí es el campo.
- **Rapier trabaja en píxeles, no en metros.** `pixels_per_meter` ajusta la
  escala de la simulación pero no encoge la geometría, así que una pelota de
  18 px de radio tiene masa ≈1018, no 0.1 — de ahí las constantes grandes en
  `config.rs`.
- **La gravedad está al 15%.** Las partículas brownianas reales están
  suspendidas en un fluido que compensa casi todo su peso; con gravedad completa
  el peso aplasta la agitación térmica y las formas solo se apilan en el suelo.
