//! El puente entre la fisica de Rapier (CPU, cuerpos rigidos) y la rejilla
//! termica (paralela, campo continuo).
//!
//! Va en los dos sentidos:
//!
//! * **cuerpos -> campo**: al moverse, un cuerpo agita el medio y deposita
//!   calor proporcional a su energia cinetica especifica.
//! * **campo -> cuerpos**: el medio caliente devuelve empujones aleatorios.
//!   La magnitud va como `sqrt(T)` porque la energia es cuadratica en la
//!   velocidad, y como `sqrt(dt)` porque es ruido blanco integrado: asi la
//!   agitacion no depende de a cuantos fps vaya la simulacion.

use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy_rapier2d::prelude::*;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f32::consts::TAU;
use std::time::Instant;

use crate::config;
use crate::field::gpu::GpuBackend;
use crate::field::parallel::RayonBackend;
use crate::field::serial::SerialBackend;
use crate::field::{BackendKind, FieldBackend, ThermalField};
use crate::spawn::{Agent, cursor_world_position};

/// Los backends viven aqui para poder cambiar de uno a otro en caliente sin
/// reconstruir nada.
#[derive(Resource)]
pub struct Backends {
    serial: SerialBackend,
    rayon: RayonBackend,
    gpu: GpuBackend,
}

impl Backends {
    /// Construye los tres. El de GPU toma prestado el dispositivo que Bevy ya
    /// tiene abierto, en vez de abrir uno propio.
    pub fn new(device: RenderDevice, queue: RenderQueue) -> Self {
        Self {
            serial: SerialBackend,
            rayon: RayonBackend,
            gpu: GpuBackend::new(device, queue),
        }
    }

    pub fn get_mut(&mut self, kind: BackendKind) -> &mut dyn FieldBackend {
        match kind {
            BackendKind::Serial => &mut self.serial,
            BackendKind::Rayon => &mut self.rayon,
            BackendKind::Gpu => &mut self.gpu,
        }
    }

    /// El nombre lo da el propio backend, para que lo que muestra el HUD no
    /// pueda desviarse de lo que se esta ejecutando.
    pub fn name(&self, kind: BackendKind) -> &'static str {
        match kind {
            BackendKind::Serial => self.serial.name(),
            BackendKind::Rayon => self.rayon.name(),
            BackendKind::Gpu => self.gpu.name(),
        }
    }
}

/// El backend de GPU necesita el dispositivo, que no existe hasta que arranca
/// el renderer, asi que el recurso se monta en `Startup` y no con `init_resource`.
pub fn setup_backends(mut commands: Commands, device: Res<RenderDevice>, queue: Res<RenderQueue>) {
    commands.insert_resource(Backends::new(device.clone(), queue.clone()));
}

/// Medidas del ultimo paso, para el HUD.
#[derive(Resource, Default)]
pub struct FieldStats {
    pub step_ms: f32,
    /// Media movil, que el valor instantaneo es demasiado ruidoso para leerlo.
    pub avg_step_ms: f32,
}

#[derive(Resource)]
pub struct BrownianRng(pub ChaCha8Rng);

impl Default for BrownianRng {
    fn default() -> Self {
        Self(ChaCha8Rng::seed_from_u64(config::RNG_SEED))
    }
}

/// Paso de tiempo que usan los tres sistemas del acoplamiento. Va acotado para
/// que un frame lento no dispare el numero de sub-pasos del campo.
fn sim_dt(time: &Time) -> f32 {
    time.delta_secs().min(config::MAX_SIM_DT)
}

/// Cambia de backend con la tecla B.
pub fn cycle_backend(keys: Res<ButtonInput<KeyCode>>, mut kind: ResMut<BackendKind>) {
    if keys.just_pressed(KeyCode::KeyB) {
        *kind = kind.next();
    }
}

/// Cuerpos -> campo: el rozamiento con el medio lo calienta.
pub fn deposit_heat(
    time: Res<Time>,
    mut field: ResMut<ThermalField>,
    agents: Query<(&Transform, &Velocity), With<Agent>>,
) {
    let dt = sim_dt(&time);
    if dt <= 0.0 {
        return;
    }

    for (transform, velocity) in &agents {
        let speed2 = velocity.linear.length_squared();
        if speed2 <= 0.0 {
            continue;
        }
        let pos = transform.translation.truncate();
        field.deposit(pos, config::HEAT_PER_SPEED2 * speed2 * dt);
    }
}

/// Boton central del raton: inyecta calor donde apunta el cursor. Sirve para
/// ver la difusion y para agitar a mano una zona del recinto.
pub fn heat_brush(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut field: ResMut<ThermalField>,
) {
    if !buttons.pressed(MouseButton::Middle) {
        return;
    }
    let dt = sim_dt(&time);
    let (Ok(window), Ok((camera, camera_transform))) = (windows.single(), cameras.single()) else {
        return;
    };
    let Some(world) = cursor_world_position(window, camera, camera_transform) else {
        return;
    };

    field.deposit(world, config::HEAT_BRUSH * dt);
}

/// Avanza la difusion con el backend activo y cronometra el paso.
pub fn step_field(
    time: Res<Time>,
    kind: Res<BackendKind>,
    mut backends: ResMut<Backends>,
    mut field: ResMut<ThermalField>,
    mut stats: ResMut<FieldStats>,
) {
    let dt = sim_dt(&time);
    if dt <= 0.0 {
        return;
    }

    let started = Instant::now();
    let backend = backends.get_mut(*kind);
    field.step(backend, dt);
    stats.step_ms = started.elapsed().as_secs_f32() * 1000.0;

    // Suavizado exponencial: lo que se lee en pantalla tiene que ser estable.
    let w = 0.1;
    stats.avg_step_ms = if stats.avg_step_ms == 0.0 {
        stats.step_ms
    } else {
        stats.avg_step_ms * (1.0 - w) + stats.step_ms * w
    };
}

/// Campo -> cuerpos: el medio caliente los sacude en direccion aleatoria.
pub fn apply_brownian_impulse(
    time: Res<Time>,
    field: Res<ThermalField>,
    mut rng: ResMut<BrownianRng>,
    mut agents: Query<(&Transform, &ReadMassProperties, &mut ExternalImpulse), With<Agent>>,
) {
    let dt = sim_dt(&time);
    if dt <= 0.0 {
        return;
    }
    // Ruido blanco integrado sobre dt: la desviacion crece con sqrt(dt).
    let dt_scale = dt.sqrt();

    for (transform, mass, mut impulse) in &mut agents {
        let temperature = field.sample(transform.translation.truncate());
        if temperature <= 0.0 {
            continue;
        }

        let angle = rng.0.random_range(0.0..TAU);
        let magnitude =
            config::IMPULSE_SCALE * (mass.get().mass * temperature).sqrt() * dt_scale;
        impulse.impulse += Vec2::new(angle.cos(), angle.sin()) * magnitude;
    }
}
