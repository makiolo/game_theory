//! El paso de la simulacion visto desde Bevy: quien lo dispara, con que backend
//! y cuanto cuesta.
//!
//! El acoplamiento entre cuerpos y campo ya no esta aqui — vive dentro de
//! [`brownian_core::Env`], que es quien lo encadena. Lo que queda en la app es
//! decidir *cuando* se da un paso (una vez por tick de `FixedUpdate`), con cual
//! de los tres backends, y cronometrarlo para el HUD.

use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use std::time::Instant;

use brownian_core::field::parallel::RayonBackend;
use brownian_core::field::serial::SerialBackend;
use brownian_core::{BackendKind, FieldBackend};

use crate::config;
use crate::gpu::GpuBackend;
use crate::res::{Backend, Sim};
use crate::spawn::cursor_world_position;

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

/// Cambia de backend con la tecla B.
pub fn cycle_backend(keys: Res<ButtonInput<KeyCode>>, mut kind: ResMut<Backend>) {
    if keys.just_pressed(KeyCode::KeyB) {
        **kind = kind.next();
    }
}

/// Boton central del raton: inyecta calor donde apunta el cursor. Sirve para
/// ver la difusion y para agitar a mano una zona del recinto.
///
/// Va antes del paso, para que el calor que suelta entre en la difusion de este
/// mismo paso y no en la del siguiente.
pub fn heat_brush(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut sim: ResMut<Sim>,
) {
    if !buttons.pressed(MouseButton::Middle) {
        return;
    }
    let (Ok(window), Ok((camera, camera_transform))) = (windows.single(), cameras.single()) else {
        return;
    };
    let Some(world) = cursor_world_position(window, camera, camera_transform) else {
        return;
    };

    sim.field
        .deposit(world, config::HEAT_BRUSH * config::SIM_DT);
}

/// Avanza el mundo un paso con el backend activo, y cronometra el campo.
pub fn step_sim(
    kind: Res<Backend>,
    mut backends: ResMut<Backends>,
    mut sim: ResMut<Sim>,
    mut stats: ResMut<FieldStats>,
) {
    let started = Instant::now();
    let backend = backends.get_mut(**kind);
    sim.step(backend);
    stats.step_ms = started.elapsed().as_secs_f32() * 1000.0;

    // Suavizado exponencial: lo que se lee en pantalla tiene que ser estable.
    let w = 0.1;
    stats.avg_step_ms = if stats.avg_step_ms == 0.0 {
        stats.step_ms
    } else {
        stats.avg_step_ms * (1.0 - w) + stats.step_ms * w
    };
}
