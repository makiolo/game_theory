//! brownian — sandbox de agentes con fisica de contacto sobre un medio termico.
//!
//! Dos simulaciones acopladas conviven aqui:
//!
//! * **cuerpos rigidos** (Rapier, CPU): colisiones, rebotes y apilamiento de las
//!   formas que crea el usuario;
//! * **campo de temperatura** (rejilla 2D): un medio continuo que se difunde y
//!   que se recorre con backends intercambiables — serie o rayon — para poder
//!   comparar en caliente lo que cuesta paralelizarlo.
//!
//! El acoplamiento entre ambos vive en [`coupling`].

// Los sistemas de Bevy declaran sus dependencias como parametros, asi que pasar
// de siete es lo normal y no dice nada sobre su complejidad.
#![allow(clippy::too_many_arguments)]

mod config;
mod coupling;
mod field;
mod hud;
mod physics;
mod shapes;
mod spawn;
mod viz;

use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::input::common_conditions::input_just_pressed;
use bevy::prelude::*;
use bevy::window::{MonitorSelection, WindowPosition, WindowResolution};
use bevy_rapier2d::prelude::*;

use field::{BackendKind, ThermalField};

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "brownian".to_string(),
                    resolution: WindowResolution::new(config::WINDOW_W, config::WINDOW_H),
                    // Centrada: dejandolo al gestor de ventanas, el recinto
                    // puede acabar parcialmente fuera de la pantalla.
                    position: WindowPosition::Centered(MonitorSelection::Primary),
                    ..default()
                }),
                ..default()
            }),
            FrameTimeDiagnosticsPlugin::default(),
            RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(config::PIXELS_PER_METER),
            // Arranca apagado: los cuerpos ya se ven por su malla, y los
            // contornos de los colliders son para depurar (F3).
            RapierDebugRenderPlugin {
                enabled: false,
                ..default()
            },
        ))
        .insert_resource(ClearColor(config::BACKGROUND))
        .init_resource::<ThermalField>()
        .insert_resource(BackendKind::from_args())
        .init_resource::<coupling::FieldStats>()
        .init_resource::<coupling::BrownianRng>()
        .init_resource::<spawn::SpawnSettings>()
        .add_systems(
            Startup,
            (
                setup_camera,
                coupling::setup_backends,
                physics::setup_gravity,
                physics::setup_arena,
                physics::spawn_initial_balls,
                viz::setup_field_texture,
                hud::setup_hud,
            ),
        )
        .add_systems(
            Update,
            (
                // Interaccion.
                spawn::shape_input,
                spawn::update_ghost,
                spawn::spawn_on_click,
                spawn::despawn_on_click,
                spawn::reset_scene,
                coupling::cycle_backend,
                toggle_debug_render.run_if(input_just_pressed(KeyCode::F3)),
            ),
        )
        .add_systems(
            Update,
            // El campo va encadenado: los cuerpos lo calientan, difunde, y el
            // resultado los sacude. Encadenarlo evita leer un campo a medio
            // actualizar y hace la simulacion reproducible.
            (
                coupling::deposit_heat,
                coupling::heat_brush,
                coupling::step_field,
                coupling::apply_brownian_impulse,
                viz::update_field_texture,
            )
                .chain(),
        )
        .add_systems(Update, hud::update_hud)
        .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn toggle_debug_render(mut debug: ResMut<DebugRenderContext>) {
    debug.enabled = !debug.enabled;
}
