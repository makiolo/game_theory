//! brownian — sandbox de agentes con fisica de contacto sobre un medio termico.
//!
//! Este binario es la parte visible: ventana, entrada del raton, mallas y HUD.
//! La simulacion —cuerpos rigidos sobre `rapier2d` y campo de temperatura— vive
//! entera en [`brownian_core`], y aqui solo se le dice cuando dar un paso y se
//! dibuja el resultado.
//!
//! El unico backend del campo que se queda en la app es el de GPU, porque
//! necesita el dispositivo que abre Bevy.

// Los sistemas de Bevy declaran sus dependencias como parametros, asi que pasar
// de siete es lo normal y no dice nada sobre su complejidad.
#![allow(clippy::too_many_arguments)]

mod config;
mod debug;
mod gpu;
mod hud;
mod physics;
mod res;
mod shapes;
mod sim;
mod spawn;
mod viz;

use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::input::common_conditions::input_just_pressed;
use bevy::prelude::*;
use bevy::window::{MonitorSelection, WindowPosition, WindowResolution};

use res::{Backend, DebugRender, Sim};

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
        ))
        .insert_resource(ClearColor(config::BACKGROUND))
        // El tick fijo tiene que valer lo mismo que el paso con el que integran
        // el campo y la fisica; si no, el tiempo simulado se separaria del real.
        .insert_resource(Time::<Fixed>::from_hz(1.0 / config::SIM_DT as f64))
        .init_resource::<Sim>()
        .init_resource::<Backend>()
        .init_resource::<DebugRender>()
        .init_resource::<sim::FieldStats>()
        .init_resource::<spawn::SpawnSettings>()
        .add_systems(
            Startup,
            (
                setup_camera,
                sim::setup_backends,
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
                sim::cycle_backend,
                debug::toggle_debug_render.run_if(input_just_pressed(KeyCode::F3)),
            ),
        )
        .add_systems(
            FixedUpdate,
            // Un paso de simulacion, encadenado: el pincel deposita antes de que
            // el campo difunda, el mundo avanza, y lo que salga se lleva a los
            // `Transform` y a la textura. Encadenarlo evita dibujar un estado a
            // medio actualizar.
            (
                sim::heat_brush,
                sim::step_sim,
                spawn::sync_transforms,
                viz::update_field_texture,
            )
                .chain(),
        )
        .add_systems(Update, (hud::update_hud, debug::draw_colliders))
        .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
