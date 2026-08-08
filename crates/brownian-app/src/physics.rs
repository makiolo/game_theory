//! Lo visible del recinto, y los cuerpos con los que arranca la escena.
//!
//! Los cuatro colliders que cierran el recinto los monta el propio `Env` al
//! construirse. Aqui solo quedan los sprites que los hacen visibles: sin ellos
//! el recinto existiria igual, pero no se veria donde termina.

use bevy::prelude::*;
use brownian_core::ShapeKind;

use crate::config;
use crate::res::Sim;
use crate::spawn::spawn_body;

pub fn setup_arena(mut commands: Commands) {
    let hw = config::ARENA_HALF_W;
    let hh = config::ARENA_HALF_H;
    let t = config::WALL_THICKNESS;

    // Las mismas cuatro posiciones y semi-extensiones que usa `Env::build_arena`.
    let walls = [
        (Vec2::new(0.0, -hh - t), Vec2::new(hw + t, t)),
        (Vec2::new(0.0, hh + t), Vec2::new(hw + t, t)),
        (Vec2::new(-hw - t, 0.0), Vec2::new(t, hh + t)),
        (Vec2::new(hw + t, 0.0), Vec2::new(t, hh + t)),
    ];

    for (pos, half) in walls {
        commands.spawn((
            Sprite {
                color: Color::srgb(0.22, 0.23, 0.30),
                custom_size: Some(half * 2.0),
                ..default()
            },
            Transform::from_xyz(pos.x, pos.y, 0.0),
        ));
    }
}

/// Unas cuantas pelotas para que al arrancar ya haya algo cayendo.
pub fn spawn_initial_balls(
    mut commands: Commands,
    mut sim: ResMut<Sim>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for i in 0..12 {
        let x = -300.0 + i as f32 * 55.0;
        let y = 200.0 + (i % 3) as f32 * 70.0;

        spawn_body(
            &mut commands,
            &mut sim,
            &mut meshes,
            &mut materials,
            ShapeKind::Ball,
            18.0,
            Vec2::new(x, y),
        );
    }
}
