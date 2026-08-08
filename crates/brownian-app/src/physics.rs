//! Recinto estatico y cuerpos iniciales.

use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

use crate::config;
use crate::shapes::ShapeKind;
use crate::spawn::spawn_body;

/// Marca los muros del recinto para poder distinguirlos de los cuerpos del usuario.
#[derive(Component)]
pub struct Wall;

/// Baja la gravedad al nivel en el que la agitacion termica puede competir
/// con el peso. Ver [`config::GRAVITY_SCALE`].
pub fn setup_gravity(mut rapier_config: Query<&mut RapierConfiguration>) -> Result<()> {
    let mut rapier_config = rapier_config.single_mut()?;
    rapier_config.gravity.y *= config::GRAVITY_SCALE;
    Ok(())
}

/// Cuatro colliders estaticos formando una caja cerrada alrededor del origen.
pub fn setup_arena(mut commands: Commands) {
    let hw = config::ARENA_HALF_W;
    let hh = config::ARENA_HALF_H;
    let t = config::WALL_THICKNESS;

    // (posicion, semi-extension) de suelo, techo, muro izquierdo y muro derecho.
    let walls = [
        (Vec2::new(0.0, -hh - t), Vec2::new(hw + t, t)),
        (Vec2::new(0.0, hh + t), Vec2::new(hw + t, t)),
        (Vec2::new(-hw - t, 0.0), Vec2::new(t, hh + t)),
        (Vec2::new(hw + t, 0.0), Vec2::new(t, hh + t)),
    ];

    for (pos, half) in walls {
        commands.spawn((
            Wall,
            // Sin malla los muros serian invisibles: el sprite solo esta para
            // que se vea donde termina el recinto.
            Sprite {
                color: Color::srgb(0.22, 0.23, 0.30),
                custom_size: Some(half * 2.0),
                ..default()
            },
            Transform::from_xyz(pos.x, pos.y, 0.0),
            Collider::cuboid(half.x, half.y),
            Restitution::coefficient(config::BODY_RESTITUTION),
        ));
    }
}

/// Unas cuantas pelotas para que al arrancar ya haya algo cayendo.
pub fn spawn_initial_balls(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for i in 0..12 {
        let x = -300.0 + i as f32 * 55.0;
        let y = 200.0 + (i % 3) as f32 * 70.0;

        spawn_body(
            &mut commands,
            &mut meshes,
            &mut materials,
            ShapeKind::Ball,
            18.0,
            Vec2::new(x, y),
        );
    }
}
