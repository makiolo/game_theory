//! Contornos de los colliders (tecla F3).
//!
//! Lo daba el `RapierDebugRenderPlugin`. Al salir `bevy_rapier2d` de la app hubo
//! que rehacerlo, y se dibuja con gizmos a partir de las poses que publica el
//! `Env`. El contorno de cada forma sale de los mismos parametros que su
//! collider —el del triangulo, literalmente de los mismos vertices—, asi que lo
//! que se ve no puede desviarse de lo que colisiona.

use bevy::prelude::*;

use crate::config;
use crate::res::{DebugRender, Sim};
use crate::shapes::ShapeVisuals;

const BODY_OUTLINE: Color = Color::srgb(0.2, 1.0, 0.4);
const WALL_OUTLINE: Color = Color::srgb(1.0, 0.35, 0.35);

pub fn toggle_debug_render(mut debug: ResMut<DebugRender>) {
    **debug = !**debug;
}

pub fn draw_colliders(debug: Res<DebugRender>, sim: Res<Sim>, mut gizmos: Gizmos) {
    if !**debug {
        return;
    }

    for pose in sim.agent_poses() {
        let rotation = Mat2::from_angle(pose.angle);
        gizmos.linestrip_2d(
            pose.shape
                .outline(pose.size)
                .into_iter()
                .map(|p| pose.position + rotation * p),
            BODY_OUTLINE,
        );
    }

    let hw = config::ARENA_HALF_W;
    let hh = config::ARENA_HALF_H;
    let t = config::WALL_THICKNESS;
    let walls = [
        (Vec2::new(0.0, -hh - t), Vec2::new(hw + t, t)),
        (Vec2::new(0.0, hh + t), Vec2::new(hw + t, t)),
        (Vec2::new(-hw - t, 0.0), Vec2::new(t, hh + t)),
        (Vec2::new(hw + t, 0.0), Vec2::new(t, hh + t)),
    ];

    for (centre, half) in walls {
        gizmos.linestrip_2d(
            [
                centre + Vec2::new(-half.x, -half.y),
                centre + Vec2::new(half.x, -half.y),
                centre + Vec2::new(half.x, half.y),
                centre + Vec2::new(-half.x, half.y),
                centre + Vec2::new(-half.x, -half.y),
            ],
            WALL_OUTLINE,
        );
    }
}
