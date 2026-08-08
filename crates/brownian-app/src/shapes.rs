//! El aspecto de cada forma: malla, color y contorno.
//!
//! La geometria vive en [`brownian_core::shapes`] — es la que decide como
//! colisiona un cuerpo. Aqui solo esta como se ve, y por eso llega como
//! extension del tipo y no como metodos suyos: core no puede depender de Bevy
//! para nombrar una `Mesh`.

use bevy::prelude::*;
use brownian_core::ShapeKind;
use std::f32::consts::TAU;

/// Segmentos con los que se aproxima un arco al dibujar contornos. Con 24 el
/// circulo ya no se ve poligonal al tamano al que se crean los cuerpos.
const ARC_SEGMENTS: usize = 24;

pub trait ShapeVisuals {
    fn mesh(self, size: f32) -> Mesh;
    fn color(self) -> Color;
    /// Contorno cerrado en coordenadas locales, para dibujarlo con gizmos.
    fn outline(self, size: f32) -> Vec<Vec2>;
}

impl ShapeVisuals for ShapeKind {
    fn mesh(self, size: f32) -> Mesh {
        match self {
            ShapeKind::Ball => Circle::new(size).into(),
            ShapeKind::Square => Rectangle::new(size * 2.0, size * 2.0).into(),
            ShapeKind::Triangle => RegularPolygon {
                circumcircle: Circle::new(size),
                sides: 3,
            }
            .into(),
            ShapeKind::Capsule => Capsule2d {
                radius: size * 0.5,
                half_length: size * 0.5,
            }
            .into(),
        }
    }

    fn color(self) -> Color {
        match self {
            ShapeKind::Ball => Color::srgb(0.35, 0.75, 1.0),
            ShapeKind::Square => Color::srgb(1.0, 0.72, 0.3),
            ShapeKind::Triangle => Color::srgb(0.55, 1.0, 0.5),
            ShapeKind::Capsule => Color::srgb(1.0, 0.45, 0.7),
        }
    }

    fn outline(self, size: f32) -> Vec<Vec2> {
        let mut points = match self {
            ShapeKind::Ball => arc(Vec2::ZERO, size, 0.0, TAU, ARC_SEGMENTS),
            ShapeKind::Square => vec![
                Vec2::new(-size, -size),
                Vec2::new(size, -size),
                Vec2::new(size, size),
                Vec2::new(-size, size),
            ],
            // Los mismos vertices que usa el collider, para que el contorno no
            // pueda desviarse de lo que de verdad colisiona.
            ShapeKind::Triangle => ShapeKind::triangle_vertices(size).to_vec(),
            ShapeKind::Capsule => {
                let (half, radius) = (size * 0.5, size * 0.5);
                let mut p = arc(
                    Vec2::new(0.0, half),
                    radius,
                    0.0,
                    std::f32::consts::PI,
                    ARC_SEGMENTS / 2,
                );
                p.extend(arc(
                    Vec2::new(0.0, -half),
                    radius,
                    std::f32::consts::PI,
                    TAU,
                    ARC_SEGMENTS / 2,
                ));
                p
            }
        };

        // Cerrado: el gizmo dibuja una polilinea, no un poligono.
        if let Some(first) = points.first().copied() {
            points.push(first);
        }
        points
    }
}

/// Puntos de un arco de `centre`, de `from` a `to` radianes.
fn arc(centre: Vec2, radius: f32, from: f32, to: f32, segments: usize) -> Vec<Vec2> {
    (0..=segments)
        .map(|i| {
            let t = from + (to - from) * i as f32 / segments as f32;
            centre + Vec2::new(radius * t.cos(), radius * t.sin())
        })
        .collect()
}
