//! Las formas que el usuario puede crear, y su traduccion a collider y a malla.

use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use std::f32::consts::FRAC_PI_2;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ShapeKind {
    #[default]
    Ball,
    Square,
    Triangle,
    Capsule,
}

impl ShapeKind {
    pub const ALL: [ShapeKind; 4] = [
        ShapeKind::Ball,
        ShapeKind::Square,
        ShapeKind::Triangle,
        ShapeKind::Capsule,
    ];

    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|s| *s == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }

    pub fn prev(self) -> Self {
        let i = Self::ALL.iter().position(|s| *s == self).unwrap_or(0);
        Self::ALL[(i + Self::ALL.len() - 1) % Self::ALL.len()]
    }

    pub fn label(self) -> &'static str {
        match self {
            ShapeKind::Ball => "circulo",
            ShapeKind::Square => "cuadrado",
            ShapeKind::Triangle => "triangulo",
            ShapeKind::Capsule => "capsula",
        }
    }

    /// Vertices del triangulo equilatero de circunradio `size`, con la punta
    /// arriba. Es el mismo criterio que usa la malla de [`RegularPolygon`], asi
    /// que collider y dibujo coinciden.
    fn triangle_vertices(size: f32) -> [Vec2; 3] {
        std::array::from_fn(|i| {
            let angle = FRAC_PI_2 + i as f32 * std::f32::consts::TAU / 3.0;
            Vec2::new(size * angle.cos(), size * angle.sin())
        })
    }

    /// `size` es el radio caracteristico: el collider queda inscrito en un
    /// circulo de ese radio, de modo que todas las formas se sienten del mismo
    /// tamano al cambiar entre ellas.
    pub fn collider(self, size: f32) -> Collider {
        match self {
            ShapeKind::Ball => Collider::ball(size),
            ShapeKind::Square => Collider::cuboid(size, size),
            ShapeKind::Triangle => {
                let [a, b, c] = Self::triangle_vertices(size);
                Collider::triangle(a, b, c)
            }
            // Media altura del tramo recto + radio = `size` de punta a punta.
            ShapeKind::Capsule => Collider::capsule_y(size * 0.5, size * 0.5),
        }
    }

    pub fn mesh(self, size: f32) -> Mesh {
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

    pub fn color(self) -> Color {
        match self {
            ShapeKind::Ball => Color::srgb(0.35, 0.75, 1.0),
            ShapeKind::Square => Color::srgb(1.0, 0.72, 0.3),
            ShapeKind::Triangle => Color::srgb(0.55, 1.0, 0.5),
            ShapeKind::Capsule => Color::srgb(1.0, 0.45, 0.7),
        }
    }
}
