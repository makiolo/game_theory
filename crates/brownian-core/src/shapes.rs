//! Las formas que puede tener un agente, y su traduccion a collider.
//!
//! Aqui solo esta la geometria. Como se dibuja cada una —malla y color— es cosa
//! de quien tenga una pantalla delante, y vive en la app.

use glam::Vec2;
use rapier2d::prelude::ColliderBuilder;
use std::f32::consts::{FRAC_PI_2, TAU};

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
    /// arriba. Es el mismo criterio que usa la malla de `RegularPolygon`, asi
    /// que collider y dibujo coinciden.
    pub fn triangle_vertices(size: f32) -> [Vec2; 3] {
        std::array::from_fn(|i| {
            let angle = FRAC_PI_2 + i as f32 * TAU / 3.0;
            Vec2::new(size * angle.cos(), size * angle.sin())
        })
    }

    /// `size` es el radio caracteristico: el collider queda inscrito en un
    /// circulo de ese radio, de modo que todas las formas se sienten del mismo
    /// tamano al cambiar entre ellas.
    ///
    /// Devuelve el constructor a medio montar, no el collider ya cerrado: quien
    /// crea el cuerpo es el que sabe con que rebote y rozamiento quiere hacerlo.
    pub fn collider(self, size: f32) -> ColliderBuilder {
        match self {
            ShapeKind::Ball => ColliderBuilder::ball(size),
            ShapeKind::Square => ColliderBuilder::cuboid(size, size),
            ShapeKind::Triangle => {
                let [a, b, c] = Self::triangle_vertices(size);
                ColliderBuilder::triangle(a, b, c)
            }
            // Media altura del tramo recto + radio = `size` de punta a punta.
            ShapeKind::Capsule => ColliderBuilder::capsule_y(size * 0.5, size * 0.5),
        }
    }
}
