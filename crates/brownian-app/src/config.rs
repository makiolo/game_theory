//! Constantes de presentacion e interaccion.
//!
//! Las de la simulacion viven en [`brownian_core::config`] y se reexportan aqui
//! para que el resto de la app siga leyendo todo bajo un unico `config::`, sin
//! tener que saber de que lado de la frontera cae cada constante.

pub use brownian_core::config::*;

use bevy::prelude::Color;

pub const WINDOW_W: u32 = 1280;
pub const WINDOW_H: u32 = 720;

pub const BACKGROUND: Color = Color::srgb(0.05, 0.05, 0.08);

/// Calor que inyecta el boton central del raton, por segundo.
pub const HEAT_BRUSH: f32 = 400.0;
