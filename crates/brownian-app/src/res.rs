//! La simulacion, envuelta como recurso de Bevy.
//!
//! `brownian-core` no conoce a Bevy, asi que sus tipos no pueden derivar
//! `Resource`: la regla del huerfano lo impide y, sobre todo, seria justo la
//! dependencia que la separacion existe para evitar. El puente es este
//! envoltorio. `Deref` lo hace transparente, de modo que los sistemas siguen
//! llamando a los metodos de core como si el tipo fuese suyo.
//!
//! Aqui vive tambien la lectura de la linea de ordenes: elegir rejilla y backend
//! al arrancar es cosa de un programa, no de una libreria de simulacion.

use bevy::prelude::*;
use brownian_core::{BackendKind, Env, config};

/// El mundo: recinto, cuerpos y campo termico.
#[derive(Resource, Deref, DerefMut)]
pub struct Sim(pub Env);

impl Default for Sim {
    fn default() -> Self {
        let (w, h) = grid_size_from_args();
        Self(Env::new(w, h, config::RNG_SEED))
    }
}

/// Backend activo del campo; se cicla en caliente con la tecla B.
#[derive(Resource, Deref, DerefMut)]
pub struct Backend(pub BackendKind);

impl Default for Backend {
    fn default() -> Self {
        Self(backend_from_args())
    }
}

/// Si se dibujan los contornos de los colliders (tecla F3).
#[derive(Resource, Default, Deref, DerefMut)]
pub struct DebugRender(pub bool);

/// Tamano de rejilla, de `--grid <ancho>x<alto>` si se pasa.
fn grid_size_from_args() -> (usize, usize) {
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg != "--grid" {
            continue;
        }
        let Some(spec) = args.next() else { break };
        let parsed = spec.split_once(['x', 'X']).and_then(|(w, h)| {
            let w = w.trim().parse::<usize>().ok()?;
            let h = h.trim().parse::<usize>().ok()?;
            (w >= 2 && h >= 2).then_some((w, h))
        });
        match parsed {
            Some(dims) => return dims,
            None => warn!("rejilla invalida: {spec}; se espera algo como 1024x576"),
        }
    }

    (config::GRID_W, config::GRID_H)
}

/// Backend inicial, de `--backend <serial|rayon|gpu>` si se pasa. Permite
/// arrancar directamente en uno concreto para medirlo, sin ciclar con la tecla.
fn backend_from_args() -> BackendKind {
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg != "--backend" {
            continue;
        }
        let Some(name) = args.next() else { break };
        match BackendKind::parse(&name) {
            Some(kind) => return kind,
            None => warn!("backend desconocido: {name}; se usa el de por defecto"),
        }
    }

    BackendKind::default()
}
