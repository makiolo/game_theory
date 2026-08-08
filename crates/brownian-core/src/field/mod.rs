//! Rejilla 2D de temperatura: el medio en el que se agitan los cuerpos.
//!
//! El campo evoluciona con difusion explicita mas decaimiento,
//!
//! ```text
//! T' = T + kx*(T[x-1] + T[x+1] - 2T) + ky*(T[y-1] + T[y+1] - 2T) - decay*T
//! ```
//!
//! Los bordes son reflectantes (Neumann homogeneo, implementado clampeando el
//! indice del vecino): el flujo a traves de la pared es nulo, asi que sin
//! decaimiento el calor total se conserva. Esa propiedad es lo que verifica el
//! test `diffusion_conserves_heat`.
//!
//! El recorrido de la rejilla es intercambiable ([`FieldBackend`]): serie,
//! rayon o GPU. Todos calculan exactamente la misma celda con la misma formula
//! ([`diffuse_row`]), de modo que la comparativa mide el paralelismo y no
//! diferencias de modelo.

pub mod gpu;
pub mod parallel;
pub mod serial;

#[cfg(test)]
mod tests;

use bevy::prelude::*;
use ndarray::Array2;

use crate::config;

/// Coeficientes ya adimensionalizados de un sub-paso de difusion.
#[derive(Clone, Copy, Debug)]
pub struct DiffusionParams {
    /// `alpha * dt / dx^2`
    pub kx: f32,
    /// `alpha * dt / dy^2`
    pub ky: f32,
    /// `decay * dt`
    pub decay: f32,
    /// Temperatura del bano hacia la que relaja el termino de decaimiento.
    pub ambient: f32,
}

impl DiffusionParams {
    /// El esquema explicito es estable mientras `kx + ky <= 1/2`. Devuelve en
    /// cuantos sub-pasos hay que trocear `dt` para respetarlo.
    pub fn substeps_for(alpha: f32, dx: f32, dy: f32, dt: f32) -> u32 {
        let k = alpha * dt * (1.0 / (dx * dx) + 1.0 / (dy * dy));
        (k / 0.5).ceil().max(1.0) as u32
    }
}

/// Cualquier forma de recorrer la rejilla para aplicar un sub-paso de difusion.
pub trait FieldBackend: Send + Sync {
    fn name(&self) -> &'static str;

    /// Calcula `dst` completo a partir de `src`. No debe leer `dst`.
    fn diffuse(&mut self, src: &Array2<f32>, dst: &mut Array2<f32>, p: &DiffusionParams);

    /// Encadena `substeps` sub-pasos, dejando el resultado en `front`.
    ///
    /// Existe como metodo aparte porque no todos los backends quieren el mismo
    /// ping-pong: en CPU encadenar `diffuse` es exactamente igual de rapido,
    /// pero un backend de GPU necesita hacer todos los sub-pasos en la tarjeta
    /// y transferir una sola vez, no una por sub-paso.
    fn run(
        &mut self,
        front: &mut Array2<f32>,
        back: &mut Array2<f32>,
        p: &DiffusionParams,
        substeps: u32,
    ) {
        for _ in 0..substeps {
            self.diffuse(front, back, p);
            std::mem::swap(front, back);
        }
    }
}

/// Cual de los backends esta activo; se cicla en caliente con una tecla.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BackendKind {
    #[default]
    Serial,
    Rayon,
    Gpu,
}

impl BackendKind {
    pub fn next(self) -> Self {
        match self {
            BackendKind::Serial => BackendKind::Rayon,
            BackendKind::Rayon => BackendKind::Gpu,
            BackendKind::Gpu => BackendKind::Serial,
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "serial" => Some(BackendKind::Serial),
            "rayon" => Some(BackendKind::Rayon),
            "gpu" => Some(BackendKind::Gpu),
            _ => None,
        }
    }

    /// Backend inicial, tomado de `--backend <serial|rayon|gpu>` si se pasa.
    /// Permite arrancar directamente en uno concreto para medirlo, sin tener
    /// que ciclar con la tecla B.
    pub fn from_args() -> Self {
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg != "--backend" {
                continue;
            }
            let Some(name) = args.next() else { break };
            match Self::parse(&name) {
                Some(kind) => return kind,
                None => warn!("backend desconocido: {name}; se usa el de por defecto"),
            }
        }
        Self::default()
    }
}

/// El campo de temperatura y su doble buffer.
#[derive(Resource)]
pub struct ThermalField {
    pub width: usize,
    pub height: usize,
    /// Tamano de celda en pixeles de mundo.
    pub dx: f32,
    pub dy: f32,
    /// Semi-extension del area cubierta, centrada en el origen.
    pub half_w: f32,
    pub half_h: f32,
    pub alpha: f32,
    pub decay: f32,
    /// Temperatura de fondo del medio.
    pub ambient: f32,
    front: Array2<f32>,
    back: Array2<f32>,
}

impl Default for ThermalField {
    fn default() -> Self {
        let (w, h) = grid_size_from_args();
        Self::new(
            w,
            h,
            config::ARENA_HALF_W,
            config::ARENA_HALF_H,
            config::ALPHA,
            config::DECAY,
            config::AMBIENT,
        )
    }
}

/// Tamano de rejilla, de `--grid <ancho>x<alto>` si se pasa.
///
/// Poder cambiarlo sin recompilar es lo que hace practico comparar los backends:
/// con rejillas pequenas la GPU pierde contra la CPU porque domina el coste de
/// subir y bajar los datos, y solo se ve su ventaja al crecer.
fn grid_size_from_args() -> (usize, usize) {
    let default = (config::GRID_W, config::GRID_H);
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

    default
}

impl ThermalField {
    pub fn new(
        width: usize,
        height: usize,
        half_w: f32,
        half_h: f32,
        alpha: f32,
        decay: f32,
        ambient: f32,
    ) -> Self {
        assert!(width >= 2 && height >= 2, "la rejilla necesita 2x2 celdas");
        Self {
            width,
            height,
            dx: 2.0 * half_w / width as f32,
            dy: 2.0 * half_h / height as f32,
            half_w,
            half_h,
            alpha,
            decay,
            ambient,
            // Arranca ya a la temperatura del bano, no frio.
            front: Array2::from_elem((height, width), ambient),
            back: Array2::from_elem((height, width), ambient),
        }
    }

    /// Vista de solo lectura del campo actual (indexada `[[fila, columna]]`).
    pub fn data(&self) -> &Array2<f32> {
        &self.front
    }

    /// Calor total del campo. Util para tests y para el HUD.
    pub fn total_heat(&self) -> f32 {
        self.front.sum()
    }

    /// Temperatura media del medio. Con el bano de fondo el total absoluto no
    /// dice gran cosa, pero la media comparada con `ambient` si: mide cuanto
    /// estan agitando los cuerpos al medio.
    pub fn mean_temperature(&self) -> f32 {
        self.total_heat() / (self.width * self.height) as f32
    }

    /// Celda que contiene un punto de mundo, o `None` si cae fuera del area.
    pub fn world_to_cell(&self, world: Vec2) -> Option<(usize, usize)> {
        let fx = (world.x + self.half_w) / self.dx;
        let fy = (world.y + self.half_h) / self.dy;
        if fx < 0.0 || fy < 0.0 {
            return None;
        }
        let (x, y) = (fx as usize, fy as usize);
        (x < self.width && y < self.height).then_some((x, y))
    }

    /// Centro en coordenadas de mundo de una celda. Es el inverso de
    /// [`Self::world_to_cell`]; de momento solo lo ejercita el test de ida y
    /// vuelta, pero el backend de GPU lo necesitara para situar la rejilla.
    #[allow(dead_code)]
    pub fn cell_to_world(&self, x: usize, y: usize) -> Vec2 {
        Vec2::new(
            (x as f32 + 0.5) * self.dx - self.half_w,
            (y as f32 + 0.5) * self.dy - self.half_h,
        )
    }

    /// Temperatura en un punto de mundo (0 fuera del area).
    pub fn sample(&self, world: Vec2) -> f32 {
        match self.world_to_cell(world) {
            Some((x, y)) => self.front[[y, x]],
            None => 0.0,
        }
    }

    /// Anade calor en la celda que contiene el punto. Fuera del area no hace nada.
    pub fn deposit(&mut self, world: Vec2, amount: f32) {
        if let Some((x, y)) = self.world_to_cell(world) {
            self.front[[y, x]] += amount;
        }
    }

    /// Devuelve el medio a su temperatura de fondo.
    pub fn clear(&mut self) {
        self.front.fill(self.ambient);
        self.back.fill(self.ambient);
    }

    /// Avanza el campo `dt` segundos, troceando en los sub-pasos que haga falta
    /// para no salirse del limite de estabilidad.
    pub fn step(&mut self, backend: &mut dyn FieldBackend, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        let n = DiffusionParams::substeps_for(self.alpha, self.dx, self.dy, dt);
        let sub_dt = dt / n as f32;
        let p = DiffusionParams {
            kx: self.alpha * sub_dt / (self.dx * self.dx),
            ky: self.alpha * sub_dt / (self.dy * self.dy),
            decay: self.decay * sub_dt,
            ambient: self.ambient,
        };

        backend.run(&mut self.front, &mut self.back, &p, n);
    }
}

/// Un sub-paso de difusion sobre la fila `y`, escrito en `dst_row`.
///
/// Es el nucleo compartido por los backends de CPU: al ser exactamente el mismo
/// codigo y el mismo orden de operaciones, serie y rayon coinciden bit a bit.
///
/// Trabaja sobre las tres filas ya resueltas como slices en vez de indexar
/// `src[[y, x]]`: la indexacion 2D de ndarray recalcula el offset por strides en
/// cada acceso, y con cinco accesos por celda eso dominaba el coste del paso.
/// El interior se recorre con `windows(3)`, que ademas elimina la comprobacion
/// de limites; los dos bordes en x se resuelven aparte.
#[inline]
pub fn diffuse_row(src: &Array2<f32>, dst_row: &mut [f32], y: usize, p: &DiffusionParams) {
    let (h, w) = src.dim();
    // Bordes reflectantes: el vecino de fuera es la propia celda del borde.
    let y_up = y.saturating_sub(1);
    let y_dn = (y + 1).min(h - 1);

    let expect = "las filas de un Array2 son contiguas";
    let mid = src.row(y);
    let up = src.row(y_up);
    let dn = src.row(y_dn);
    let mid = mid.as_slice().expect(expect);
    let up = up.as_slice().expect(expect);
    let dn = dn.as_slice().expect(expect);

    #[inline(always)]
    fn cell(c: f32, left: f32, right: f32, above: f32, below: f32, p: &DiffusionParams) -> f32 {
        let lap_x = left + right - 2.0 * c;
        let lap_y = above + below - 2.0 * c;
        // El decaimiento relaja hacia el bano termico, no hacia cero: es lo que
        // mantiene el medio a temperatura de fondo entre choque y choque.
        c + p.kx * lap_x + p.ky * lap_y - p.decay * (c - p.ambient)
    }

    // Columna izquierda: su vecino de la izquierda es ella misma.
    dst_row[0] = cell(mid[0], mid[0], mid[1.min(w - 1)], up[0], dn[0], p);

    // Interior, sin comprobaciones de limites.
    if w > 2 {
        let inner = &mut dst_row[1..w - 1];
        for (i, (d, window)) in inner.iter_mut().zip(mid.windows(3)).enumerate() {
            let x = i + 1;
            *d = cell(window[1], window[0], window[2], up[x], dn[x], p);
        }
    }

    // Columna derecha, con el mismo criterio que la izquierda.
    if w > 1 {
        let last = w - 1;
        dst_row[last] = cell(mid[last], mid[last - 1], mid[last], up[last], dn[last], p);
    }
}
