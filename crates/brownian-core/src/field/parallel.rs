//! Backend de CPU paralelo: las filas de la rejilla se reparten entre hebras
//! con rayon.
//!
//! El reparto por filas es seguro sin sincronizacion porque el paso escribe en
//! `dst` y solo lee de `src`: no hay dos hebras tocando la misma celda ni
//! lecturas de datos que otra hebra este escribiendo.

use ndarray::parallel::prelude::*;
use ndarray::{Array2, Axis};

use super::{DiffusionParams, FieldBackend, diffuse_row};

#[derive(Default)]
pub struct RayonBackend;

impl FieldBackend for RayonBackend {
    fn name(&self) -> &'static str {
        "rayon"
    }

    fn diffuse(&mut self, src: &Array2<f32>, dst: &mut Array2<f32>, p: &DiffusionParams) {
        debug_assert_eq!(src.dim(), dst.dim());

        dst.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                let row = row.as_slice_mut().expect("filas contiguas");
                diffuse_row(src, row, y, p);
            });
    }
}
