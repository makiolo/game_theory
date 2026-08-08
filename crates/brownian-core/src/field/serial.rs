//! Backend de referencia: una sola hebra recorriendo la rejilla fila a fila.
//!
//! Es deliberadamente ingenuo. Su papel es doble: servir de baseline en la
//! comparativa y de oraculo de correccion en los tests de los demas backends.

use ndarray::{Array2, Axis};

use super::{DiffusionParams, FieldBackend, diffuse_row};

#[derive(Default)]
pub struct SerialBackend;

impl FieldBackend for SerialBackend {
    fn name(&self) -> &'static str {
        "serial"
    }

    fn diffuse(&mut self, src: &Array2<f32>, dst: &mut Array2<f32>, p: &DiffusionParams) {
        debug_assert_eq!(src.dim(), dst.dim());

        for (y, mut row) in dst.axis_iter_mut(Axis(0)).enumerate() {
            let row = row.as_slice_mut().expect("filas contiguas");
            diffuse_row(src, row, y, p);
        }
    }
}
