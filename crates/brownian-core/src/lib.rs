//! La simulacion de brownian, sin nada delante.
//!
//! Aqui vive lo que hace avanzar el mundo —el campo termico, el ruido, y (en
//! adelante) la fisica de contacto sobre `rapier2d`— separado de como se
//! dibuja. La separacion no es estetica: un entorno de aprendizaje necesita
//! poder correr esta simulacion miles de veces en paralelo, sin ventana, sin
//! GPU y sin el reloj de un renderer decidiendo el paso de tiempo.
//!
//! Lo que consume este crate es la app de escritorio hoy, y el binding de
//! Python manana. Ninguno de los dos aparece entre sus dependencias.

pub mod config;
pub mod env;
pub mod field;
pub mod shapes;
pub mod sim;

pub use env::{AgentPose, Env};
pub use shapes::ShapeKind;
pub use field::{BackendKind, DiffusionParams, FieldBackend, ThermalField};
pub use sim::SimClock;
