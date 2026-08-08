//! Reloj de la simulacion y ruido reproducible.
//!
//! El sandbox se conformaba con un `ChaCha8Rng` compartido: daba igual en que
//! orden se sirviera a los cuerpos mientras una partida entera fuese repetible.
//! Un entorno de aprendizaje no se conforma con eso. Necesita poder reproducir
//! un paso concreto, y necesita que repartir los agentes entre hebras no cambie
//! lo que sale — algo imposible con un flujo secuencial, donde el n-esimo
//! numero depende de cuantos se hayan pedido antes.
//!
//! Aqui el ruido no se consume de un flujo, se *direcciona*: la sacudida del
//! agente `slot` en el paso `step` es una funcion pura de
//! `(seed, step, slot, stream)`. Ni el orden de iteracion ni el numero de hebras
//! entran en la formula, asi que la trayectoria es la misma en serie que en
//! paralelo. Es lo que hace posible la vectorizacion de la fase 5 sin tocar la
//! fisica.

use std::f32::consts::TAU;

use crate::config;

/// Flujos independientes de ruido.
///
/// Separan usos que no deben correlacionarse entre si: sin esto, dos magnitudes
/// derivadas del mismo agente y paso saldrian de la misma palabra y quedarian
/// ligadas.
pub mod stream {
    /// Direccion de la sacudida browniana.
    pub const IMPULSE_ANGLE: u32 = 0;
}

/// Paso actual de la simulacion y semilla del episodio.
///
/// El contador es la coordenada temporal del ruido, asi que tiene que avanzar
/// exactamente una vez por paso de simulacion, ni mas ni menos.
#[derive(Debug, Clone, Copy)]
pub struct SimClock {
    pub seed: u64,
    pub step: u64,
}

impl Default for SimClock {
    fn default() -> Self {
        Self {
            seed: config::RNG_SEED,
            step: 0,
        }
    }
}

impl SimClock {
    /// Arranca un episodio nuevo. Reiniciar el contador es parte del reset: si
    /// siguiera creciendo, dos episodios con la misma semilla no coincidirian.
    pub fn reset(&mut self, seed: u64) {
        self.seed = seed;
        self.step = 0;
    }

    /// Avanza al paso siguiente. Va al principio del paso de simulacion.
    pub fn advance(&mut self) {
        self.step = self.step.wrapping_add(1);
    }
}

/// Mezclador de SplitMix64.
///
/// Tres rondas de xor-shift y multiplicacion bastan para avalancha completa:
/// cambiar un bit de la entrada cambia la mitad de los de salida. Sin estado y
/// unos pocos ciclos, que es lo que hace falta para llamarlo una vez por agente
/// y paso.
#[inline]
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Palabra de 64 bits para la coordenada `(seed, step, slot, stream)`.
#[inline]
pub fn noise(seed: u64, step: u64, slot: u32, stream: u32) -> u64 {
    // Cada mitad de la coordenada se mezcla antes de combinarse. Combinarlas en
    // crudo dejaria colisiones evidentes: `step+1, slot` y `step, slot+1`
    // acabarian en la misma palabra, y con ellas dos agentes vecinos recibiendo
    // el ruido de otro un paso mas tarde.
    let time = mix(seed ^ step.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let space = mix(((slot as u64) << 32) | stream as u64);
    mix(time ^ space)
}

/// El mismo ruido llevado a `[0, 1)`.
#[inline]
pub fn noise_unit(seed: u64, step: u64, slot: u32, stream: u32) -> f32 {
    // 24 bits, los que caben en la mantisa de un f32, y tomados de la mitad
    // alta: en cualquier generador multiplicativo los bits bajos son los peores.
    const SCALE: f32 = 1.0 / (1u32 << 24) as f32;
    (noise(seed, step, slot, stream) >> 40) as f32 * SCALE
}

/// Angulo uniforme en `[0, TAU)`.
#[inline]
pub fn noise_angle(seed: u64, step: u64, slot: u32, stream: u32) -> f32 {
    noise_unit(seed, step, slot, stream) * TAU
}

#[cfg(test)]
mod tests {
    use super::*;

    /// La propiedad que justifica todo el modulo: el ruido de un agente no
    /// depende de cuando se le pregunte ni de a cuantos se haya preguntado
    /// antes. Es lo que rompia el flujo secuencial y lo que permitira sacudir a
    /// los agentes desde varias hebras sin desviar la trayectoria.
    #[test]
    fn noise_is_independent_of_query_order() {
        let (seed, step) = (0xABCD_1234, 42);
        let forward: Vec<u64> = (0..64).map(|s| noise(seed, step, s, 0)).collect();
        let backward: Vec<u64> = (0..64).rev().map(|s| noise(seed, step, s, 0)).collect();

        for (slot, value) in backward.iter().rev().enumerate() {
            assert_eq!(*value, forward[slot], "el slot {slot} cambio con el orden");
        }
    }

    /// Ninguna de las cuatro coordenadas puede quedar sin efecto: si `stream` no
    /// entrase en la mezcla, dos usos distintos del mismo agente y paso saldrian
    /// perfectamente correlacionados.
    #[test]
    fn every_coordinate_changes_the_result() {
        let base = noise(1, 1, 1, 1);
        assert_ne!(base, noise(2, 1, 1, 1), "la semilla no entra en la mezcla");
        assert_ne!(base, noise(1, 2, 1, 1), "el paso no entra en la mezcla");
        assert_ne!(base, noise(1, 1, 2, 1), "el slot no entra en la mezcla");
        assert_ne!(base, noise(1, 1, 1, 2), "el flujo no entra en la mezcla");
    }

    /// Un desplazamiento del paso contra uno del slot no debe colisionar: son
    /// justo las dos direcciones en las que se recorre la coordenada.
    #[test]
    fn step_and_slot_do_not_alias() {
        for k in 1..1000u64 {
            assert_ne!(noise(7, k, 3, 0), noise(7, k - 1, 4, 0));
            assert_ne!(noise(7, k, 3, 0), noise(7, k + 1, 2, 0));
        }
    }

    /// El impulso browniano necesita un angulo sin sesgo; un generador que se
    /// quedase en media rueda dejaria a los cuerpos derivando en esa direccion.
    #[test]
    fn unit_noise_is_uniform() {
        const N: u32 = 100_000;
        let mut sum = 0.0f64;
        let mut buckets = [0u32; 10];

        for slot in 0..N {
            let u = noise_unit(0xDEAD_BEEF, 1, slot, 0);
            assert!((0.0..1.0).contains(&u), "fuera de rango: {u}");
            sum += u as f64;
            buckets[(u * 10.0) as usize] += 1;
        }

        let mean = sum / N as f64;
        assert!((mean - 0.5).abs() < 0.01, "media sesgada: {mean}");

        // Con 100k muestras en 10 cubos, la desviacion tipica relativa es del
        // 1%; un 15% de margen no delata ruido bueno y si delata uno roto.
        let expected = N as f64 / 10.0;
        for (i, count) in buckets.iter().enumerate() {
            let deviation = (*count as f64 - expected).abs() / expected;
            assert!(deviation < 0.15, "cubo {i} desviado un {:.1}%", deviation * 100.0);
        }
    }
}
