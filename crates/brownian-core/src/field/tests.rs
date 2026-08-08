use glam::Vec2;
use ndarray::Array2;

use super::parallel::RayonBackend;
use super::serial::SerialBackend;
use super::{DiffusionParams, FieldBackend, ThermalField};

fn params() -> DiffusionParams {
    DiffusionParams {
        kx: 0.2,
        ky: 0.2,
        decay: 0.0,
        ambient: 0.0,
    }
}

/// Campo de prueba determinista, sin depender de un RNG.
fn seeded(h: usize, w: usize) -> Array2<f32> {
    Array2::from_shape_fn((h, w), |(y, x)| ((y * 7 + x * 13) % 23) as f32)
}

/// Con bordes reflectantes y sin decaimiento la difusion solo reparte calor:
/// el total tiene que mantenerse. Es la comprobacion de que el laplaciano y las
/// condiciones de contorno son coherentes.
#[test]
fn diffusion_conserves_heat() {
    let mut field = ThermalField::new(64, 48, 320.0, 240.0, 5000.0, 0.0, 0.0);
    field.deposit(Vec2::new(0.0, 0.0), 1000.0);
    field.deposit(Vec2::new(-100.0, 80.0), 500.0);
    let before = field.total_heat();

    let mut backend = SerialBackend;
    for _ in 0..200 {
        field.step(&mut backend, 1.0 / 60.0);
    }

    let after = field.total_heat();
    assert!(
        (after - before).abs() / before < 1e-3,
        "calor no conservado: {before} -> {after}"
    );
}

/// El calor depositado en un punto tiene que acabar repartido por el vecindario,
/// no quedarse clavado en su celda.
#[test]
fn diffusion_spreads_heat() {
    let mut field = ThermalField::new(32, 32, 160.0, 160.0, 5000.0, 0.0, 0.0);
    let center = Vec2::ZERO;
    field.deposit(center, 100.0);
    let peak_before = field.sample(center);

    let mut backend = SerialBackend;
    for _ in 0..30 {
        field.step(&mut backend, 1.0 / 60.0);
    }

    let peak_after = field.sample(center);
    assert!(
        peak_after < peak_before,
        "el pico no ha bajado: {peak_before} -> {peak_after}"
    );
    let neighbour = field.sample(Vec2::new(20.0, 0.0));
    assert!(neighbour > 0.0, "el calor no ha llegado al vecindario");
}

/// Serie y rayon comparten el nucleo por celda, asi que deben coincidir bit a
/// bit: cualquier divergencia delataria una carrera o un reparto mal indexado.
#[test]
fn serial_matches_rayon() {
    let src = seeded(97, 131);
    let p = params();

    let mut dst_serial = Array2::zeros(src.dim());
    let mut dst_rayon = Array2::ones(src.dim());

    SerialBackend.diffuse(&src, &mut dst_serial, &p);
    RayonBackend.diffuse(&src, &mut dst_rayon, &p);

    assert_eq!(dst_serial, dst_rayon);
}

/// Y tambien tras encadenar muchos pasos, donde una diferencia minima se
/// amplificaria.
#[test]
fn serial_matches_rayon_over_time() {
    let mut a = ThermalField::new(80, 60, 400.0, 300.0, 4000.0, 0.5, 0.0);
    let mut b = ThermalField::new(80, 60, 400.0, 300.0, 4000.0, 0.5, 0.0);
    for f in [&mut a, &mut b] {
        f.deposit(Vec2::new(50.0, -40.0), 300.0);
        f.deposit(Vec2::new(-120.0, 100.0), 700.0);
    }

    let mut serial = SerialBackend;
    let mut rayon = RayonBackend;
    for _ in 0..100 {
        a.step(&mut serial, 1.0 / 60.0);
        b.step(&mut rayon, 1.0 / 60.0);
    }

    assert_eq!(a.data(), b.data());
}

#[test]
fn world_to_cell_roundtrip() {
    let field = ThermalField::new(16, 8, 160.0, 80.0, 1.0, 0.0, 0.0);

    // El centro de cada celda tiene que volver a su propio indice.
    for y in 0..field.height {
        for x in 0..field.width {
            let world = field.cell_to_world(x, y);
            assert_eq!(field.world_to_cell(world), Some((x, y)), "celda {x},{y}");
        }
    }

    // Esquinas justo dentro del area.
    assert_eq!(field.world_to_cell(Vec2::new(-159.9, -79.9)), Some((0, 0)));
    assert_eq!(field.world_to_cell(Vec2::new(159.9, 79.9)), Some((15, 7)));

    // Y fuera del area no hay celda, ni por poco ni por mucho.
    assert_eq!(field.world_to_cell(Vec2::new(-160.1, 0.0)), None);
    assert_eq!(field.world_to_cell(Vec2::new(160.1, 0.0)), None);
    assert_eq!(field.world_to_cell(Vec2::new(0.0, 80.1)), None);
    assert_eq!(field.world_to_cell(Vec2::new(0.0, -1000.0)), None);
}

/// Depositar fuera del area no debe colarse en una celda del borde ni entrar en
/// panico: es lo que ocurre cuando un cuerpo se sale del recinto.
#[test]
fn deposit_outside_is_ignored() {
    let mut field = ThermalField::new(16, 8, 160.0, 80.0, 1.0, 0.0, 0.0);
    field.deposit(Vec2::new(1e6, 1e6), 500.0);
    field.deposit(Vec2::new(-1e6, 0.0), 500.0);
    assert_eq!(field.total_heat(), 0.0);
}

/// El bano termico es lo que sostiene el movimiento browniano: un medio frio
/// tiene que calentarse hasta el fondo, y un punto caliente relajarse hasta el
/// mismo sitio. Sin esto la simulacion se apaga sola.
#[test]
fn relaxes_towards_ambient() {
    let ambient = 2.0;
    let mut field = ThermalField::new(32, 32, 160.0, 160.0, 500.0, 4.0, ambient);
    let mut backend = SerialBackend;

    // Un punto muy por encima del fondo tiene que bajar hasta el.
    field.deposit(Vec2::ZERO, 500.0);
    for _ in 0..600 {
        field.step(&mut backend, 1.0 / 60.0);
    }
    let hot = field.sample(Vec2::ZERO);
    assert!(
        (hot - ambient).abs() < 0.05,
        "no ha relajado al fondo: {hot} vs {ambient}"
    );

    // Y un medio arrancado en frio tiene que subir hasta el.
    let mut cold = ThermalField::new(32, 32, 160.0, 160.0, 500.0, 4.0, ambient);
    cold.clear();
    cold.data().iter().for_each(|v| assert_eq!(*v, ambient));

    let mut chilled = ThermalField::new(32, 32, 160.0, 160.0, 500.0, 4.0, 0.0);
    chilled.ambient = ambient;
    for _ in 0..600 {
        chilled.step(&mut backend, 1.0 / 60.0);
    }
    let warmed = chilled.sample(Vec2::ZERO);
    assert!(
        (warmed - ambient).abs() < 0.05,
        "no ha subido al fondo: {warmed} vs {ambient}"
    );
}

/// Comparativa de backends sobre la misma rejilla. No es una asercion, es una
/// medida: se ejecuta a mano con
///
/// ```text
/// cargo test --release -- --ignored --nocapture bench
/// ```
///
/// En debug no dice nada util, porque el nucleo va sin optimizar.
#[test]
#[ignore = "medida de rendimiento, no comprobacion"]
fn bench_backends() {
    use std::time::Instant;

    for (w, h) in [(512, 288), (1024, 576), (2048, 1152)] {
        let mut serial = SerialBackend;
        let mut rayon = RayonBackend;
        let p = params();
        let src = seeded(h, w);
        let mut dst = Array2::zeros((h, w));

        // Una pasada previa para no medir el primer fallo de cache ni el
        // arranque del pool de rayon.
        serial.diffuse(&src, &mut dst, &p);
        rayon.diffuse(&src, &mut dst, &p);

        let rounds = 50;
        let t0 = Instant::now();
        for _ in 0..rounds {
            serial.diffuse(&src, &mut dst, &p);
        }
        let serial_ms = t0.elapsed().as_secs_f64() * 1000.0 / rounds as f64;

        let t1 = Instant::now();
        for _ in 0..rounds {
            rayon.diffuse(&src, &mut dst, &p);
        }
        let rayon_ms = t1.elapsed().as_secs_f64() * 1000.0 / rounds as f64;

        println!(
            "{w:5}x{h:<5} ({:>8} celdas)  serial {serial_ms:6.2} ms   rayon {rayon_ms:6.2} ms   x{:.2}",
            w * h,
            serial_ms / rayon_ms
        );
    }
}

/// El troceado en sub-pasos existe para no salirse del limite de estabilidad;
/// si fallara, el campo divergiria en lugar de relajarse.
#[test]
fn stays_stable_with_large_dt() {
    let mut field = ThermalField::new(64, 64, 320.0, 320.0, 20_000.0, 0.0, 0.0);
    field.deposit(Vec2::ZERO, 1000.0);
    let mut backend = SerialBackend;

    // Un dt enorme, muy por encima del limite explicito para esta difusividad.
    field.step(&mut backend, 0.5);

    let max = field.data().iter().cloned().fold(f32::MIN, f32::max);
    assert!(max.is_finite(), "el campo ha divergido");
    assert!(max <= 1000.0, "el maximo ha crecido: {max}");
}
