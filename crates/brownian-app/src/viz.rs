//! Visualizacion del campo termico como una textura de fondo.
//!
//! La rejilla se vuelca cada frame en una `Image` del tamano exacto de la
//! rejilla, estirada por el sprite hasta cubrir el recinto. El muestreo lo hace
//! la GPU, asi que ver el campo no cuesta relleno por pixel en la CPU.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::config;
use crate::res::Sim;

/// Rango minimo de la escala de color: por debajo de este exceso sobre el fondo
/// no merece la pena estirar el contraste, seria amplificar ruido.
const MIN_RANGE: f32 = 0.5;

#[derive(Resource)]
pub struct FieldTexture {
    handle: Handle<Image>,
    /// Exceso maximo sobre el fondo, suavizado. La escala se adapta sola para
    /// que siempre haya contraste sin saturar cuando algo calienta mucho.
    scale: f32,
}

pub fn setup_field_texture(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    sim: Res<Sim>,
) {
    let field = &sim.field;
    let mut image = Image::new_fill(
        Extent3d {
            width: field.width as u32,
            height: field.height as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        // Se escribe desde el main world cada frame, asi que tiene que quedarse
        // ahi ademas de subirse a la GPU.
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    // Interpolacion lineal: el campo es continuo, no queremos ver los texels.
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());

    let handle = images.add(image);

    commands.spawn((
        Sprite {
            image: handle.clone(),
            custom_size: Some(Vec2::new(
                config::ARENA_HALF_W * 2.0,
                config::ARENA_HALF_H * 2.0,
            )),
            ..default()
        },
        // Detras de todo lo demas.
        Transform::from_xyz(0.0, 0.0, -10.0),
    ));

    commands.insert_resource(FieldTexture {
        handle,
        scale: MIN_RANGE,
    });
}

pub fn update_field_texture(
    sim: Res<Sim>,
    mut texture: ResMut<FieldTexture>,
    mut images: ResMut<Assets<Image>>,
) {
    let field = &sim.field;
    let data = field.data();
    let ambient = field.ambient;

    // Lo que se pinta es el exceso sobre el bano termico. Normalizar contra el
    // valor absoluto dejaria el fondo saturado en el tope de la rampa y los
    // puntos calientes, invisibles.
    let peak_excess = data.iter().copied().fold(0.0f32, f32::max) - ambient;

    // La escala sube rapido para no saturar y baja despacio para no parpadear.
    let target = peak_excess.max(MIN_RANGE);
    texture.scale = if target > texture.scale {
        target
    } else {
        texture.scale * 0.98 + target * 0.02
    };
    let inv_scale = 1.0 / texture.scale;

    let handle = texture.handle.clone();
    let Some(mut image) = images.get_mut(&handle) else {
        return;
    };
    let Some(bytes) = image.data.as_mut() else {
        return;
    };

    let (h, w) = (field.height, field.width);
    for y in 0..h {
        // La textura crece hacia abajo y la rejilla hacia arriba.
        let row = h - 1 - y;
        for x in 0..w {
            let t = ((data[[y, x]] - ambient) * inv_scale).clamp(0.0, 1.0);
            let [r, g, b] = heat_color(t);
            let i = (row * w + x) * 4;
            bytes[i] = r;
            bytes[i + 1] = g;
            bytes[i + 2] = b;
            bytes[i + 3] = 255;
        }
    }
}

/// Rampa tipo "inferno": negro -> purpura -> naranja -> amarillo.
#[inline]
fn heat_color(t: f32) -> [u8; 3] {
    const STOPS: [[f32; 3]; 4] = [
        [0.02, 0.02, 0.10],
        [0.45, 0.08, 0.42],
        [0.95, 0.42, 0.10],
        [1.00, 0.98, 0.72],
    ];

    let scaled = t * (STOPS.len() - 1) as f32;
    let i = (scaled.floor() as usize).min(STOPS.len() - 2);
    let f = scaled - i as f32;

    let mut out = [0u8; 3];
    for c in 0..3 {
        let v = STOPS[i][c] + (STOPS[i + 1][c] - STOPS[i][c]) * f;
        out[c] = (v.clamp(0.0, 1.0) * 255.0) as u8;
    }
    out
}
