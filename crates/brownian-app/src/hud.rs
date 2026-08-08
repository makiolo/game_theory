//! Panel de estado: forma activa, backend del campo y coste del paso.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

use crate::res::{Backend, Sim};
use crate::sim::{Backends, FieldStats};
use crate::spawn::SpawnSettings;

#[derive(Component)]
pub struct HudText;

pub fn setup_hud(mut commands: Commands) {
    commands.spawn((
        HudText,
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(15.0),
            ..default()
        },
        TextColor(Color::srgb(0.85, 0.87, 0.95)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

pub fn update_hud(
    settings: Res<SpawnSettings>,
    kind: Res<Backend>,
    backends: Res<Backends>,
    stats: Res<FieldStats>,
    sim: Res<Sim>,
    diagnostics: Res<DiagnosticsStore>,
    mut text: Query<&mut Text, With<HudText>>,
) {
    let Ok(mut text) = text.single_mut() else {
        return;
    };

    let field = &sim.field;
    let count = sim.agent_count();
    let mean_speed = sim.mean_speed();

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);

    **text = format!(
        "forma: {}  (flechas izq/der)\n\
         tamano: {:.0} px  (flechas arr/aba)\n\
         cuerpos: {}   vel. media: {:.0} px/s\n\
         \n\
         campo: {}x{} = {} celdas\n\
         backend: {}  (B para cambiar)\n\
         paso del campo: {:.2} ms\n\
         temp. media: {:.2}  (fondo {:.2})\n\
         fps: {:.0}\n\
         \n\
         raton izq: crear   raton der: borrar\n\
         raton central: calentar\n\
         R: reiniciar   F3: colliders",
        settings.shape.label(),
        settings.size,
        count,
        mean_speed,
        field.width,
        field.height,
        field.width * field.height,
        backends.name(**kind),
        stats.avg_step_ms,
        field.mean_temperature(),
        field.ambient,
        fps,
    );
}
