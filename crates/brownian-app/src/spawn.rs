//! Interaccion con el raton y el teclado: elegir forma, previsualizarla y
//! crear o borrar cuerpos.

use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

use crate::config;
use crate::field::ThermalField;
use crate::physics::Wall;
use crate::shapes::ShapeKind;

/// Marca los cuerpos creados por el usuario: son los que se acoplan al campo
/// termico y los que cuenta el HUD.
#[derive(Component)]
pub struct Agent;

/// La silueta que sigue al cursor mostrando que se va a crear.
#[derive(Component)]
pub struct Ghost;

#[derive(Resource)]
pub struct SpawnSettings {
    pub shape: ShapeKind,
    pub size: f32,
    /// Evita crear un cuerpo por frame mientras se mantiene pulsado el boton.
    pub cooldown: Timer,
}

impl Default for SpawnSettings {
    fn default() -> Self {
        Self {
            shape: ShapeKind::default(),
            size: 18.0,
            cooldown: Timer::from_seconds(0.08, TimerMode::Repeating),
        }
    }
}

pub const MIN_SIZE: f32 = 6.0;
pub const MAX_SIZE: f32 = 60.0;

/// Crea un cuerpo dinamico completo: malla, collider y los componentes que
/// necesita el acoplamiento termico. Es el unico sitio donde se define de que
/// se compone un agente, para que los del arranque y los del raton sean iguales.
pub fn spawn_body(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    shape: ShapeKind,
    size: f32,
    position: Vec2,
) -> Entity {
    commands
        .spawn((
            Agent,
            Mesh2d(meshes.add(shape.mesh(size))),
            MeshMaterial2d(materials.add(shape.color())),
            Transform::from_xyz(position.x, position.y, 1.0),
            RigidBody::Dynamic,
            shape.collider(size),
            Restitution::coefficient(config::BODY_RESTITUTION),
            Friction::coefficient(config::BODY_FRICTION),
            Damping {
                linear_damping: config::LINEAR_DAMPING,
                angular_damping: config::ANGULAR_DAMPING,
            },
            Velocity::zero(),
            ExternalImpulse::default(),
            // El impulso browniano necesita la masa del cuerpo, que Rapier
            // calcula a partir del collider y publica aqui.
            ReadMassProperties::default(),
            // En un bano termico ningun cuerpo esta nunca del todo en reposo:
            // si Rapier los durmiera, dejarian de temblar al asentarse.
            Sleeping::disabled(),
            // Con la agitacion alta los cuerpos pequenos alcanzan velocidades
            // a las que podrian colarse entre muros de un paso al siguiente.
            Ccd::enabled(),
        ))
        .id()
}

/// Posicion del cursor en coordenadas de mundo.
pub fn cursor_world_position(
    window: &Window,
    camera: &Camera,
    camera_transform: &GlobalTransform,
) -> Option<Vec2> {
    let cursor = window.cursor_position()?;
    camera.viewport_to_world_2d(camera_transform, cursor).ok()
}

/// Flechas izquierda/derecha cambian de forma; arriba/abajo, de tamano.
pub fn shape_input(keys: Res<ButtonInput<KeyCode>>, mut settings: ResMut<SpawnSettings>) {
    if keys.just_pressed(KeyCode::ArrowRight) {
        settings.shape = settings.shape.next();
    }
    if keys.just_pressed(KeyCode::ArrowLeft) {
        settings.shape = settings.shape.prev();
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        settings.size = (settings.size * 1.25).min(MAX_SIZE);
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        settings.size = (settings.size / 1.25).max(MIN_SIZE);
    }
}

/// Mantiene la silueta bajo el cursor, reconstruyendola cuando cambia la forma
/// o el tamano y ocultandola si el cursor sale de la ventana.
pub fn update_ghost(
    mut commands: Commands,
    settings: Res<SpawnSettings>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    ghosts: Query<Entity, With<Ghost>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut last: Local<Option<(ShapeKind, f32)>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };

    let existing = ghosts.iter().next();
    let Some(world) = cursor_world_position(window, camera, camera_transform) else {
        // Sin cursor sobre la ventana no hay nada que previsualizar.
        if let Some(entity) = existing {
            commands.entity(entity).despawn();
        }
        return;
    };

    // `settings` se marca como modificado cada frame (el cooldown vive dentro),
    // asi que comparamos a mano contra lo ultimo dibujado: regenerar la malla
    // en cada frame seria tirar memoria sin necesidad.
    let current = (settings.shape, settings.size);
    let stale = last.map(|prev| prev != current).unwrap_or(true);

    if stale || existing.is_none() {
        *last = Some(current);
        if let Some(entity) = existing {
            commands.entity(entity).despawn();
        }
        let mut color = settings.shape.color();
        color.set_alpha(0.3);
        commands.spawn((
            Ghost,
            Mesh2d(meshes.add(settings.shape.mesh(settings.size))),
            MeshMaterial2d(materials.add(color)),
            Transform::from_xyz(world.x, world.y, 5.0),
        ));
    } else if let Some(entity) = existing {
        commands
            .entity(entity)
            .insert(Transform::from_xyz(world.x, world.y, 5.0));
    }
}

/// Boton izquierdo mantenido: crea cuerpos con la forma activa.
pub fn spawn_on_click(
    mut commands: Commands,
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut settings: ResMut<SpawnSettings>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    if !buttons.pressed(MouseButton::Left) {
        return;
    }

    // El primer clic crea al instante; mantenerlo pulsado va soltando cuerpos
    // al ritmo del cooldown.
    if buttons.just_pressed(MouseButton::Left) {
        settings.cooldown.reset();
    } else {
        let delta = time.delta();
        settings.cooldown.tick(delta);
        if !settings.cooldown.just_finished() {
            return;
        }
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let Some(world) = cursor_world_position(window, camera, camera_transform) else {
        return;
    };

    let (shape, size) = (settings.shape, settings.size);
    spawn_body(
        &mut commands,
        &mut meshes,
        &mut materials,
        shape,
        size,
        world,
    );
}

/// Boton derecho: borra el cuerpo bajo el cursor, pero nunca los muros.
pub fn despawn_on_click(
    mut commands: Commands,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    rapier: ReadRapierContext,
    walls: Query<(), With<Wall>>,
) {
    if !buttons.pressed(MouseButton::Right) {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let Some(world) = cursor_world_position(window, camera, camera_transform) else {
        return;
    };
    let Ok(context) = rapier.single() else {
        return;
    };

    // La consulta recorre los colliders por callback, asi que recogemos primero
    // y borramos despues: no conviene tocar el mundo mientras se itera.
    let mut targets = Vec::new();
    context.intersect_point(world, QueryFilter::default(), |entity| {
        if !walls.contains(entity) {
            targets.push(entity);
        }
        true
    });

    for entity in targets {
        commands.entity(entity).despawn();
    }
}

/// Tecla R: deja el recinto como al principio, sin cuerpos y con el medio frio.
/// Enfriar el campo tambien es parte del reset: si no, el calor acumulado
/// seguiria sacudiendo a los cuerpos nuevos.
pub fn reset_scene(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    agents: Query<Entity, With<Agent>>,
    mut field: ResMut<ThermalField>,
) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }
    for entity in &agents {
        commands.entity(entity).despawn();
    }
    field.clear();
}
