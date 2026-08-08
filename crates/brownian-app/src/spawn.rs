//! Interaccion con el raton y el teclado: elegir forma, previsualizarla y
//! crear o borrar cuerpos.
//!
//! Desde que la fisica vive en [`brownian_core`], una entidad de Bevy aqui es
//! solo la parte visible de un agente: malla, material y `Transform`. El cuerpo
//! —masa, collider, velocidad— esta en el `Env`, y el vinculo entre ambos es el
//! slot que guarda [`Agent`].

use bevy::prelude::*;
use brownian_core::ShapeKind;

use crate::config;
use crate::res::Sim;
use crate::shapes::ShapeVisuals;

/// La parte visible de un agente. El `slot` es su identidad dentro del `Env`.
#[derive(Component, Clone, Copy)]
pub struct Agent {
    pub slot: u32,
}

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

/// Crea un agente: el cuerpo en la simulacion y su representacion en el ECS.
/// Es el unico sitio donde se atan los dos, para que no puedan separarse.
pub fn spawn_body(
    commands: &mut Commands,
    sim: &mut Sim,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    shape: ShapeKind,
    size: f32,
    position: Vec2,
) -> Entity {
    let slot = sim.spawn(shape, size, position);

    commands
        .spawn((
            Agent { slot },
            Mesh2d(meshes.add(shape.mesh(size))),
            MeshMaterial2d(materials.add(shape.color())),
            Transform::from_xyz(position.x, position.y, 1.0),
        ))
        .id()
}

/// Lleva a los `Transform` lo que ha calculado la simulacion. Cierra el frame:
/// sin esto las mallas se quedarian donde nacieron.
pub fn sync_transforms(sim: Res<Sim>, mut agents: Query<(&Agent, &mut Transform)>) {
    for (agent, mut transform) in &mut agents {
        let Some(pose) = sim.agent_pose(agent.slot) else {
            continue;
        };
        transform.translation.x = pose.position.x;
        transform.translation.y = pose.position.y;
        transform.rotation = Quat::from_rotation_z(pose.angle);
    }
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
    mut sim: ResMut<Sim>,
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
        &mut sim,
        &mut meshes,
        &mut materials,
        shape,
        size,
        world,
    );
}

/// Boton derecho: borra el cuerpo bajo el cursor.
///
/// Quien decide que hay debajo es la consulta espacial del `Env`, que ademas
/// deja fuera los muros por construccion. Aqui solo queda retirar las mallas de
/// los slots que se han llevado.
pub fn despawn_on_click(
    mut commands: Commands,
    buttons: Res<ButtonInput<MouseButton>>,
    mut sim: ResMut<Sim>,
    windows: Query<&Window>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    agents: Query<(Entity, &Agent)>,
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

    let removed = sim.despawn_at(world);
    if removed.is_empty() {
        return;
    }

    for (entity, agent) in &agents {
        if removed.contains(&agent.slot) {
            commands.entity(entity).despawn();
        }
    }
}

/// Tecla R: deja el recinto como al principio, sin cuerpos y con el medio frio.
/// Enfriar el campo tambien es parte del reset: si no, el calor acumulado
/// seguiria sacudiendo a los cuerpos nuevos.
///
/// Es el equivalente manual del `reset(seed)` que pedira el entorno de
/// aprendizaje, y por eso reinicia tambien el reloj: dos episodios que arrancan
/// con la misma semilla tienen que recibir el mismo ruido, y no lo harian si el
/// contador de pasos siguiera donde lo dejo el anterior.
pub fn reset_scene(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    agents: Query<Entity, With<Agent>>,
    mut sim: ResMut<Sim>,
) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }
    for entity in &agents {
        commands.entity(entity).despawn();
    }
    sim.clear_agents();
    sim.field.clear();
    sim.clock.reset(config::RNG_SEED);
}

#[cfg(test)]
mod tests {
    use super::*;
    use brownian_core::Env;
    use brownian_core::field::serial::SerialBackend;

    /// El puente entre lo que se simula y lo que se ve. Si se rompe, los cuerpos
    /// siguen moviendose en el `Env` y las mallas se quedan clavadas donde
    /// nacieron: un fallo que no da error por ningun lado y que solo se nota
    /// mirando la ventana.
    #[test]
    fn transforms_follow_the_simulation() {
        let mut sim = Sim(Env::new(64, 36, 1));
        let slot = sim.spawn(ShapeKind::Ball, 18.0, Vec2::new(0.0, 200.0));
        let start = sim.agent_position(slot).expect("recien creado");

        let mut backend = SerialBackend;
        for _ in 0..120 {
            sim.step(&mut backend);
        }
        let moved = sim.agent_position(slot).expect("sigue vivo");
        assert_ne!(start, moved, "el cuerpo no se movio: el test no prueba nada");

        let mut world = World::new();
        let entity = world
            .spawn((
                Agent { slot },
                Transform::from_xyz(start.x, start.y, 1.0),
            ))
            .id();
        world.insert_resource(sim);

        let mut schedule = Schedule::default();
        schedule.add_systems(sync_transforms);
        schedule.run(&mut world);

        let transform = world.get::<Transform>(entity).expect("la entidad sigue ahi");
        assert_eq!(
            transform.translation.truncate(),
            moved,
            "la malla no siguio al cuerpo"
        );
        // La z no la toca nadie: es la capa de dibujo, no parte de la fisica.
        assert_eq!(transform.translation.z, 1.0);
    }

    /// Borrar con el raton tiene que llevarse las dos mitades del agente. Si se
    /// queda la entidad, aparece una malla fantasma que ya no se mueve.
    #[test]
    fn despawning_removes_body_and_mesh_together() {
        let mut sim = Sim(Env::new(64, 36, 1));
        let slot = sim.spawn(ShapeKind::Square, 20.0, Vec2::ZERO);

        let removed = sim.despawn_at(Vec2::ZERO);
        assert_eq!(removed, vec![slot]);
        assert_eq!(sim.agent_count(), 0);
        assert!(
            sim.agent_pose(slot).is_none(),
            "el slot borrado sigue publicando pose, y la malla lo seguiria"
        );
    }
}
