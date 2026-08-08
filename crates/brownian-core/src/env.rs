//! El mundo de brownian, avanzable paso a paso y sin nada delante.
//!
//! Monta la fisica sobre `rapier2d` —el motor oficial de Dimforge, el mismo que
//! `bevy_rapier2d` usa por dentro— con los sets y el `PhysicsPipeline` que
//! describe su guia de usuario. No hay envoltorio propio ni reimplementacion de
//! nada: lo unico que cambia respecto a la app es quien lleva el reloj.
//!
//! Conviene saber que desde la 0.33 rapier trabaja en `glam`, y que su alias
//! `Vector` es exactamente `glam::Vec2` — el mismo tipo que usa Bevy. Por eso no
//! hay una sola conversion en este archivo: los vectores cruzan la frontera tal
//! cual.
//!
//! Un paso de [`Env::step`] hace, en este orden:
//!
//! 1. avanza el reloj, que es la coordenada temporal del ruido;
//! 2. los cuerpos calientan el medio al moverse;
//! 3. el campo difunde;
//! 4. el medio caliente sacude a los cuerpos;
//! 5. la fisica integra.
//!
//! Es el mismo orden que encadena la app en su `FixedUpdate`, y esa coincidencia
//! es lo que permite comparar una trayectoria contra la otra.

use glam::Vec2;
use rapier2d::prelude::{
    BroadPhaseBvh, CCDSolver, ColliderBuilder, ColliderSet, ImpulseJointSet, IntegrationParameters,
    IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline, RigidBodyBuilder,
    RigidBodyHandle, RigidBodySet,
};

use crate::config;
use crate::field::{FieldBackend, ThermalField};
use crate::shapes::ShapeKind;
use crate::sim::{SimClock, noise_angle, stream};

/// Lo que hace falta saber de un agente para dibujarlo.
#[derive(Clone, Copy, Debug)]
pub struct AgentPose {
    pub slot: u32,
    pub position: Vec2,
    pub angle: f32,
    pub shape: ShapeKind,
    pub size: f32,
}

/// Los agentes en columnas, indexados por slot.
///
/// El slot es la identidad estable del agente: la coordenada con la que se
/// direcciona su ruido y la que vera una politica. Por eso este vector no se
/// reordena nunca y los huecos que deja borrar se reutilizan, en vez de
/// compactarse — compactar renumeraria a los supervivientes.
#[derive(Default)]
struct Agents {
    handle: Vec<Option<RigidBodyHandle>>,
    shape: Vec<ShapeKind>,
    size: Vec<f32>,
    free: Vec<u32>,
}

impl Agents {
    fn alloc(&mut self, handle: RigidBodyHandle, shape: ShapeKind, size: f32) -> u32 {
        if let Some(slot) = self.free.pop() {
            let i = slot as usize;
            self.handle[i] = Some(handle);
            self.shape[i] = shape;
            self.size[i] = size;
            return slot;
        }

        self.handle.push(Some(handle));
        self.shape.push(shape);
        self.size.push(size);
        (self.handle.len() - 1) as u32
    }

    fn free_slot(&mut self, slot: u32) -> Option<RigidBodyHandle> {
        let handle = self.handle.get_mut(slot as usize)?.take()?;
        self.free.push(slot);
        Some(handle)
    }

    fn get(&self, slot: u32) -> Option<RigidBodyHandle> {
        *self.handle.get(slot as usize)?
    }

    fn alive(&self) -> impl Iterator<Item = (u32, RigidBodyHandle)> + '_ {
        self.handle
            .iter()
            .enumerate()
            .filter_map(|(i, h)| h.map(|h| (i as u32, h)))
    }

    fn count(&self) -> usize {
        self.handle.len() - self.free.len()
    }

    fn clear(&mut self) {
        self.handle.clear();
        self.shape.clear();
        self.size.clear();
        self.free.clear();
    }
}

/// Un mundo completo: recinto, cuerpos y medio termico.
pub struct Env {
    pub field: ThermalField,
    pub clock: SimClock,

    agents: Agents,

    // Los sets de rapier, tal cual los pide `PhysicsPipeline::step`.
    bodies: RigidBodySet,
    colliders: ColliderSet,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    pipeline: PhysicsPipeline,
    params: IntegrationParameters,
    gravity: Vec2,
}

impl Default for Env {
    fn default() -> Self {
        Self::new(config::GRID_W, config::GRID_H, config::RNG_SEED)
    }
}

impl Env {
    pub fn new(grid_w: usize, grid_h: usize, seed: u64) -> Self {
        let params = IntegrationParameters {
            dt: config::SIM_DT,
            // La via oficial para trabajar en pixeles en vez de metros: escala
            // las tolerancias internas del solver, pensadas para objetos de
            // tamano humano. Es lo que hace `pixels_per_meter` en bevy_rapier.
            length_unit: config::PIXELS_PER_METER,
            ..Default::default()
        };

        let mut env = Self {
            field: ThermalField::with_grid(grid_w, grid_h),
            clock: SimClock {
                seed,
                ..Default::default()
            },
            agents: Agents::default(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            pipeline: PhysicsPipeline::new(),
            params,
            // En pixeles por segundo al cuadrado, y rebajada: ver
            // `config::GRAVITY_SCALE`.
            gravity: Vec2::new(
                0.0,
                -9.81 * config::PIXELS_PER_METER * config::GRAVITY_SCALE,
            ),
        };

        env.build_arena();
        env
    }

    /// Cuatro colliders fijos formando una caja cerrada alrededor del origen.
    fn build_arena(&mut self) {
        let hw = config::ARENA_HALF_W;
        let hh = config::ARENA_HALF_H;
        let t = config::WALL_THICKNESS;

        // (posicion, semi-extension) de suelo, techo, muro izquierdo y derecho.
        let walls = [
            (Vec2::new(0.0, -hh - t), Vec2::new(hw + t, t)),
            (Vec2::new(0.0, hh + t), Vec2::new(hw + t, t)),
            (Vec2::new(-hw - t, 0.0), Vec2::new(t, hh + t)),
            (Vec2::new(hw + t, 0.0), Vec2::new(t, hh + t)),
        ];

        for (pos, half) in walls {
            let collider = ColliderBuilder::cuboid(half.x, half.y)
                .translation(pos)
                .restitution(config::BODY_RESTITUTION)
                .build();
            self.colliders.insert(collider);
        }
    }

    /// Crea un agente y devuelve su slot.
    pub fn spawn(&mut self, shape: ShapeKind, size: f32, position: Vec2) -> u32 {
        let body = RigidBodyBuilder::dynamic()
            .translation(position)
            .linear_damping(config::LINEAR_DAMPING)
            .angular_damping(config::ANGULAR_DAMPING)
            // En un bano termico ningun cuerpo esta nunca del todo en reposo: si
            // rapier los durmiera, dejarian de temblar al asentarse.
            .can_sleep(false)
            // Con la agitacion alta los cuerpos pequenos alcanzan velocidades a
            // las que podrian colarse entre muros de un paso al siguiente.
            .ccd_enabled(true)
            .build();
        let handle = self.bodies.insert(body);

        let collider = shape
            .collider(size)
            .restitution(config::BODY_RESTITUTION)
            .friction(config::BODY_FRICTION)
            .build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);

        let slot = self.agents.alloc(handle, shape, size);
        // El slot viaja en el cuerpo para poder recuperarlo desde una consulta
        // espacial, que devuelve colliders y no indices nuestros.
        self.bodies[handle].user_data = slot as u128;
        slot
    }

    /// Atajo para el caso mas comun.
    pub fn spawn_ball(&mut self, radius: f32, position: Vec2) -> u32 {
        self.spawn(ShapeKind::Ball, radius, position)
    }

    /// Borra un agente. Su slot queda libre para el siguiente.
    pub fn despawn(&mut self, slot: u32) {
        let Some(handle) = self.agents.free_slot(slot) else {
            return;
        };
        self.bodies.remove(
            handle,
            &mut self.islands,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            true,
        );
    }

    /// Borra los agentes cuyo cuerpo contiene el punto, y devuelve sus slots.
    ///
    /// Recorre los agentes en vez de consultar el arbol del broad-phase, y no es
    /// por descuido: ese arbol no conoce a un cuerpo hasta que pasa por
    /// `PhysicsPipeline::step`, de modo que una consulta espacial no encuentra
    /// lo que se acaba de crear. Con el raton eso se traduce en que un cuerpo
    /// recien soltado es imposible de borrar hasta el siguiente paso.
    ///
    /// Recorrerlos cuesta O(n) sobre las decenas de cuerpos que hay en el
    /// recinto, y sucede una vez por clic. Es un precio que no se nota a cambio
    /// de una respuesta que siempre es la correcta.
    ///
    /// Los muros tampoco corren peligro: no estan entre los agentes.
    pub fn despawn_at(&mut self, point: Vec2) -> Vec<u32> {
        let hits: Vec<u32> = self
            .agents
            .alive()
            .filter(|(_, handle)| {
                self.bodies[*handle].colliders().iter().any(|collider| {
                    let collider = &self.colliders[*collider];
                    collider.shape().contains_point(collider.position(), point)
                })
            })
            .map(|(slot, _)| slot)
            .collect();

        for slot in &hits {
            self.despawn(*slot);
        }
        hits
    }

    /// Vacia el recinto. El recinto en si —los muros— se queda.
    pub fn clear_agents(&mut self) {
        for (_, handle) in self.agents.alive().collect::<Vec<_>>() {
            self.bodies.remove(
                handle,
                &mut self.islands,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                true,
            );
        }
        self.agents.clear();
    }

    pub fn agent_count(&self) -> usize {
        self.agents.count()
    }

    /// Slots vivos, en orden.
    pub fn alive_slots(&self) -> impl Iterator<Item = u32> + '_ {
        self.agents.alive().map(|(slot, _)| slot)
    }

    /// Posicion y orientacion de cada agente vivo, para dibujarlos.
    pub fn agent_poses(&self) -> impl Iterator<Item = AgentPose> + '_ {
        self.agents
            .alive()
            .map(|(slot, handle)| self.pose_of(slot, handle))
    }

    /// Lo mismo para un agente suelto, o `None` si su slot esta libre.
    pub fn agent_pose(&self, slot: u32) -> Option<AgentPose> {
        Some(self.pose_of(slot, self.agents.get(slot)?))
    }

    fn pose_of(&self, slot: u32, handle: RigidBodyHandle) -> AgentPose {
        let body = &self.bodies[handle];
        AgentPose {
            slot,
            position: body.translation(),
            angle: body.rotation().angle(),
            shape: self.agents.shape[slot as usize],
            size: self.agents.size[slot as usize],
        }
    }

    /// Posicion de un agente en coordenadas de mundo.
    pub fn agent_position(&self, slot: u32) -> Option<Vec2> {
        Some(self.bodies[self.agents.get(slot)?].translation())
    }

    /// Velocidad lineal de un agente.
    pub fn agent_velocity(&self, slot: u32) -> Option<Vec2> {
        Some(self.bodies[self.agents.get(slot)?].linvel())
    }

    /// Velocidad media de los agentes vivos. Con el medio frio tiende a cero, y
    /// es la forma directa de ver si el bano los esta agitando de verdad.
    pub fn mean_speed(&self) -> f32 {
        let count = self.agents.count();
        if count == 0 {
            return 0.0;
        }
        let total: f32 = self
            .agents
            .alive()
            .map(|(_, h)| self.bodies[h].linvel().length())
            .sum();
        total / count as f32
    }

    /// Avanza el mundo un paso de `config::SIM_DT`.
    pub fn step(&mut self, backend: &mut dyn FieldBackend) {
        self.clock.advance();
        self.deposit_heat();
        self.field.step(backend, config::SIM_DT);
        self.apply_brownian_impulse();
        self.step_physics();
    }

    /// Cuerpos -> campo: el rozamiento con el medio lo calienta.
    fn deposit_heat(&mut self) {
        for (_, handle) in self.agents.alive() {
            let body = &self.bodies[handle];
            let speed2 = body.linvel().length_squared();
            if speed2 <= 0.0 {
                continue;
            }
            self.field.deposit(
                body.translation(),
                config::HEAT_PER_SPEED2 * speed2 * config::SIM_DT,
            );
        }
    }

    /// Campo -> cuerpos: el medio caliente los sacude en direccion aleatoria.
    fn apply_brownian_impulse(&mut self) {
        // Ruido blanco integrado sobre dt: la desviacion crece con sqrt(dt).
        let dt_scale = config::SIM_DT.sqrt();
        let (seed, step) = (self.clock.seed, self.clock.step);

        for (slot, handle) in self.agents.alive() {
            let body = &mut self.bodies[handle];
            let temperature = self.field.sample(body.translation());
            if temperature <= 0.0 {
                continue;
            }

            let angle = noise_angle(seed, step, slot, stream::IMPULSE_ANGLE);
            let magnitude = config::IMPULSE_SCALE * (body.mass() * temperature).sqrt() * dt_scale;
            body.apply_impulse(Vec2::new(angle.cos(), angle.sin()) * magnitude, true);
        }
    }

    fn step_physics(&mut self) {
        self.pipeline.step(
            self.gravity,
            &self.params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::serial::SerialBackend;

    /// Una fila de bolas del mismo tamano, repartidas a lo ancho del recinto.
    fn seeded_env(seed: u64, radius: f32, count: u32) -> Env {
        let mut env = Env::new(128, 72, seed);
        for i in 0..count {
            let x = -300.0 + i as f32 * 80.0;
            env.spawn_ball(radius, Vec2::new(x, 200.0));
        }
        env
    }

    fn run(env: &mut Env, steps: u32) {
        let mut backend = SerialBackend;
        for _ in 0..steps {
            env.step(&mut backend);
        }
    }

    fn positions(env: &Env) -> Vec<Vec2> {
        env.agent_poses().map(|p| p.position).collect()
    }

    /// La razon de ser de la fase 0: misma semilla, misma trayectoria. Si esto
    /// falla, cualquier medida sobre el entorno deja de ser comparable.
    #[test]
    fn same_seed_gives_the_same_trajectory() {
        let (mut a, mut b) = (seeded_env(99, 18.0, 8), seeded_env(99, 18.0, 8));
        run(&mut a, 400);
        run(&mut b, 400);

        assert_eq!(
            positions(&a),
            positions(&b),
            "dos ejecuciones identicas divergieron"
        );
    }

    /// Y el reverso: si la semilla no llegara al ruido, el test anterior pasaria
    /// igual y no estaria comprobando nada.
    #[test]
    fn different_seeds_diverge() {
        let (mut a, mut b) = (seeded_env(1, 18.0, 8), seeded_env(2, 18.0, 8));
        run(&mut a, 400);
        run(&mut b, 400);

        assert_ne!(
            positions(&a),
            positions(&b),
            "dos semillas distintas dieron la misma trayectoria"
        );
    }

    /// El recinto tiene que contener a los cuerpos: es lo que verifica de paso
    /// que los muros y el CCD sobrevivieron al cambio de motor.
    #[test]
    fn bodies_stay_inside_the_arena() {
        let mut env = seeded_env(7, 8.0, 10);
        run(&mut env, 1200);

        for pose in env.agent_poses() {
            let p = pose.position;
            assert!(
                p.x.abs() <= config::ARENA_HALF_W && p.y.abs() <= config::ARENA_HALF_H,
                "el agente {} se escapo del recinto: {p:?}",
                pose.slot
            );
        }
    }

    /// El bano termico no deja quieto a nadie. Sin el, los cuerpos caerian, se
    /// asentarian y el movimiento browniano no llegaria a existir.
    #[test]
    fn the_bath_keeps_bodies_agitated() {
        let mut env = seeded_env(3, 10.0, 6);
        run(&mut env, 900);
        assert!(
            env.mean_speed() > 1.0,
            "los cuerpos se pararon: el bano no los esta agitando"
        );
    }

    /// La firma visual del movimiento browniano, y la comprobacion de que el
    /// acoplamiento sobrevivio al port: el impulso va como `sqrt(m*T)`, asi que
    /// la velocidad va como `sqrt(T/m)` y las formas pequenas se agitan mucho
    /// mas que las grandes. La gravedad aporta lo mismo a todas —con arrastre
    /// lineal la velocidad terminal no depende de la masa—, asi que la
    /// diferencia que quede es termica.
    #[test]
    fn small_bodies_shake_more_than_large_ones() {
        let (mut small, mut large) = (seeded_env(11, 6.0, 6), seeded_env(11, 40.0, 6));
        run(&mut small, 900);
        run(&mut large, 900);

        let (vs, vl) = (small.mean_speed(), large.mean_speed());
        assert!(
            vs > 2.0 * vl,
            "equiparticion rota: pequenas {vs:.1} px/s frente a grandes {vl:.1} px/s"
        );
    }

    /// Un cuerpo tiene que poder borrarse nada mas crearlo, sin esperar a un
    /// paso. Consultando el arbol del broad-phase esto fallaba: ese arbol no
    /// conoce a un cuerpo hasta que pasa por el `PhysicsPipeline`, asi que con
    /// el raton un cuerpo recien soltado era inmune al clic derecho.
    #[test]
    fn a_body_can_be_removed_before_the_first_step() {
        let mut env = Env::new(64, 36, 5);
        let slot = env.spawn(ShapeKind::Ball, 20.0, Vec2::ZERO);
        assert_eq!(env.despawn_at(Vec2::ZERO), vec![slot]);
        assert_eq!(env.agent_count(), 0);
    }

    /// Borrar por punto tiene que acertar al cuerpo que hay debajo y, sobre
    /// todo, no llevarse por delante los muros: no estan entre los agentes, asi
    /// que ninguna forma de apuntarles deberia tocarlos.
    #[test]
    fn despawn_at_hits_the_body_under_the_point_and_spares_the_walls() {
        let mut env = Env::new(64, 36, 5);
        let slot = env.spawn(ShapeKind::Square, 20.0, Vec2::new(0.0, 0.0));
        env.spawn(ShapeKind::Ball, 20.0, Vec2::new(300.0, 0.0));
        run(&mut env, 1);

        assert!(env.despawn_at(Vec2::new(0.0, 0.0)).contains(&slot));
        assert_eq!(env.agent_count(), 1, "se borro de mas");

        // Sobre un muro no hay nada que borrar, y el recinto sigue en pie.
        let on_the_wall = Vec2::new(0.0, -config::ARENA_HALF_H - config::WALL_THICKNESS);
        assert!(env.despawn_at(on_the_wall).is_empty());

        run(&mut env, 600);
        for pose in env.agent_poses() {
            assert!(
                pose.position.y.abs() <= config::ARENA_HALF_H,
                "el suelo desaparecio: el cuerpo se cayo del recinto"
            );
        }
    }

    /// Los slots se reutilizan, que es lo que mantiene acotado el `N` que vera
    /// una politica por mucho que el usuario cree y borre cuerpos.
    #[test]
    fn slots_are_recycled() {
        let mut env = Env::new(64, 36, 5);
        let a = env.spawn_ball(10.0, Vec2::new(-50.0, 0.0));
        let b = env.spawn_ball(10.0, Vec2::new(50.0, 0.0));
        assert_eq!((a, b), (0, 1));

        env.despawn(a);
        assert_eq!(env.agent_count(), 1);
        assert_eq!(env.agent_position(a), None, "el slot borrado sigue vivo");

        let c = env.spawn_ball(10.0, Vec2::new(0.0, 100.0));
        assert_eq!(c, a, "el slot libre no se reutilizo");
        assert_eq!(env.agent_count(), 2);
    }
}
