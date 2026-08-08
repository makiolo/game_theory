//! Constantes de la simulacion, en un solo sitio para poder afinarla rapido.
//!
//! Solo lo que decide como se mueve el mundo. Lo que decide como se ve —tamano
//! de ventana, colores— vive en el `config` de la app: la simulacion tiene que
//! poder correr sin que exista una pantalla.

/// Escala de Rapier: cuantos pixeles de mundo equivalen a un metro fisico.
pub const PIXELS_PER_METER: f32 = 100.0;

/// Semi-extension del recinto en pixeles de mundo (el origen esta en el centro).
pub const ARENA_HALF_W: f32 = 620.0;
pub const ARENA_HALF_H: f32 = 340.0;
/// Grosor de los muros; generoso para que nada los atraviese a alta velocidad.
pub const WALL_THICKNESS: f32 = 20.0;

/// Rebote y rozamiento por defecto de los cuerpos que crea el usuario.
pub const BODY_RESTITUTION: f32 = 0.75;
pub const BODY_FRICTION: f32 = 0.3;

// --- Rejilla termica -------------------------------------------------------
// La rejilla cubre exactamente el interior del recinto. Sus dos dimensiones son
// independientes, asi que dx != dy y el laplaciano lleva coeficientes separados.

pub const GRID_W: usize = 512;
pub const GRID_H: usize = 288;

/// Difusividad termica del medio, en pixeles^2/s.
///
/// El esquema explicito solo es estable si `alpha*dt/dx^2` se queda por debajo
/// de 1/2, asi que este valor fija cuantos sub-pasos hay que dar por frame. Con
/// celdas de ~2.4 px salen unos pocos; subirlo mucho dispara el coste sin que
/// se note en pantalla.
pub const ALPHA: f32 = 300.0;
/// Ritmo con el que el medio vuelve a su temperatura de fondo, en 1/s.
pub const DECAY: f32 = 0.8;

/// Temperatura del bano termico.
///
/// Sin ella el sistema es puramente disipativo: los cuerpos ceden energia al
/// medio, se paran, y el movimiento browniano no llega a existir. Un bano a
/// temperatura constante es justo lo que agita las particulas en el fenomeno
/// real, y ademas hace visible el campo de fondo.
pub const AMBIENT: f32 = 1.0;

/// Calor que un cuerpo deposita por unidad de energia cinetica especifica.
/// Calibrado para que un cuerpo rapido caliente su celda bastante por encima
/// del fondo, dejando una estela visible.
pub const HEAT_PER_SPEED2: f32 = 2.0e-5;

/// Magnitud del impulso browniano, por unidad de `sqrt(masa * temperatura)`.
///
/// El impulso escala con `sqrt(masa)` siguiendo la ecuacion de Langevin, de
/// modo que la velocidad resultante va como `sqrt(T/m)`: es la equiparticion de
/// energia, y es lo que da la firma visual del movimiento browniano — las
/// formas pequenas vibran mucho y las grandes apenas se inmutan.
///
/// El numero es grande porque Rapier trabaja aqui en pixeles, no en metros:
/// `pixels_per_meter` ajusta la escala de la simulacion pero no encoge la
/// geometria, asi que una pelota de 18 px de radio tiene masa ~1018, no 0.1.
/// Calibrado junto con [`LINEAR_DAMPING`] para que en equilibrio una pelota de
/// 18 px ronde los 150 px/s.
pub const IMPULSE_SCALE: f32 = 10_000.0;

/// Arrastre viscoso del medio, en 1/s.
///
/// Es la otra mitad de la ecuacion de Langevin y no es opcional: el ruido
/// inyecta energia en cada paso, asi que sin un termino disipativo que la
/// retire la velocidad crece como un paseo aleatorio y la simulacion acaba
/// siendo un gas descontrolado. Con ambos terminos el sistema se asienta en un
/// equilibrio, que es lo que dice el teorema de fluctuacion-disipacion.
///
/// De paso da la sensacion de cuerpos suspendidos en un fluido, coherente con
/// [`GRAVITY_SCALE`]: caen a velocidad terminal en vez de desplomarse.
pub const LINEAR_DAMPING: f32 = 2.5;
pub const ANGULAR_DAMPING: f32 = 2.0;

/// Fraccion de la gravedad terrestre que sienten los cuerpos.
///
/// Las particulas brownianas reales estan suspendidas en un fluido que compensa
/// casi todo su peso; por eso la agitacion termica puede con ellas. Con la
/// gravedad completa el peso aplasta el efecto y las formas solo se apilan en
/// el suelo. Dejamos algo de gravedad para que siga habiendo un "abajo".
pub const GRAVITY_SCALE: f32 = 0.15;

// --- Aprendizaje ------------------------------------------------------------

/// Aceleracion que puede imprimirse un agente a si mismo, en pixeles/s^2, con la
/// accion a fondo.
///
/// Va como aceleracion y no como impulso, es decir escalada por la masa: asi
/// todos los tamanos tienen la misma autoridad sobre si mismos, mientras que el
/// ruido termico —que va como `sqrt(T/m)`— zarandea mucho mas a los pequenos.
/// De esa asimetria sale lo interesante de la tarea: un cuerpo grande se dirige
/// con facilidad pero le cuesta arrancar, y uno pequeno es agil y a la vez
/// esclavo del ruido.
///
/// Con [`LINEAR_DAMPING`] deja la velocidad de crucero en unos 160 px/s,
/// comparable a la que alcanza una pelota de 18 px solo por agitacion termica.
/// Si la accion pudiera mucho mas que el bano, la politica aprenderia a ignorar
/// el medio, que es justo lo que hay que evitar.
pub const ACTION_ACCEL: f32 = 400.0;

/// Pasos que dura un episodio antes de truncarse (60 s a [`SIM_DT`]).
pub const MAX_EPISODE_STEPS: u64 = 3_600;

/// Escala con la que se normaliza el exceso de temperatura al convertirlo en
/// recompensa. Deja la senal en un rango cercano a [0, 1] para lo que se ve
/// normalmente en una celda caliente, sin recortarla.
pub const REWARD_TEMPERATURE_SCALE: f32 = 4.0;

/// Semilla del RNG: fija para que una misma partida sea reproducible.
pub const RNG_SEED: u64 = 0x5EED_B12E_1234_5678;

/// Paso de tiempo de la simulacion.
///
/// Fijo, no el delta del frame. Con un paso variable la trayectoria depende de
/// lo cargada que este la maquina: dos ejecuciones con la misma semilla
/// divergen, y un entorno de aprendizaje que no se puede reproducir tampoco se
/// puede depurar.
///
/// De paso desaparece la espiral que antes obligaba a poner un tope — un frame
/// lento ya no pide mas sub-pasos de difusion. Lo que ocurre ahora es que
/// `FixedUpdate` acumula el retraso y da varios pasos seguidos, o los descarta
/// si no llega; la simulacion se separa del reloj de pared, pero cada paso vale
/// exactamente lo mismo.
pub const SIM_DT: f32 = 1.0 / 60.0;
