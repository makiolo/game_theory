// Un sub-paso de difusion de la rejilla termica, una celda por invocacion.
//
// Es la traduccion literal de `diffuse_row` en el modulo padre: mismo
// laplaciano de cinco puntos, mismos bordes reflectantes (el vecino de fuera es
// la propia celda) y misma relajacion hacia el bano termico. Si las dos
// versiones se separan, el test que compara GPU contra CPU lo detecta.

struct Params {
    kx: f32,
    ky: f32,
    decay: f32,
    ambient: f32,
    width: u32,
    height: u32,
    // La alineacion de un uniform es de 16 bytes.
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn diffuse(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    // La rejilla no tiene por que ser multiplo del tamano de grupo.
    if (x >= p.width || y >= p.height) {
        return;
    }

    let w = p.width;
    // Bordes reflectantes. El `x - 1u` de la rama descartada desborda, pero en
    // enteros sin signo eso solo da la vuelta y su valor no llega a usarse.
    let xl = select(x - 1u, x, x == 0u);
    let xr = select(x + 1u, x, x + 1u >= w);
    let yu = select(y - 1u, y, y == 0u);
    let yd = select(y + 1u, y, y + 1u >= p.height);

    let row = y * w;
    let c = src[row + x];
    let lap_x = src[row + xl] + src[row + xr] - 2.0 * c;
    let lap_y = src[yu * w + x] + src[yd * w + x] - 2.0 * c;

    dst[row + x] = c + p.kx * lap_x + p.ky * lap_y - p.decay * (c - p.ambient);
}
