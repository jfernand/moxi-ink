// MoXi ink dispersion as compute shaders: the passes of `Ink::step` (src/ink.rs), one entry point
// each, with the same arithmetic in the same order. See src/gpu.rs for the buffers.

struct Params {
    width: u32,
    height: u32,
    omega: f32,
    advection: f32,
    evaporation: f32,
    edge_evaporation: f32,
    block_base: f32,
    block_grain: f32,
    block_alum: f32,
    block_glue: f32,
    block_fixed: f32,
    pin_base: f32,
    pin_fixed: f32,
    pin_disorder: f32,
    toe_glue: f32,
    pin_diagonal: f32,
    capacity: f32,
    surface_capacity: f32,
    receptivity: f32,
    base_mask: f32,
    hindrance: f32,
    hindrance_speed: f32,
    hindrance_still: f32,
    fix_base: f32,
    fix_dryness: f32,
    fix_glue: f32,
    load_water: f32,
    load_glue: f32,
    load_pigment: vec4<f32>,
}

// Per-site scalars, CELL apart.
const CELL: u32 = 12u;
const WATER: u32 = 0u;
const LAST_WATER: u32 = 1u;
const UX: u32 = 2u;
const UY: u32 = 3u;
const SURFACE_WATER: u32 = 4u;
const SEEP: u32 = 5u;
const BLOCK: u32 = 6u;
const STAIN: u32 = 7u;
const HELD: u32 = 8u;
const DRIED: u32 = 9u;

// A pinned site's blocking: "infinite", as in the CPU version.
const PINNED: f32 = 1e30;
const DRY: f32 = 1e-5;

@group(0) @binding(0) var<uniform> p: Params;
// Distributions, 9 per site: before streaming, and after.
@group(0) @binding(1) var<storage, read_write> f: array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> cell: array<f32>;
// Ink concentrations in the flow layer, and the next step's (also the footprint, in .x, for
// `deposit`).
@group(0) @binding(4) var<storage, read_write> ink: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> ink_out: array<vec4<f32>>;
// Two per site: surface ink concentrations, then fixed ink.
@group(0) @binding(6) var<storage, read_write> layers: array<vec4<f32>>;
// grain, alum, pinning, rougher.
@group(0) @binding(7) var<storage, read> paper: array<vec4<f32>>;

const DIRECTIONS = array<vec2<i32>, 9>(
    vec2(0, 0), vec2(0, -1), vec2(1, 0), vec2(-1, 0), vec2(0, 1),
    vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(-1, 1),
);
const OPPOSITE = array<u32, 9>(0u, 4u, 3u, 2u, 1u, 8u, 7u, 6u, 5u);
const WEIGHTS = array<f32, 9>(
    4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
);

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

fn smoothstep_(a: f32, b: f32, x: f32) -> f32 {
    if b <= a {
        return select(1.0, 0.0, x < a);
    }
    let t = clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// The site at `c + d`, or -1 off the canvas.
fn neighbour(c: u32, d: vec2<i32>) -> i32 {
    let x = i32(c % p.width) + d.x;
    let y = i32(c / p.width) + d.y;
    if x < 0 || y < 0 || x >= i32(p.width) || y >= i32(p.height) {
        return -1;
    }
    return y * i32(p.width) + x;
}

fn site(id: vec3<u32>) -> i32 {
    if id.x >= p.width || id.y >= p.height {
        return -1;
    }
    return i32(id.y * p.width + id.x);
}

fn fixed_sum(c: u32) -> f32 {
    let v = layers[2u * c + 1u];
    return 0.0 + v.x + v.y + v.z + v.w;
}

@compute @workgroup_size(8, 8)
fn block_and_pin(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = site(id);
    if s < 0 { return; }
    let c = u32(s);
    let pp = paper[c];
    let grain = pp.x;
    let glue = ink[c].w;
    let fixed = fixed_sum(c);
    let kappa = min(p.block_base + p.block_grain * (1.0 - grain) + p.block_alum * pp.y + p.block_glue * glue
        + p.block_fixed * fixed, 1.0);
    var block = kappa;
    if cell[c * CELL + WATER] <= 0.0 {
        var wettest_glue = 0.0;
        for (var i = 1u; i < 9u; i++) {
            let n = neighbour(c, DIRECTIONS[i]);
            if n >= 0 { wettest_glue = max(wettest_glue, ink[u32(n)].w); }
        }
        let toes = smoothstep_(0.0, p.toe_glue, wettest_glue);
        let disorder = lerp(grain, pp.z, toes);
        let sigma = p.pin_base + p.pin_fixed * fixed + p.pin_disorder * (1.0 - disorder);
        var pinned = true;
        for (var i = 1u; i < 9u; i++) {
            let threshold = select(sigma * p.pin_diagonal, sigma, i < 5u);
            let n = neighbour(c, DIRECTIONS[i]);
            if n >= 0 && !(cell[u32(n) * CELL + WATER] < threshold) { pinned = false; }
        }
        if pinned { block = PINNED; }
    }
    cell[c * CELL + BLOCK] = block;
    cell[c * CELL + LAST_WATER] = cell[c * CELL + WATER];
}

@compute @workgroup_size(8, 8)
fn collide(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = site(id);
    if s < 0 { return; }
    let c = u32(s);
    let rho = cell[c * CELL + WATER];
    if rho <= 0.0 { return; }
    let u = vec2(cell[c * CELL + UX], cell[c * CELL + UY]);
    let psi = smoothstep_(0.0, p.advection, rho);
    let uu = u.x * u.x + u.y * u.y;
    for (var i = 0u; i < 9u; i++) {
        let eu = f32(DIRECTIONS[i].x) * u.x + f32(DIRECTIONS[i].y) * u.y;
        let eq = WEIGHTS[i] * (rho + psi * (3.0 * eu + 4.5 * eu * eu - 1.5 * uu));
        let k = c * 9u + i;
        f[k] = f[k] + p.omega * (eq - f[k]);
    }
}

@compute @workgroup_size(8, 8)
fn stream(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = site(id);
    if s < 0 { return; }
    let c = u32(s);
    f_out[c * 9u] = f[c * 9u];
    for (var i = 1u; i < 9u; i++) {
        let back = f[c * 9u + OPPOSITE[i]];
        let n = neighbour(c, -DIRECTIONS[i]);
        var out = back;
        if n >= 0 {
            let k = (cell[c * CELL + BLOCK] + cell[u32(n) * CELL + BLOCK]) / 2.0;
            if k > 1.0 {
                out = max(back - p.edge_evaporation, 0.0);
            } else {
                out = lerp(f[u32(n) * 9u + i], back, k);
            }
        }
        f_out[c * 9u + i] = out;
    }
}

@compute @workgroup_size(8, 8)
fn gather_water(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = site(id);
    if s < 0 { return; }
    let c = u32(s);
    var rho = 0.0;
    for (var i = 0u; i < 9u; i++) { rho += f_out[c * 9u + i]; }
    let arrived = rho > 0.0;
    if rho > 0.0 && p.evaporation > 0.0 {
        let kept = max(rho - p.evaporation, 0.0) / rho;
        for (var i = 0u; i < 9u; i++) { f_out[c * 9u + i] *= kept; }
        rho *= kept;
    }
    var dried = 0.0;
    if rho < DRY {
        for (var i = 0u; i < 9u; i++) { f_out[c * 9u + i] = 0.0; }
        rho = 0.0;
        if arrived { dried = 1.0; }
    }
    var u = vec2(0.0, 0.0);
    for (var i = 1u; i < 9u; i++) {
        u.x += f32(DIRECTIONS[i].x) * f_out[c * 9u + i];
        u.y += f32(DIRECTIONS[i].y) * f_out[c * 9u + i];
    }
    let ws = cell[c * CELL + SURFACE_WATER];
    let seep = clamp(ws, 0.0, max(p.capacity - rho, 0.0));
    cell[c * CELL + SURFACE_WATER] = max(ws - seep, 0.0);
    f_out[c * 9u] += seep;
    rho += seep;
    cell[c * CELL + SEEP] = seep;
    cell[c * CELL + WATER] = rho;
    cell[c * CELL + UX] = u.x;
    cell[c * CELL + UY] = u.y;
    cell[c * CELL + STAIN] = max(cell[c * CELL + STAIN], rho);
    cell[c * CELL + DRIED] = dried;
}

@compute @workgroup_size(8, 8)
fn fix_ink(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = site(id);
    if s < 0 { return; }
    let c = u32(s);
    let before = cell[c * CELL + LAST_WATER];
    if before > 0.0 {
        let now = cell[c * CELL + WATER] - cell[c * CELL + SEEP];
        let lost = max(before - now, 0.0) / before;
        let flow = ink[c];
        let dryness = clamp(p.fix_dryness + p.fix_glue * flow.w, 0.0, 1.0);
        var fix = 1.0;
        if now > 0.0 { fix = min(max(lost * (1.0 - smoothstep_(0.0, dryness, now)), p.fix_base), 1.0); }
        layers[2u * c + 1u] += fix * flow * before;
        ink[c] = flow * (1.0 - fix);
    }
    // γ* for the ink pass.
    let ux = cell[c * CELL + UX];
    let uy = cell[c * CELL + UY];
    let speed = sqrt(ux * ux + uy * uy);
    cell[c * CELL + HELD] = lerp(p.hindrance_still, p.hindrance, smoothstep_(0.0, p.hindrance_speed, speed));
}

fn carried(c: u32) -> vec4<f32> {
    let here = ink[c];
    var kept = f[c * 9u];
    var out = vec4(0.0);
    for (var i = 1u; i < 9u; i++) {
        let own = f[c * 9u + OPPOSITE[i]];
        let n = neighbour(c, -DIRECTIONS[i]);
        if n < 0 {
            kept += own;
        } else {
            let sn = u32(n);
            let k = min((cell[c * CELL + BLOCK] + cell[sn * CELL + BLOCK]) / 2.0, 1.0);
            kept += k * own;
            let share = (1.0 - k) * f[sn * 9u + i] * (1.0 - cell[sn * CELL + HELD]);
            out += share * ink[sn];
        }
        let d = neighbour(c, DIRECTIONS[i]);
        if d >= 0 {
            let k = min((cell[c * CELL + BLOCK] + cell[u32(d) * CELL + BLOCK]) / 2.0, 1.0);
            kept += (1.0 - k) * f[c * 9u + i] * cell[c * CELL + HELD];
        }
    }
    return out + kept * here;
}

@compute @workgroup_size(8, 8)
fn move_ink(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = site(id);
    if s < 0 { return; }
    let c = u32(s);
    if cell[c * CELL + DRIED] > 0.0 {
        layers[2u * c + 1u] += carried(c);
    }
    let water = cell[c * CELL + WATER];
    if water <= 0.0 {
        ink_out[c] = vec4(0.0);
        return;
    }
    let seep = cell[c * CELL + SEEP];
    let flowing = water - seep;
    var out = vec4(0.0);
    if flowing > 0.0 { out = carried(c) / flowing; }
    if seep > 0.0 {
        out = (out * max(flowing, 0.0) + layers[2u * c] * seep) / water;
    }
    ink_out[c] = out;
}

// `Ink::deposit`, with the footprint in `ink_out[c].x` and the load in the parameters.
@compute @workgroup_size(8, 8)
fn deposit(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = site(id);
    if s < 0 { return; }
    let c = u32(s);
    let weight = ink_out[c].x;
    if weight <= 0.0 { return; }
    let mask = max(1.0 - cell[c * CELL + WATER] / p.receptivity, p.base_mask);
    let before = cell[c * CELL + SURFACE_WATER];
    let after = min(before + weight * mask * p.load_water, p.surface_capacity);
    let added = after - before;
    if added <= 0.0 { return; }
    let load = vec4(p.load_pigment.xyz, p.load_glue);
    layers[2u * c] = (layers[2u * c] * before + load * added) / after;
    cell[c * CELL + SURFACE_WATER] = after;
}
