// Shared lighting estimator for world-space and view-space receivers.
// Bindings 6 and 7 belong to each consumer's input/output layout. Everything
// else, including source sampling, cache termination and reach, is shared.
override DIFFUSE_HITS: u32 = 3u;
override LIMITED_REACH: bool = false;
// The far tier's ray budget (NAADF's): path, sun and emitter
// visibility rays stop after PATH_STEPS traversal steps. With BUDGETED_RAYS
// a ray out of steps is a miss, as in NAADF, and a region missing from the
// directory is crossed as empty space (the `visible_geometry` rule) instead
// of walking the rest of the world to tell unloaded from sky, which made
// horizon rays cost the same at any step budget. The defaults are today's
// rule.
override PATH_STEPS: u32 = 2048u;
override BUDGETED_RAYS: bool = false;
struct Params {
    counts: vec4<u32>, // tasks, samples per face, directory mask, light count
    minimum: vec4<i32>, // xyz: scene bounds, w: shared corner task count
    maximum: vec4<i32>, // xyz: scene bounds, w: directional light count
    environment: vec4<f32>,
    extra: vec4<f32>, // reserved, directional horizon, emitter power/count
    transport: vec4<f32>, // shortened diffuse/voxel-emitter reach, reserved
}
struct Material {
    diffuse: vec4<f32>,
    diffuse_extra: vec4<f32>,
    emission: vec4<f32>,
    emission_extra: vec4<f32>,
}
struct Light {
    position_range: vec4<f32>, // negative range denotes a directional source
    energy: vec4<f32>,
    extra: vec4<f32>,
}
struct EmissiveFace {
    voxel: vec3<i32>,
    face: u32,
    sampling: vec4<f32>, // cumulative probability, face probability, material
}
struct Energy {
    value: vec4<f32>,
    custom_visible: vec3<f32>,
}
// #include voxel_traversal.wgsl
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> directory: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read> regions: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> cells: array<u32>;
@group(0) @binding(4) var<storage, read> materials: array<Material>;
@group(0) @binding(5) var<storage, read> lights: array<Light>;
@group(0) @binding(8) var<storage, read> emitter_words: array<vec4<u32>>;
// Resident measurements (nine uint4s per slot) and the (voxel, face) -> slot
// index. A path that reaches a measured face ends there with that face's
// cached incident energy instead of tracing further bounces.
@group(0) @binding(9) var<storage, read> cache_state: array<vec4<u32>>;
@group(0) @binding(10) var<storage, read> cache_index: array<vec4<u32>>;
// A cached face ends a path once its mean's relative standard error is at
// most 1/4: 16 samples at relative sigma one, the Monte Carlo prior. A
// variance-aware test (shrunk toward that prior, as the view filter does)
// accepts quiet faces a couple of samples sooner and noisy emitter-lit ones
// later, but cannot certify a black face (zero mean, zero variance), which
// needs this count anyway. That difference was judged not worth a second
// cache load on the hottest path (2026-10-24 review; not measured).
const CACHE_MIN_SAMPLES: u32 = 16u;

struct CachedEnergy { found: bool, value: vec4<f32> }
fn cached_energy(voxel: vec3<i32>, face: u32) -> CachedEnergy {
    var cached: CachedEnergy;
    if face > 5u { return cached; }
    let header = cache_index[0];
    let mask = header.x;
    var slot = (hash_voxel(voxel) ^ pcg(face + 1u)) & mask;
    for (var attempt = 0u; attempt < header.y; attempt += 1u) {
        let entry = cache_index[slot + 1u];
        if entry.w == 0xffffffffu { return cached; }
        if all(bitcast<vec3<i32>>(entry.xyz) == voxel) && (entry.w >> 29u) == face {
            let base = (entry.w & 0x1fffffffu) * 9u;
            let words = cache_state[base + 1u];
            // The high bit marks a face changed by an edit in this batch.
            if words.y < CACHE_MIN_SAMPLES || (words.y & 0x80000000u) != 0u { return cached; }
            cached.found = true;
            cached.value = bitcast<vec4<f32>>(cache_state[base]);
            return cached;
        }
        slot = (slot + 1u) & mask;
    }
    return cached;
}

fn load_emitter(index: u32) -> EmissiveFace {
    let a = emitter_words[index * 2u];
    return EmissiveFace(bitcast<vec3<i32>>(a.xyz), a.w, bitcast<vec4<f32>>(emitter_words[index * 2u + 1u]));
}
struct LightNode {
    minimum_power: vec4<f32>,
    maximum: vec3<f32>,
    normals: u32,
    pivot: vec3<i32>,
    face: u32,
    children: vec4<u32>,
}
fn load_node(index: u32) -> LightNode {
    let offset = u32(params.extra.w) * 2u + index * 4u;
    let a = emitter_words[offset + 1u];
    let b = emitter_words[offset + 2u];
    return LightNode(bitcast<vec4<f32>>(emitter_words[offset]), bitcast<vec3<f32>>(a.xyz), a.w,
        bitcast<vec3<i32>>(b.xyz), b.w, emitter_words[offset + 3u]);
}
fn node_importance(node: LightNode, position: vec3<f32>, normal: vec3<f32>) -> f32 {
    let delta = (node.minimum_power.xyz + node.maximum) * 0.5 - position;
    let radius = length(node.maximum - node.minimum_power.xyz) * 0.5;
    let distance = max(length(delta), 1e-5);
    let direction = delta / distance;
    let spread = min(radius / distance, 1.0);
    let receiver = clamp(dot(normal, direction) + spread, 0.0, 1.0);
    // Voxel normals are the six signed axes. Reduce their projections directly;
    // an absent normal contributes -1, which remains nonpositive after spread.
    let positive = select(vec3(-1.0), -direction, (vec3(node.normals) & vec3(1u, 4u, 16u)) != vec3(0u));
    let negative = select(vec3(-1.0), direction, (vec3(node.normals) & vec3(2u, 8u, 32u)) != vec3(0u));
    let projected = max(positive, negative);
    let emitter = clamp(max(max(projected.x, projected.y), projected.z) + spread, 0.0, 1.0);
    return node.minimum_power.w * receiver * emitter / (distance * distance + radius * radius + 0.25);
}
fn node_left_probability(node: LightNode, position: vec3<f32>, normal: vec3<f32>) -> f32 {
    let left = load_node(node.children.x);
    let right = load_node(node.children.y);
    let power = left.minimum_power.w / (left.minimum_power.w + right.minimum_power.w);
    let a = node_importance(left, position, normal);
    let b = node_importance(right, position, normal);
    if a + b <= 1e-20 { return clamp(power, 1e-6, 1.0 - 1e-6); }
    return clamp(0.05 * power + 0.95 * a / (a + b), 1e-6, 1.0 - 1e-6);
}
fn tree_emitter(position: vec3<f32>, normal: vec3<f32>, choice: f32) -> EmissiveFace {
    var index = 0u;
    var uniform = choice;
    var probability = 1.0;
    loop {
        let node = load_node(index);
        if node.children.x == 0xffffffffu {
            var emitter = load_emitter(node.children.y);
            emitter.sampling.y = probability;
            return emitter;
        }
        let p = node_left_probability(node, position, normal);
        if uniform < p { probability *= p; uniform /= p; index = node.children.x; }
        else { probability *= 1.0 - p; uniform = (uniform - p) / (1.0 - p); index = node.children.y; }
        uniform = min(uniform, bitcast<f32>(0x3f7fffffu));
    }
    return EmissiveFace(vec3(0), 0u, vec4(0.0));
}
fn key_less(voxel: vec3<i32>, face: u32, node: LightNode) -> bool {
    let axis = node.children.z;
    for (var j = 0u; j < 3u; j += 1u) {
        let i = (axis + j) % 3u;
        if voxel[i] != node.pivot[i] { return voxel[i] < node.pivot[i]; }
    }
    return face < node.face;
}
fn tree_probability(position: vec3<f32>, normal: vec3<f32>, voxel: vec3<i32>, hit_normal: vec3<f32>) -> f32 {
    var face = 0u;
    for (var i = 0u; i < 6u; i += 1u) { if all(face_normal(i) == hit_normal) { face = i; } }
    var index = 0u;
    var probability = 1.0;
    loop {
        let node = load_node(index);
        if node.children.x == 0xffffffffu {
            let found = load_emitter(node.children.y);
            return select(0.0, probability, all(found.voxel == voxel) && found.face == face);
        }
        let p = node_left_probability(node, position, normal);
        if key_less(voxel, face, node) { probability *= p; index = node.children.x; }
        else { probability *= 1.0 - p; index = node.children.y; }
    }
    return 0.0;
}

// Hash-based Owen scramble: Nathan Vegdahl (sobol_burley), MIT; notice in
// licenses/sobol_burley-MIT.txt.
fn scramble_reverse(input: u32, seed: u32) -> u32 {
    var value = input;
    value ^= value * 0x3d20adeau;
    value += seed;
    value *= (seed >> 16u) | 1u;
    value ^= value * 0x05526c56u;
    value ^= value * 0x53a22864u;
    return value;
}
fn sample_1d(stream: u32, sequence: u32, dimension: u32) -> f32 {
    if dimension < 12u {
        let primes = array<u32,12>(2u,3u,5u,7u,11u,13u,17u,19u,23u,29u,31u,37u);
        let base = primes[dimension];
        var index = (sequence & 0x00ffffffu) + 1u;
        var scale = 1.0;
        var value = 0.0;
        while index != 0u {
            scale /= f32(base);
            value += f32(index % base) * scale;
            index /= base;
        }
        let rotation = f32(pcg(stream ^ pcg(dimension)) >> 9u) / 8388608.0;
        return fract(value + rotation);
    }
    let seed = pcg(stream ^ pcg(dimension / 2u));
    var index = reverseBits(scramble_reverse(reverseBits(sequence), pcg(seed ^ 0x79c68e4au)));
    if (dimension & 1u) != 0u {
        index ^= (index >> 1u) & 0x55555555u;
        index ^= (index >> 2u) & 0x33333333u;
        index ^= (index >> 4u) & 0x0f0f0f0fu;
        index ^= (index >> 8u) & 0x00ff00ffu;
        index ^= (index >> 16u) & 0x0000ffffu;
    }
    let value = reverseBits(scramble_reverse(index, pcg(seed ^ pcg(dimension))));
    return f32(value >> 9u) / 8388608.0;
}
// The original three-hit prefix uses dimensions 2..7 for scattering and
// 9..10 for roulette. Extending 2 + bounce*2 would alias earlier roulette
// dimensions at hit four. Give deeper hits distinct scrambled dimensions; keep
// the established near prefix byte-for-byte identical.
fn continuation_sample(stream: u32, sequence: u32, bounce: u32, dimension: u32) -> f32 {
    if bounce < 3u {
        let legacy = select(2u + bounce * 2u + dimension, 8u + bounce, dimension == 2u);
        return sample_1d(stream, sequence, legacy);
    }
    return sample_1d(stream, sequence, 16u + (bounce - 3u) * 3u + dimension);
}

fn face_normal(face: u32) -> vec3<f32> {
    var n = vec3(0.0);
    n[face / 2u] = select(1.0, -1.0, (face & 1u) != 0u);
    return n;
}
fn hemisphere(n: vec3<f32>, u: f32, v: f32) -> vec3<f32> {
    var tangent = vec3(0.0);
    if abs(n.z) < 0.999 { tangent = normalize(cross(n, vec3(0.0, 0.0, 1.0))); }
    else { tangent = normalize(cross(n, vec3(0.0, 1.0, 0.0))); }
    let bitangent = cross(n, tangent);
    let phi = 6.28318530718 * u;
    return tangent * (cos(phi) * sqrt(v)) + bitangent * (sin(phi) * sqrt(v)) + n * sqrt(1.0 - v);
}
fn trace(origin: vec3<f32>, direction: vec3<f32>, max_distance: f32) -> RayResult {
    var result = trace_domain_limited(origin, direction, max_distance, BUDGETED_RAYS, PATH_STEPS);
    if BUDGETED_RAYS && result.status == 2u && result.material == 0xffffffffu { result.status = 0u; }
    return result;
}

// Canonical grid-corner sun visibility of view-owned faces.
fn sky_corner_visibility(position: vec3<f32>) -> u32 {
    var word = 0x80000000u;
    // Lit bits count lights with visible color, as `sky_lights` does on CPU.
    var bit = 0u;
    for (var i = 0u; i < u32(params.maximum.w) && bit < 4u; i += 1u) {
        let light = lights[i];
        if all(max(light.energy.rgb - light.extra.yzw, vec3(0.0)) == vec3(0.0)) { continue; }
        let direction = normalize(light.position_range.xyz);
        // A canonical light-directed origin makes visibility independent of
        // the receiving face. Axis ties use the same positive grid cell as CPU.
        let origin = position + direction * 0.002;
        // Display shadows match rendered loaded geometry. Do not turn gaps in
        // the streaming directory into opaque slabs; continue to real occluders.
        // Path transport and simulation direct energy still use strict trace().
        let visibility = trace_domain(origin, direction, params.extra.y, true).status;
        if visibility == 2u { word = 0u; break; }
        if visibility == 0u { word |= 1u << bit; }
        bit += 1u;
    }
    return word;
}
fn connection_reach() -> f32 {
    if LIMITED_REACH { return params.transport.x; }
    return params.extra.y;
}
fn trace_diffuse(origin: vec3<f32>, direction: vec3<f32>) -> RayResult {
    let reach = connection_reach();
    var result = trace(origin, direction, reach);
    // A truncated miss is sky only if the remaining coarse frontier is empty.
    // Otherwise it ends at the reach with no energy: a known black sample
    // (status 3), not unloaded space. Specialization removes this policy from
    // default and shadow traversal.
    if LIMITED_REACH && result.status == 0u && geometry_ahead(origin + direction * reach, direction, params.extra.y - reach) {
        result.status = 3u;
    }
    return result;
}
fn direct_at(position: vec3<f32>, normal: vec3<f32>) -> Energy {
    var energy: Energy;
    let origin = position + normal * 0.002;
    for (var i = 0u; i < params.counts.w; i += 1u) {
        let light = lights[i];
        var direction = normalize(light.position_range.xyz);
        var distance = params.extra.y;
        var attenuation = 1.0;
        if light.position_range.w >= 0.0 {
            let delta = light.position_range.xyz - position;
            let d2 = dot(delta, delta);
            if d2 <= 0.0000001 || d2 > light.position_range.w * light.position_range.w { continue; }
            distance = sqrt(d2);
            direction = delta / distance;
            distance -= 0.004;
            attenuation = 1.0 / max(d2, 1.0);
        }
        let cosine = max(dot(normal, direction), 0.0);
        if cosine > 0.0 && trace(origin, direction, distance).status == 0u {
            energy.value += light.energy * (cosine * attenuation);
            energy.custom_visible += light.extra.yzw * (cosine * attenuation);
        }
    }
    return energy;
}

fn choose_emitter(position: vec3<f32>, normal: vec3<f32>, choice: f32) -> EmissiveFace {
    let count = u32(params.extra.w);
    var emitter: EmissiveFace;
    if load_emitter(0u).sampling.w > 0.0 { emitter = tree_emitter(position, normal, choice); }
    else {
        var low = 0u;
        var high = count;
        while low < high {
            let middle = (low + high) / 2u;
            if load_emitter(middle).sampling.x <= choice { low = middle + 1u; }
            else { high = middle; }
        }
        emitter = load_emitter(min(low, count - 1u));
    }
    return emitter;
}

fn cosine_pdf(normal: vec3<f32>, direction: vec3<f32>) -> f32 {
    return max(dot(normal, direction), 0.0) / 3.14159265359;
}

// Emitter next-event estimate, balanced against cosine sampling of the same
// directions (the continuation and terminal mixture use the same density).
fn sample_emitter(position: vec3<f32>, normal: vec3<f32>, source_stream: u32, sequence: u32, bounce: u32) -> Energy {
    var result: Energy;
    if params.extra.w == 0.0 { return result; }
    let stream = source_stream ^ pcg(bounce + 0xa341316cu);
    let emitter = choose_emitter(position, normal, sample_1d(stream, sequence, 12u));
    let emitter_normal = face_normal(emitter.face);
    let axis = emitter.face / 2u;
    let tangent = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), axis == 0u);
    let bitangent = select(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), axis == 2u);
    let light_position = vec3<f32>(emitter.voxel) + vec3(0.5) + emitter_normal * 0.5
        + tangent * (sample_1d(stream, sequence, 13u) - 0.5) + bitangent * (sample_1d(stream, sequence, 14u) - 0.5);
    let delta = light_position - position;
    let d2 = dot(delta, delta);
    if d2 < 0.0001 || d2 > connection_reach() * connection_reach() { return result; }
    let distance = sqrt(d2);
    let direction = delta / distance;
    let receiver_cosine = max(dot(normal, direction), 0.0);
    let emitter_cosine = max(dot(emitter_normal, -direction), 0.0);
    if receiver_cosine <= 0.0 || emitter_cosine <= 0.0 || emitter.sampling.y <= 0.0 { return result; }
    let light_pdf = emitter.sampling.y * d2 / emitter_cosine;
    let directional_pdf = receiver_cosine / 3.14159265359;
    // Visibility connects the offset endpoints, not a translated copy of the
    // physical ray. A fixed length trim falsely occludes grazing light samples.
    // The physical points above still define the geometry factors and PDFs.
    let shadow_origin = position + normal * 0.002;
    let shadow_delta = light_position + emitter_normal * 0.002 - shadow_origin;
    let shadow_distance = length(shadow_delta);
    if trace(shadow_origin, shadow_delta / shadow_distance, shadow_distance).status != 0u { return result; }
    let material = materials[u32(emitter.sampling.z)];
    let factor = directional_pdf / (light_pdf + directional_pdf);
    result.value = material.emission * factor;
    return result;
}
// One terminal emission connection from a 50/50 light/cosine mixture for the
// last explicit hit. Only uncached hits reach it (cold start, cache misses).
// Escaping sky and nonemitting hits contribute zero here; no clamp is added.
fn sample_terminal_emission(position: vec3<f32>, normal: vec3<f32>, stream: u32, sequence: u32, bounce: u32) -> Energy {
    var result: Energy;
    if params.extra.w == 0.0 { return result; }
    if continuation_sample(stream, sequence, bounce, 2u) < 0.5 {
        result = sample_emitter(position, normal, stream, sequence, bounce + 1u);
        result.value *= 2.0;
        return result;
    }
    let direction = hemisphere(normal, continuation_sample(stream, sequence, bounce, 0u), continuation_sample(stream, sequence, bounce, 1u));
    if dot(normal, direction) <= 0.0 { return result; }
    let hit = trace(position + normal * 0.002, direction, connection_reach());
    if hit.status != 1u { return result; }
    let material = materials[hit.material];
    let power = material.emission_extra.y;
    let delta = hit.position - position;
    let d2 = dot(delta, delta);
    if power == 0.0 || d2 < 0.0001 || d2 > connection_reach() * connection_reach() { return result; }
    var selection_pdf = power / max(params.extra.z, 1e-20);
    if load_emitter(0u).sampling.w > 0.0 {
        selection_pdf = tree_probability(position, normal, hit.voxel, hit.normal);
    }
    let light_pdf = selection_pdf * d2 / max(dot(hit.normal, -direction), 1e-8);
    let directional_pdf = cosine_pdf(normal, direction);
    let factor = 2.0 * directional_pdf / (directional_pdf + light_pdf);
    result.value = material.emission * factor;
    return result;
}

struct PathEnergy {
    energy: Energy,
    // Reached an unloaded region: no information, excluded from the count.
    unknown: bool,
    // Retained by the view consumer for reconnection/replay. World-space
    // moment reduction does not consume this field (the compiler removes it).
    first_hit: RayResult,
}
// Explicit state lets a view path pause at a dispatch boundary without
// turning the work budget into a physical path-depth cutoff. The near consumer
// still requests its existing finite prefix and terminal emission mixture.
struct PathState {
    result: PathEnergy,
    throughput: Energy,
    origin: vec3<f32>,
    direction: vec3<f32>,
    previous_position: vec3<f32>,
    previous_normal: vec3<f32>,
    previous_pdf: f32,
    bounce: u32,
    done: bool,
}
fn begin_path(start: vec3<f32>, direction: vec3<f32>, normal: vec3<f32>) -> PathState {
    var state: PathState;
    state.throughput = Energy(vec4(1.0), vec3(0.0));
    state.origin = start;
    state.direction = direction;
    state.previous_normal = normal;
    state.previous_pdf = cosine_pdf(normal, direction);
    state.previous_position = start - normal * 0.001;
    state.done = state.previous_pdf <= 0.0;
    return state;
}
fn advance_path(initial: PathState, stream: u32, sequence: u32, steps: u32, terminal: bool) -> PathState {
    var state = initial;
    if state.done { return state; }
    for (var step = 0u; step < steps; step += 1u) {
        let bounce = state.bounce;
        // Every exit below completes the path. Only a surviving continuation
        // clears this flag, allowing the dispatch boundary to pause it.
        state.done = true;
        let hit = trace_diffuse(state.origin, state.direction);
        if bounce == 0u { state.result.first_hit = hit; }
        if hit.status == 0u {
            let horizon = clamp(state.direction.y * 0.35 + 0.65, 0.15, 1.0);
            state.result.energy.value += state.throughput.value * params.environment * horizon;
            break;
        }
        if hit.status == 2u { state.result.unknown = true; break; }
        if hit.status == 3u { break; }
        let material = materials[hit.material];
        let direct = direct_at(hit.position, hit.normal);
        let emission_weight = material.emission_extra.y;
        let delta = hit.position - state.origin;
        var selection_pdf = emission_weight / max(params.extra.z, 1e-20);
        if emission_weight > 0.0 && params.extra.w > 0.0 && load_emitter(0u).sampling.w > 0.0 {
            selection_pdf = tree_probability(state.previous_position, state.previous_normal, hit.voxel, hit.normal);
        }
        let light_pdf = selection_pdf * dot(delta, delta) / max(dot(hit.normal, -state.direction), 1e-8);
        var mis = 1.0;
        if light_pdf > 0.0 { mis = state.previous_pdf / (state.previous_pdf + light_pdf); }
        // A measured face already holds the incident energy that NEE plus a
        // continued path would estimate here: end the path with it.
        let cached = cached_energy(hit.voxel, face_index_of(hit.normal));
        if cached.found {
            state.result.energy.value += state.throughput.value * (material.emission * mis + material.diffuse * (direct.value + cached.value));
            break;
        }
        var nee: Energy;
        if terminal && step + 1u == steps {
            nee = sample_terminal_emission(hit.position, hit.normal, stream, sequence, bounce);
        } else {
            nee = sample_emitter(hit.position, hit.normal, stream, sequence, bounce + 1u);
        }
        state.result.energy.value += state.throughput.value * (material.emission * mis + material.diffuse * (direct.value + nee.value));
        if terminal && step + 1u == steps { break; }
        state.throughput.value *= material.diffuse;
        let maximum = max(state.throughput.value.x, max(state.throughput.value.y, state.throughput.value.z));
        // A small weight can still reach a bright source. Only exact zero is
        // safely discarded; roulette compensates surviving nonzero paths.
        if maximum == 0.0 { break; }
        if bounce >= 1u {
            let survival = clamp(maximum, 0.1, 0.95);
            if continuation_sample(stream, sequence, bounce, 2u) > survival { break; }
            state.throughput.value /= survival;
        }
        state.origin = hit.position + hit.normal * 0.002;
        state.previous_position = hit.position;
        state.previous_normal = hit.normal;
        state.direction = hemisphere(hit.normal, continuation_sample(stream, sequence, bounce, 0u), continuation_sample(stream, sequence, bounce, 1u));
        state.previous_pdf = cosine_pdf(hit.normal, state.direction);
        if state.previous_pdf <= 0.0 { break; }
        state.bounce += 1u;
        state.done = false;
    }
    return state;
}

fn trace_path(start: vec3<f32>, first_direction: vec3<f32>, initial_normal: vec3<f32>, stream: u32, sequence: u32) -> PathEnergy {
    return advance_path(begin_path(start, first_direction, initial_normal), stream, sequence, DIFFUSE_HITS, true).result;
}

struct ReceiverSample { energy: Energy, known: bool }
struct ReceiverPath {
    emitter: Energy,
    continuation: PathEnergy,
    direction: vec3<f32>,
}
fn sample_receiver_path(surface: vec3<f32>, normal: vec3<f32>, stream: u32, sequence: u32) -> ReceiverPath {
    var result: ReceiverPath;
    result.direction = hemisphere(normal, sample_1d(stream, sequence, 0u), sample_1d(stream, sequence, 1u));
    result.emitter = sample_emitter(surface, normal, stream, sequence, 0u);
    // Cosine sampling: cosine / pdf is exactly one for the continuation.
    if dot(normal, result.direction) > 0.0 {
        result.continuation = trace_path(surface + normal * 0.001, result.direction, normal, stream, sequence);
    }
    return result;
}

fn trace_receiver(surface: vec3<f32>, normal: vec3<f32>, stream: u32, sequence: u32) -> ReceiverSample {
    let path = sample_receiver_path(surface, normal, stream, sequence);
    var result = ReceiverSample(path.emitter, !path.continuation.unknown);
    result.energy.value += path.continuation.energy.value;
    return result;
}
