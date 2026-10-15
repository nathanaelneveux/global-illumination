// World-space face transport. No camera, framebuffer or screen-space history.
// Layouts and sampling dimensions are mirrored by gpu.rs and sampling.rs.
// Bounded, lazily cached quality/layout variants, not per-face loop controls.
override DIFFUSE_HITS: u32 = 3u;
override LIMITED_REACH: bool = false;
override PATH_LANES: u32 = 4u;
struct Params {
    counts: vec4<u32>, // tasks, samples per face, directory mask, light count
    minimum: vec4<i32>, // xyz: scene bounds, w: shared corner task count
    maximum: vec4<i32>, // xyz: scene bounds, w: directional light count
    environment: vec4<f32>,
    extra: vec4<f32>, // environment ionizing, directional horizon, emitter power/count
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
struct FaceTask {
    voxel: vec3<i32>,
    face: u32,
    sequence: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}
struct FaceResult {
    mean: vec4<f32>,
    mean_extra: vec4<f32>,
    second: vec4<f32>,
    second_extra: vec4<f32>, // ionizing second moment, padding
    direct: vec4<f32>,
    direct_extra: vec4<f32>,
}
struct EmissiveFace {
    voxel: vec3<i32>,
    face: u32,
    sampling: vec4<f32>, // cumulative probability, face probability, material
}
struct Energy {
    value: vec4<f32>,
    radiation: f32,
    custom_visible: vec3<f32>,
}
struct RayResult {
    voxel: vec3<i32>,
    material: u32,
    position: vec3<f32>,
    status: u32, // 0 escaped, 1 hit, 2 unknown
    normal: vec3<f32>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> directory: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read> regions: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> cells: array<u32>;
@group(0) @binding(4) var<storage, read> materials: array<Material>;
@group(0) @binding(5) var<storage, read> lights: array<Light>;
@group(0) @binding(6) var<storage, read> tasks: array<FaceTask>;
// Six float4s per face, followed by two packed uint4s per shared sky corner.
@group(0) @binding(7) var<storage, read_write> results: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> emitter_words: array<vec4<u32>>;

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

fn pcg(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn hash_voxel(voxel: vec3<i32>) -> u32 {
    return pcg(pcg(pcg(bitcast<u32>(voxel.x)) ^ bitcast<u32>(voxel.y)) ^ bitcast<u32>(voxel.z));
}
// Hash-based Owen scramble: Nathan Vegdahl, MIT, licenses/sobol_burley-MIT.txt.
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
fn lookup_region(coordinate: vec3<i32>) -> u32 {
    var slot = hash_voxel(coordinate) & params.counts.z;
    for (var attempt = 0u; attempt < 32u; attempt += 1u) {
        let entry = directory[slot];
        if entry.w == -1 { return 0xffffffffu; }
        if all(entry.xyz == coordinate) { return u32(entry.w); }
        slot = (slot + 1u) & params.counts.z;
    }
    return 0xffffffffu;
}
fn in_bounds(region: vec3<i32>) -> bool {
    return all(region >= params.minimum.xyz) && all(region <= params.maximum.xyz);
}
fn unpack_bytes(value: u32) -> vec3<i32> {
    return vec3<i32>(i32(value & 255u), i32((value >> 8u) & 255u), i32((value >> 16u) & 255u));
}
fn negative_extents(bits: u32) -> vec3<i32> {
    return vec3<i32>(i32(bits & 3u), i32((bits >> 4u) & 3u), i32((bits >> 8u) & 3u));
}
fn positive_extents(bits: u32) -> vec3<i32> {
    return vec3<i32>(i32((bits >> 2u) & 3u), i32((bits >> 6u) & 3u), i32((bits >> 10u) & 3u));
}
fn advance_box(position: vec3<f32>, direction: vec3<f32>, cell: vec3<i32>, negative: vec3<i32>, positive: vec3<i32>, size: f32) -> vec4<f32> {
    let minimum = vec3<f32>(cell - negative) * size;
    let maximum = vec3<f32>(cell + positive + vec3(1)) * size;
    let moving = abs(direction) > vec3(1e-8);
    let plane = select(minimum, maximum, direction > vec3(0.0));
    let raw = (plane - position) / select(vec3(1.0), direction, moving);
    let travel = select(vec3(1e30), raw, moving & (raw >= vec3(-0.001)) & (raw < vec3(1e30)));
    // Match the scalar XYZ ordering, including a tiny negative crossing after
    // roundoff: each accepted crossing is clamped before considering the next.
    let choose_y = travel.y < max(travel.x, 0.0);
    let distance_xy = max(select(travel.x, travel.y, choose_y), 0.0);
    let choose_z = travel.z < distance_xy;
    let distance = max(select(distance_xy, travel.z, choose_z), 0.0);
    let axis_xy = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), choose_y);
    let axis = select(axis_xy, vec3(0.0, 0.0, 1.0), choose_z);
    let normal = select(vec3(0.0), axis * select(vec3(1.0), vec3(-1.0), direction > vec3(0.0)), distance < 1e30);
    return vec4(normal, max(distance + 0.001, 0.001));
}
fn geometry_ahead(start: vec3<f32>, direction: vec3<f32>, max_distance: f32) -> bool {
    var position = start;
    var remaining = max_distance;
    for (var step = 0u; step < 2048u; step += 1u) {
        let region = vec3<i32>(floor(position / 16.0));
        if remaining < 0.0 || !in_bounds(region) { return false; }
        let index = lookup_region(region);
        if index != 0xffffffffu && regions[index].x != 1u { return true; }
        let travel = advance_box(position, direction, region, vec3(0), vec3(0), 16.0).w;
        position += direction * travel;
        remaining -= travel;
    }
    return true;
}
fn trace_domain(origin: vec3<f32>, direction: vec3<f32>, max_distance: f32, visible_geometry: bool) -> RayResult {
    var result: RayResult;
    var near = -1e30;
    var far = 1e30;
    var normal = vec3(0.0);
    let minimum = vec3<f32>(params.minimum.xyz) * 16.0;
    let maximum = vec3<f32>(params.maximum.xyz + vec3(1)) * 16.0;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if abs(direction[axis]) <= 1e-8 {
            if origin[axis] < minimum[axis] || origin[axis] > maximum[axis] { return result; }
            continue;
        }
        let first = (minimum[axis] - origin[axis]) / direction[axis];
        let second = (maximum[axis] - origin[axis]) / direction[axis];
        if min(first, second) > near {
            near = min(first, second);
            normal = vec3(0.0);
            normal[axis] = select(1.0, -1.0, first < second);
        }
        far = min(far, max(first, second));
    }
    if far < max(near, 0.0) { return result; }
    var distance = max(near, 0.0) + 0.001;
    var position = origin + direction * distance;
    var last_region = vec3<i32>(0x7fffffff);
    var metadata = vec4<u32>(0u);
    for (var step = 0u; step < 2048u; step += 1u) {
        if distance > max_distance { return result; }
        let voxel = vec3<i32>(floor(position));
        let region = voxel >> vec3<u32>(4u);
        if !in_bounds(region) { return result; }
        if any(last_region != region) {
            let index = lookup_region(region);
            if index == 0xffffffffu {
                if visible_geometry {
                    let advance = advance_box(position, direction, region, vec3(0), vec3(0), 16.0);
                    distance += advance.w;
                    position += direction * advance.w;
                    normal = advance.xyz;
                    continue;
                }
                result.status = select(0u, 2u, geometry_ahead(position, direction, max_distance - distance));
                return result;
            }
            metadata = regions[index];
            last_region = region;
        }
        var advance = vec4(0.0);
        if metadata.x == 1u {
            advance = advance_box(position, direction, region, unpack_bytes(metadata.z), unpack_bytes(metadata.w), 16.0);
        } else {
            var material = metadata.y;
            if metadata.x == 3u {
                let local = vec3<u32>(voxel & vec3<i32>(15));
                let brick = local / 4u;
                let brick_data = cells[metadata.y + (brick.z * 4u + brick.y) * 4u + brick.x];
                let brick_state = brick_data & 3u;
                if brick_state == 0u {
                    let extents = brick_data >> 2u;
                    advance = advance_box(position, direction, region * 4 + vec3<i32>(brick), negative_extents(extents), positive_extents(extents), 4.0);
                    material = 256u;
                } else if brick_state == 1u {
                    material = brick_data >> 2u;
                } else {
                    let inner = local & vec3<u32>(3u);
                    let index = (inner.z * 4u + inner.y) * 4u + inner.x;
                    let word = cells[(brick_data >> 2u) + index / 2u];
                    let cell = (word >> ((index & 1u) * 16u)) & 65535u;
                    material = select(256u, cell >> 1u, (cell & 1u) != 0u);
                    if material == 256u {
                        let extents = cell >> 1u;
                        advance = advance_box(position, direction, voxel, negative_extents(extents), positive_extents(extents), 1.0);
                    }
                }
            }
            if material != 256u {
                result.voxel = voxel;
                result.material = material;
                result.position = position;
                result.normal = normal;
                result.status = 1u;
                return result;
            }
        }
        distance += advance.w;
        position += direction * advance.w;
        normal = advance.xyz;
    }
    result.status = 2u;
    return result;
}
fn trace(origin: vec3<f32>, direction: vec3<f32>, max_distance: f32) -> RayResult {
    return trace_domain(origin, direction, max_distance, false);
}
fn connection_reach() -> f32 {
    if LIMITED_REACH { return params.transport.x; }
    return params.extra.y;
}
fn trace_diffuse(origin: vec3<f32>, direction: vec3<f32>) -> RayResult {
    let reach = connection_reach();
    var result = trace(origin, direction, reach);
    // A truncated miss is sky only if the remaining coarse frontier is empty.
    // Specialization removes this policy from default and shadow traversal.
    if LIMITED_REACH && result.status == 0u && geometry_ahead(origin + direction * reach, direction, params.extra.y - reach) {
        result.status = 2u;
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
            energy.radiation += light.extra.x * (cosine * attenuation);
            energy.custom_visible += light.extra.yzw * (cosine * attenuation);
        }
    }
    return energy;
}

fn pack_sky(value: vec3<f32>) -> u32 {
    let maximum = max(value.x, max(value.y, value.z));
    if maximum < 1e-30 { return 1u; } // measured black, not unknown
    let exponent = (bitcast<u32>(maximum) >> 23u) & 255u;
    let quantum = exp2(f32(exponent) - 134.0);
    let mantissa = vec3<u32>(min(round(max(value, vec3(0.0)) / quantum), vec3(255.0)));
    return mantissa.x | (mantissa.y << 8u) | (mantissa.z << 16u) | (exponent << 24u);
}

@compute @workgroup_size(64)
fn trace_sky_corners(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u32(params.minimum.w) { return; }
    let position = vec3<f32>(tasks[params.counts.x + id.x].voxel);
    var values: array<vec3<f32>, 6>;
    var valid = true;
    for (var i = 0u; i < u32(params.maximum.w); i += 1u) {
        let light = lights[i];
        let direction = normalize(light.position_range.xyz);
        let energy = max(light.energy.rgb - light.extra.yzw, vec3(0.0));
        if all(energy == vec3(0.0)) { continue; }
        // A canonical light-directed origin makes visibility independent of
        // the receiving face. Axis ties use the same positive grid cell as CPU.
        let origin = position + direction * 0.002;
        // Display shadows match rendered loaded geometry. Do not turn gaps in
        // the streaming directory into opaque slabs; continue to real occluders.
        // Path transport and simulation direct energy still use strict trace().
        let visibility = trace_domain(origin, direction, params.extra.y, true).status;
        if visibility == 2u { valid = false; break; }
        if visibility == 0u {
            for (var face = 0u; face < 6u; face += 1u) {
                values[face] += energy * max(dot(face_normal(face), direction), 0.0);
            }
        }
    }
    let offset = params.counts.x * 6u + id.x * 2u;
    results[offset] = select(vec4(0u), vec4(pack_sky(values[0]), pack_sky(values[1]), pack_sky(values[2]), pack_sky(values[3])), valid);
    results[offset + 1u] = select(vec4(0u), vec4(pack_sky(values[4]), pack_sky(values[5]), 0u, 0u), valid);
}
fn sample_emitter(position: vec3<f32>, normal: vec3<f32>, source_stream: u32, sequence: u32, bounce: u32) -> Energy {
    var result: Energy;
    let count = u32(params.extra.w);
    if count == 0u { return result; }
    let stream = source_stream ^ pcg(bounce + 0xa341316cu);
    let choice = sample_1d(stream, sequence, 12u);
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
    let cosine_pdf = receiver_cosine / 3.14159265359;
    if trace(position + normal * 0.002, direction, distance - 0.008).status != 0u { return result; }
    let material = materials[u32(emitter.sampling.z)];
    let factor = cosine_pdf / (light_pdf + cosine_pdf);
    result.value = material.emission * factor;
    result.radiation = material.emission_extra.x * factor;
    return result;
}
// One terminal emission connection drawn from a 50/50 light/cosine mixture.
// The finite-bounce integrand is unchanged: escaping sky and nonemitting hits
// contribute zero here. No brightness clamp or extra connection ray is added.
fn sample_terminal_emission(position: vec3<f32>, normal: vec3<f32>, stream: u32, sequence: u32, bounce: u32) -> Energy {
    var result: Energy;
    if params.extra.w == 0.0 { return result; }
    if sample_1d(stream, sequence, 8u + bounce) < 0.5 {
        result = sample_emitter(position, normal, stream, sequence, bounce + 1u);
        result.value *= 2.0;
        result.radiation *= 2.0;
        return result;
    }
    let direction = hemisphere(normal, sample_1d(stream, sequence, 2u + bounce * 2u), sample_1d(stream, sequence, 3u + bounce * 2u));
    let hit = trace(position + normal * 0.002, direction, connection_reach());
    if hit.status != 1u { return result; }
    let material = materials[hit.material];
    let power = max(max(max(material.emission.x, material.emission.y), max(material.emission.z, material.emission.w)), material.emission_extra.x);
    let delta = hit.position - position;
    let d2 = dot(delta, delta);
    if power == 0.0 || d2 < 0.0001 || d2 > connection_reach() * connection_reach() { return result; }
    var selection_pdf = power / max(params.extra.z, 1e-20);
    if load_emitter(0u).sampling.w > 0.0 {
        selection_pdf = tree_probability(position, normal, hit.voxel, hit.normal);
    }
    let light_pdf = selection_pdf * d2 / max(dot(hit.normal, -direction), 1e-8);
    let cosine_pdf = max(dot(normal, direction), 0.0) / 3.14159265359;
    let factor = 2.0 * cosine_pdf / (cosine_pdf + light_pdf);
    result.value = material.emission * factor;
    result.radiation = material.emission_extra.x * factor;
    return result;
}
// Diffuse suffix reuse reconnects only the first segment. Primary emitter NEE
// stays local and first-hit emission MIS is re-evaluated for each receiver.
struct RecordedPath {
    first: RayResult,
    direction: vec3<f32>,
    suffix: Energy,
}
fn trace_path_record(start: vec3<f32>, first_direction: vec3<f32>, initial_normal: vec3<f32>, stream: u32, sequence: u32, split_first: bool) -> RecordedPath {
    var result: Energy;
    var record: RecordedPath;
    record.direction = first_direction;
    var throughput = Energy(vec4(1.0), 1.0, vec3(0.0));
    var origin = start;
    var direction = first_direction;
    var previous_normal = initial_normal;
    var previous_position = start - initial_normal * 0.001;
    for (var bounce = 0u; bounce < DIFFUSE_HITS; bounce += 1u) {
        let hit = trace_diffuse(origin, direction);
        if bounce == 0u { record.first = hit; }
        if hit.status == 0u {
            let horizon = clamp(direction.y * 0.35 + 0.65, 0.15, 1.0);
            result.value += throughput.value * params.environment * horizon;
            result.radiation += throughput.radiation * params.extra.x * horizon;
            break;
        }
        if hit.status == 2u { break; }
        let material = materials[hit.material];
        let direct = direct_at(hit.position, hit.normal);
        let emission_weight = max(max(max(material.emission.x, material.emission.y), max(material.emission.z, material.emission.w)), material.emission_extra.x);
        let cosine_pdf = max(dot(previous_normal, direction), 0.0) / 3.14159265359;
        let delta = hit.position - origin;
        var selection_pdf = emission_weight / max(params.extra.z, 1e-20);
        if emission_weight > 0.0 && params.extra.w > 0.0 && load_emitter(0u).sampling.w > 0.0 {
            selection_pdf = tree_probability(previous_position, previous_normal, hit.voxel, hit.normal);
        }
        let light_pdf = selection_pdf * dot(delta, delta) / max(dot(hit.normal, -direction), 1e-8);
        var mis = 1.0;
        if light_pdf > 0.0 { mis = cosine_pdf / (cosine_pdf + light_pdf); }
        if split_first && bounce == 0u { mis = 0.0; }
        var nee: Energy;
        if bounce + 1u == DIFFUSE_HITS {
            nee = sample_terminal_emission(hit.position, hit.normal, stream, sequence, bounce);
        } else {
            nee = sample_emitter(hit.position, hit.normal, stream, sequence, bounce + 1u);
        }
        result.value += throughput.value * (material.emission * mis + material.diffuse * (direct.value + nee.value));
        result.radiation += throughput.radiation * (material.emission_extra.x * mis + material.diffuse_extra.x * (direct.radiation + nee.radiation));
        if bounce + 1u == DIFFUSE_HITS { break; }
        throughput.value *= material.diffuse;
        throughput.radiation *= material.diffuse_extra.x;
        let maximum = max(max(max(throughput.value.x, throughput.value.y), max(throughput.value.z, throughput.value.w)), throughput.radiation);
        if maximum < 0.002 { break; }
        if bounce >= 1u {
            let survival = clamp(maximum, 0.1, 0.95);
            if sample_1d(stream, sequence, 8u + bounce) > survival { break; }
            throughput.value /= survival;
            throughput.radiation /= survival;
        }
        origin = hit.position + hit.normal * 0.002;
        previous_position = hit.position;
        previous_normal = hit.normal;
        direction = hemisphere(hit.normal, sample_1d(stream, sequence, 2u + bounce * 2u), sample_1d(stream, sequence, 3u + bounce * 2u));
    }
    record.suffix = result;
    return record;
}

fn trace_path(start: vec3<f32>, direction: vec3<f32>, normal: vec3<f32>, stream: u32, sequence: u32) -> Energy {
    return trace_path_record(start, direction, normal, stream, sequence, false).suffix;
}

fn evaluate_record(record: RecordedPath, origin: vec3<f32>, normal: vec3<f32>) -> Energy {
    var energy = record.suffix;
    if record.first.status != 1u { return energy; }
    let hit = record.first;
    let material = materials[hit.material];
    let delta = hit.position - origin;
    let direction = normalize(delta);
    let power = max(max(max(material.emission.x, material.emission.y), max(material.emission.z, material.emission.w)), material.emission_extra.x);
    var probability = power / max(params.extra.z, 1e-20);
    if power > 0.0 && params.extra.w > 0.0 && load_emitter(0u).sampling.w > 0.0 {
        probability = tree_probability(origin - normal * 0.001, normal, hit.voxel, hit.normal);
    }
    let cosine_pdf = max(dot(normal, direction), 0.0) / 3.14159265359;
    let light_pdf = probability * dot(delta, delta) / max(dot(hit.normal, -direction), 1e-8);
    var weight = 1.0;
    if light_pdf > 0.0 { weight = cosine_pdf / (cosine_pdf + light_pdf); }
    energy.value += material.emission * weight;
    energy.radiation += material.emission_extra.x * weight;
    return energy;
}

// Density in the shared first-hit area measure (solid angle for sky).
fn record_pdf(record: RecordedPath, origin: vec3<f32>, normal: vec3<f32>) -> f32 {
    if record.first.status == 2u { return 0.0; }
    if record.first.status == 0u { return max(dot(normal, record.direction), 0.0) / 3.14159265359; }
    let delta = record.first.position - origin;
    let d2 = dot(delta, delta);
    if d2 < 0.0000001 || d2 > connection_reach() * connection_reach() { return 0.0; }
    let direction = delta / sqrt(d2);
    return max(dot(normal, direction), 0.0) * max(dot(record.first.normal, -direction), 0.0) / (3.14159265359 * d2);
}
fn record_visible(record: RecordedPath, origin: vec3<f32>) -> bool {
    if record.first.status == 0u { return trace_diffuse(origin, record.direction).status == 0u; }
    let delta = record.first.position - origin;
    let distance = length(delta);
    return trace(origin, delta / distance, max(distance - 0.008, 0.0)).status == 0u;
}

// Reject ill-conditioned shifts symmetrically in both terms of the MIS
// partition. The original sample retains weight one when reuse is rejected;
// no radiance clipping or sample-dependent energy selection is introduced.
fn compatible_pdf(record: RecordedPath, origin: vec3<f32>, normal: vec3<f32>, reference_pdf: f32) -> f32 {
    let other_pdf = record_pdf(record, origin, normal);
    if min(reference_pdf, other_pdf) < 0.25 * max(reference_pdf, other_pdf) || other_pdf <= 0.0 { return 0.0; }
    if !record_visible(record, origin) { return 0.0; }
    return other_pdf;
}

// Reciprocal pairs evaluate each first-vertex reconnection once, then publish
// the two weighted energies. Three uint4s per path; no geometry readback.
fn path_offset(face: u32, sample: u32) -> u32 {
    return params.counts.x * 6u + u32(params.minimum.w) * 2u + (face * params.counts.y + sample) * 3u;
}
@compute @workgroup_size(64)
fn trace_faces(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.counts.x { return; }
    let task = tasks[id.x];
    let normal = face_normal(task.face);
    let center = vec3<f32>(task.voxel) + vec3(0.5);
    let origin = center + normal * 0.501;
    let stream = hash_voxel(task.voxel) ^ pcg(task.face + 1u);
    var result: FaceResult;
    let direct = direct_at(center + normal * 0.5, normal);
    result.direct = direct.value;
    result.direct_extra.x = direct.radiation;
    result.direct_extra.y = direct.custom_visible.x;
    result.direct_extra.z = direct.custom_visible.y;
    result.direct_extra.w = direct.custom_visible.z;
    for (var i = 0u; i < params.counts.y; i += 1u) {
        let sequence = task.sequence + i;
        let direction = hemisphere(normal, sample_1d(stream, sequence, 0u), sample_1d(stream, sequence, 1u));
        let sample = trace_path(origin, direction, normal, stream, sequence);
        let nee = sample_emitter(center + normal * 0.5, normal, stream, sequence, 0u);
        let value = sample.value + nee.value;
        let radiation = sample.radiation + nee.radiation;
        result.mean += value;
        result.mean_extra.x += radiation;
        result.second += value * value;
        result.second_extra.x += radiation * radiation;
    }
    let inverse = 1.0 / f32(params.counts.y);
    result.mean *= inverse;
    result.mean_extra *= inverse;
    result.second *= inverse;
    result.second_extra *= inverse;
    let offset = id.x * 6u;
    results[offset] = bitcast<vec4<u32>>(result.mean);
    results[offset + 1u] = bitcast<vec4<u32>>(result.mean_extra);
    results[offset + 2u] = bitcast<vec4<u32>>(result.second);
    results[offset + 3u] = bitcast<vec4<u32>>(result.second_extra);
    results[offset + 4u] = bitcast<vec4<u32>>(result.direct);
    results[offset + 5u] = bitcast<vec4<u32>>(result.direct_extra);
}

@compute @workgroup_size(64)
fn trace_paths_recorded(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32) {
    let face = id.x / PATH_LANES;
    if face >= params.counts.x { return; }
    let task = tasks[face];
    let normal = face_normal(task.face);
    let origin = vec3<f32>(task.voxel) + vec3(0.5) + normal * 0.501;
    let stream = hash_voxel(task.voxel) ^ pcg(task.face + 1u);
    let has_partner = task.padding0 > 0u && task.padding0 <= params.counts.x;
    var other_origin = origin;
    if has_partner { other_origin = vec3<f32>(tasks[task.padding0 - 1u].voxel) + vec3(0.5) + normal * 0.501; }
    for (var i = local % PATH_LANES; i < params.counts.y; i += PATH_LANES) {
        let sequence = task.sequence + i;
        let direction = hemisphere(normal, sample_1d(stream, sequence, 0u), sample_1d(stream, sequence, 1u));
        let record = trace_path_record(origin, direction, normal, stream, sequence, true);
        var own = evaluate_record(record, origin, normal);
        var shifted = Energy(vec4(0.0), 0.0, vec3(0.0));
        if has_partner {
            let q = record_pdf(record, origin, normal);
            if q > 0.0 {
                let other_q = compatible_pdf(record, other_origin, normal, q);
                own.value *= q / (q + other_q);
                own.radiation *= q / (q + other_q);
                if other_q > 0.0 {
                    shifted = evaluate_record(record, other_origin, normal);
                    shifted.value *= other_q / (q + other_q);
                    shifted.radiation *= other_q / (q + other_q);
                }
            }
        }
        let offset = path_offset(face, i);
        results[offset] = bitcast<vec4<u32>>(own.value);
        results[offset+1u] = bitcast<vec4<u32>>(shifted.value);
        results[offset+2u] = vec4(bitcast<u32>(own.radiation), bitcast<u32>(shifted.radiation), 0u, 0u);
    }
}

@compute @workgroup_size(64)
fn resolve_reused_faces(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32) {
    let face = id.x / PATH_LANES;
    let lane = local % PATH_LANES;
    var moment: PathMoments;
    if face < params.counts.x {
        let task = tasks[face];
        let normal = face_normal(task.face);
        let center = vec3<f32>(task.voxel) + vec3(0.5);
        let origin = center + normal * 0.501;
        let stream = hash_voxel(task.voxel) ^ pcg(task.face + 1u);
        let has_partner = task.padding0 > 0u && task.padding0 <= params.counts.x;
        let partner = task.padding0 - 1u;
        for (var i = lane; i < params.counts.y; i += PATH_LANES) {
            let offset = path_offset(face, i);
            var energy = Energy(bitcast<vec4<f32>>(results[offset]), bitcast<f32>(results[offset+2u].x), vec3(0.0));
            if has_partner {
                let donated = path_offset(partner, i);
                energy.value += bitcast<vec4<f32>>(results[donated+1u]);
                energy.radiation += bitcast<f32>(results[donated+2u].y);
            }
            let nee = sample_emitter(center + normal * 0.5, normal, stream, task.sequence + i, 0u);
            let value = energy.value + nee.value;
            let radiation = energy.radiation + nee.radiation;
            moment.mean += value;
            moment.second += value * value;
            moment.radiation += vec2(radiation, radiation * radiation);
        }
    }
    path_moments[local] = moment;
    workgroupBarrier();
    if lane != 0u || face >= params.counts.x { return; }
    var total: PathMoments;
    for (var i = 0u; i < PATH_LANES; i += 1u) {
        total.mean += path_moments[local + i].mean;
        total.second += path_moments[local + i].second;
        total.radiation += path_moments[local + i].radiation;
    }
    let task = tasks[face];
    let normal = face_normal(task.face);
    let direct = direct_at(vec3<f32>(task.voxel) + vec3(0.5) + normal * 0.5, normal);
    let inverse = 1.0 / f32(params.counts.y);
    let offset = face * 6u;
    results[offset] = bitcast<vec4<u32>>(total.mean * inverse);
    results[offset + 1u] = bitcast<vec4<u32>>(vec4(total.radiation.x * inverse, 0.0, 0.0, 0.0));
    results[offset + 2u] = bitcast<vec4<u32>>(total.second * inverse);
    results[offset + 3u] = bitcast<vec4<u32>>(vec4(total.radiation.y * inverse, 0.0, 0.0, 0.0));
    results[offset + 4u] = bitcast<vec4<u32>>(direct.value);
    results[offset + 5u] = bitcast<vec4<u32>>(vec4(direct.radiation, direct.custom_visible));
}

var<workgroup> pair_own: array<vec4<f32>, 64>;
var<workgroup> pair_donated: array<vec4<f32>, 64>;
var<workgroup> pair_radiation: array<vec2<f32>, 64>;

// A compact root list assigns adjacent lane groups to reciprocal receivers.
// Each path and reconnection is evaluated once. Sharing stays in workgroup
// memory; no global path records, second dispatch, or CPU face permutation.
@compute @workgroup_size(64)
fn trace_paired_faces(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32) {
    let pair = id.x / (2u * PATH_LANES);
    let lane = local % PATH_LANES;
    let side = (local / PATH_LANES) % 2u;
    let pair_valid = pair < u32(params.transport.y);
    var face = 0u;
    var partner = 0u;
    var paired = false;
    if pair_valid {
        let root = tasks[pair].padding1;
        paired = tasks[root].padding0 > 0u;
        face = root;
        partner = tasks[root].padding0 - 1u;
        if side == 1u && paired { face = partner; partner = root; }
    }
    let valid = pair_valid && (side == 0u || paired);
    var moment: PathMoments;
    // Every lane reaches both barriers even for a short/nonmultiple tail.
    for (var base = 0u; base < params.counts.y; base += PATH_LANES) {
        let i = base + lane;
        var own = Energy(vec4(0.0), 0.0, vec3(0.0));
        var donated = Energy(vec4(0.0), 0.0, vec3(0.0));
        var nee = Energy(vec4(0.0), 0.0, vec3(0.0));
        if valid && i < params.counts.y {
            let task = tasks[face];
            let normal = face_normal(task.face);
            let center = vec3<f32>(task.voxel) + vec3(0.5);
            let origin = center + normal * 0.501;
            let stream = hash_voxel(task.voxel) ^ pcg(task.face + 1u);
            let sequence = task.sequence + i;
            let direction = hemisphere(normal, sample_1d(stream, sequence, 0u), sample_1d(stream, sequence, 1u));
            let record = trace_path_record(origin, direction, normal, stream, sequence, true);
            own = evaluate_record(record, origin, normal);
            if paired {
                let other_origin = vec3<f32>(tasks[partner].voxel) + vec3(0.5) + normal * 0.501;
                let q = record_pdf(record, origin, normal);
                if q > 0.0 {
                    let other_q = compatible_pdf(record, other_origin, normal, q);
                    own.value *= q / (q + other_q);
                    own.radiation *= q / (q + other_q);
                    if other_q > 0.0 {
                        donated = evaluate_record(record, other_origin, normal);
                        donated.value *= other_q / (q + other_q);
                        donated.radiation *= other_q / (q + other_q);
                    }
                }
            }
            nee = sample_emitter(center + normal * 0.5, normal, stream, sequence, 0u);
        }
        pair_own[local] = own.value;
        pair_donated[local] = donated.value;
        pair_radiation[local] = vec2(own.radiation, donated.radiation);
        workgroupBarrier();
        let other = local ^ PATH_LANES;
        let value = pair_own[local] + pair_donated[other] + nee.value;
        let radiation = pair_radiation[local].x + pair_radiation[other].y + nee.radiation;
        moment.mean += value;
        moment.second += value * value;
        moment.radiation += vec2(radiation, radiation * radiation);
        workgroupBarrier();
    }
    path_moments[local] = moment;
    workgroupBarrier();
    if lane != 0u || !valid { return; }
    var total: PathMoments;
    for (var i = 0u; i < PATH_LANES; i += 1u) {
        total.mean += path_moments[local + i].mean;
        total.second += path_moments[local + i].second;
        total.radiation += path_moments[local + i].radiation;
    }
    let task = tasks[face];
    let normal = face_normal(task.face);
    let direct = direct_at(vec3<f32>(task.voxel) + vec3(0.5) + normal * 0.5, normal);
    let inverse = 1.0 / f32(params.counts.y);
    let offset = face * 6u;
    results[offset] = bitcast<vec4<u32>>(total.mean * inverse);
    results[offset + 1u] = bitcast<vec4<u32>>(vec4(total.radiation.x * inverse, 0.0, 0.0, 0.0));
    results[offset + 2u] = bitcast<vec4<u32>>(total.second * inverse);
    results[offset + 3u] = bitcast<vec4<u32>>(vec4(total.radiation.y * inverse, 0.0, 0.0, 0.0));
    results[offset + 4u] = bitcast<vec4<u32>>(direct.value);
    results[offset + 5u] = bitcast<vec4<u32>>(vec4(direct.radiation, direct.custom_visible));
}

// Cooperative layout: four or eight lanes share a receiver, each tracing
// an independent subsequence. Identical paths and moments, not sample reuse.
// The serial entry point remains the paired throughput/correctness reference.
struct PathMoments {
    mean: vec4<f32>,
    second: vec4<f32>,
    radiation: vec2<f32>,
}
var<workgroup> path_moments: array<PathMoments, 64>;

@compute @workgroup_size(64)
fn trace_faces_parallel(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32) {
    let face_index = id.x / PATH_LANES;
    let lane = local % PATH_LANES;
    var moment: PathMoments;
    if face_index < params.counts.x {
        let task = tasks[face_index];
        let normal = face_normal(task.face);
        let center = vec3<f32>(task.voxel) + vec3(0.5);
        let origin = center + normal * 0.501;
        let stream = hash_voxel(task.voxel) ^ pcg(task.face + 1u);
        for (var i = lane; i < params.counts.y; i += PATH_LANES) {
            let sequence = task.sequence + i;
            let direction = hemisphere(normal, sample_1d(stream, sequence, 0u), sample_1d(stream, sequence, 1u));
            let sample = trace_path(origin, direction, normal, stream, sequence);
            let nee = sample_emitter(center + normal * 0.5, normal, stream, sequence, 0u);
            let value = sample.value + nee.value;
            let radiation = sample.radiation + nee.radiation;
            moment.mean += value;
            moment.second += value * value;
            moment.radiation += vec2(radiation, radiation * radiation);
        }
    }
    path_moments[local] = moment;
    // All lanes, including the unused tail, participate in the barrier.
    workgroupBarrier();
    if lane != 0u || face_index >= params.counts.x { return; }
    var total: PathMoments;
    for (var i = 0u; i < PATH_LANES; i += 1u) {
        total.mean += path_moments[local + i].mean;
        total.second += path_moments[local + i].second;
        total.radiation += path_moments[local + i].radiation;
    }
    let task = tasks[face_index];
    let normal = face_normal(task.face);
    let direct = direct_at(vec3<f32>(task.voxel) + vec3(0.5) + normal * 0.5, normal);
    let inverse = 1.0 / f32(params.counts.y);
    let offset = face_index * 6u;
    results[offset] = bitcast<vec4<u32>>(total.mean * inverse);
    results[offset + 1u] = bitcast<vec4<u32>>(vec4(total.radiation.x * inverse, 0.0, 0.0, 0.0));
    results[offset + 2u] = bitcast<vec4<u32>>(total.second * inverse);
    results[offset + 3u] = bitcast<vec4<u32>>(vec4(total.radiation.y * inverse, 0.0, 0.0, 0.0));
    results[offset + 4u] = bitcast<vec4<u32>>(direct.value);
    results[offset + 5u] = bitcast<vec4<u32>>(vec4(direct.radiation, direct.custom_visible));
}
