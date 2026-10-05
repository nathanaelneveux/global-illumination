// World-space face transport. No camera, framebuffer or screen-space history.
// Layouts and sampling dimensions are mirrored by gpu.rs and sampling.rs.
struct Params {
    counts: vec4<u32>, // tasks, samples per face, directory mask, light count
    minimum: vec4<i32>, // xyz: scene bounds, w: shared corner task count
    maximum: vec4<i32>, // xyz: scene bounds, w: directional light count
    environment: vec4<f32>,
    extra: vec4<f32>, // environment ionizing, path distance
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
    second_extra: vec4<f32>, // ionizing second moment, ordinary direct sky RGB
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
    sky_visible: vec3<f32>,
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
@group(0) @binding(8) var<storage, read> emitters: array<EmissiveFace>;

fn pcg(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn hash_voxel(voxel: vec3<i32>) -> u32 {
    return pcg(pcg(pcg(bitcast<u32>(voxel.x)) ^ bitcast<u32>(voxel.y)) ^ bitcast<u32>(voxel.z));
}
fn sample_1d(stream: u32, sequence: u32, dimension: u32) -> f32 {
    let primes = array<u32, 16>(2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u, 41u, 43u, 47u, 53u);
    let base = primes[dimension % 16u];
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
    var distance = 1e30;
    var normal = vec3(0.0);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if abs(direction[axis]) <= 1e-8 { continue; }
        let plane = select(minimum[axis], maximum[axis], direction[axis] > 0.0);
        let travel = (plane - position[axis]) / direction[axis];
        if travel >= -0.001 && travel < distance {
            distance = max(travel, 0.0);
            normal = vec3(0.0);
            normal[axis] = select(1.0, -1.0, direction[axis] > 0.0);
        }
    }
    return vec4(normal, max(distance + 0.001, 0.001));
}
fn loaded_region_ahead(start: vec3<f32>, direction: vec3<f32>) -> bool {
    var position = start;
    for (var step = 0u; step < 2048u; step += 1u) {
        let region = vec3<i32>(floor(position / 16.0));
        if !in_bounds(region) { return false; }
        if lookup_region(region) != 0xffffffffu { return true; }
        position += direction * advance_box(position, direction, region, vec3(0), vec3(0), 16.0).w;
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
                result.status = select(0u, 2u, loaded_region_ahead(position, direction));
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
            if light.position_range.w < 0.0 {
                energy.sky_visible += max(light.energy.rgb - light.extra.yzw, vec3(0.0)) * cosine;
            }
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
fn sample_emitter(position: vec3<f32>, normal: vec3<f32>, source_stream: u32, sequence: u32, bounce: u32, has_cosine_path: bool) -> Energy {
    var result: Energy;
    let count = u32(params.extra.w);
    if count == 0u { return result; }
    let stream = source_stream ^ pcg(bounce + 0xa341316cu);
    let choice = sample_1d(stream, sequence, 12u);
    var low = 0u;
    var high = count;
    while low < high {
        let middle = (low + high) / 2u;
        if emitters[middle].sampling.x <= choice { low = middle + 1u; }
        else { high = middle; }
    }
    let emitter = emitters[min(low, count - 1u)];
    let emitter_normal = face_normal(emitter.face);
    let axis = emitter.face / 2u;
    let tangent = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), axis == 0u);
    let bitangent = select(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), axis == 2u);
    let light_position = vec3<f32>(emitter.voxel) + vec3(0.5) + emitter_normal * 0.5
        + tangent * (sample_1d(stream, sequence, 13u) - 0.5) + bitangent * (sample_1d(stream, sequence, 14u) - 0.5);
    let delta = light_position - position;
    let d2 = dot(delta, delta);
    if d2 < 0.0001 || d2 > params.extra.y * params.extra.y { return result; }
    let distance = sqrt(d2);
    let direction = delta / distance;
    let receiver_cosine = max(dot(normal, direction), 0.0);
    let emitter_cosine = max(dot(emitter_normal, -direction), 0.0);
    if receiver_cosine <= 0.0 || emitter_cosine <= 0.0 || emitter.sampling.y <= 0.0 { return result; }
    let light_pdf = emitter.sampling.y * d2 / emitter_cosine;
    let cosine_pdf = receiver_cosine / 3.14159265359;
    let weight = select(1.0, light_pdf / (light_pdf + cosine_pdf), has_cosine_path);
    if trace(position + normal * 0.002, direction, distance - 0.008).status != 0u { return result; }
    let material = materials[u32(emitter.sampling.z)];
    let factor = cosine_pdf * weight / light_pdf;
    result.value = material.emission * factor;
    result.radiation = material.emission_extra.x * factor;
    return result;
}
fn trace_path(start: vec3<f32>, first_direction: vec3<f32>, initial_normal: vec3<f32>, stream: u32, sequence: u32) -> Energy {
    var result: Energy;
    var throughput = Energy(vec4(1.0), 1.0, vec3(0.0), vec3(0.0));
    var origin = start;
    var direction = first_direction;
    var previous_normal = initial_normal;
    for (var bounce = 0u; bounce < 3u; bounce += 1u) {
        let hit = trace(origin, direction, params.extra.y);
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
        let light_pdf = emission_weight / max(params.extra.z, 1e-20) * dot(delta, delta) / max(dot(hit.normal, -direction), 1e-8);
        var mis = 1.0;
        if light_pdf > 0.0 { mis = cosine_pdf / (cosine_pdf + light_pdf); }
        let nee = sample_emitter(hit.position, hit.normal, stream, sequence, bounce + 1u, bounce + 1u < 3u);
        result.value += throughput.value * (material.emission * mis + material.diffuse * (direct.value + nee.value));
        result.radiation += throughput.radiation * (material.emission_extra.x * mis + material.diffuse_extra.x * (direct.radiation + nee.radiation));
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
        previous_normal = hit.normal;
        direction = hemisphere(hit.normal, sample_1d(stream, sequence, 2u + bounce * 2u), sample_1d(stream, sequence, 3u + bounce * 2u));
    }
    return result;
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
        let nee = sample_emitter(center + normal * 0.5, normal, stream, sequence, 0u, true);
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
    // Use spare readback lanes; direct sky is deterministic, not accumulated
    // with the path moments and not multiplied by the samples-per-face count.
    result.second_extra.y = direct.sky_visible.x;
    result.second_extra.z = direct.sky_visible.y;
    result.second_extra.w = direct.sky_visible.z;
    let offset = id.x * 6u;
    results[offset] = bitcast<vec4<u32>>(result.mean);
    results[offset + 1u] = bitcast<vec4<u32>>(result.mean_extra);
    results[offset + 2u] = bitcast<vec4<u32>>(result.second);
    results[offset + 3u] = bitcast<vec4<u32>>(result.second_extra);
    results[offset + 4u] = bitcast<vec4<u32>>(result.direct);
    results[offset + 5u] = bitcast<vec4<u32>>(result.direct_extra);
}
