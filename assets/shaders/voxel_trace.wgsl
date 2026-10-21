// World-space face and shared-corner consumers of the common transport kernel.
// Layouts are mirrored by gpu.rs; scheduling and accumulation live outside it.
override PATH_LANES: u32 = 4u;
// #include voxel_path.wgsl
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
@group(0) @binding(6) var<storage, read> tasks: array<FaceTask>;
// Six float4s per face, followed by two packed uint4s per shared sky corner.
@group(0) @binding(7) var<storage, read_write> results: array<vec4<u32>>;

// One word per corner: 0 unknown, else bit 31 plus a lit bit per directional
// light (source order, at most four). Energy and the receiving normal are the
// display's: a recolored or slightly moved sun needs no retrace.
@compute @workgroup_size(64)
fn trace_sky_corners(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u32(params.minimum.w) { return; }
    let position = vec3<f32>(tasks[params.counts.x + id.x].voxel);
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
    let offset = params.counts.x * 6u + id.x * 2u;
    results[offset] = vec4(word, 0u, 0u, 0u);
    results[offset + 1u] = vec4(0u);
}

fn write_result(face_index: u32, mean: vec4<f32>, second: vec4<f32>, radiation: vec2<f32>, known: u32) {
    let task = tasks[face_index];
    let normal = face_normal(task.face);
    let direct = direct_at(vec3<f32>(task.voxel) + vec3(0.5) + normal * 0.5, normal);
    let inverse = 1.0 / f32(max(known, 1u));
    let offset = face_index * 6u;
    results[offset] = bitcast<vec4<u32>>(mean * inverse);
    results[offset + 1u] = bitcast<vec4<u32>>(vec4(radiation.x * inverse, f32(known), 0.0, 0.0));
    results[offset + 2u] = bitcast<vec4<u32>>(second * inverse);
    results[offset + 3u] = bitcast<vec4<u32>>(vec4(radiation.y * inverse, 0.0, 0.0, 0.0));
    results[offset + 4u] = bitcast<vec4<u32>>(direct.value);
    results[offset + 5u] = bitcast<vec4<u32>>(vec4(direct.radiation, direct.custom_visible));
}

struct PathMoments {
    mean: vec4<f32>,
    second: vec4<f32>,
    radiation: vec2<f32>,
    known: u32,
}
fn trace_moments(face_index: u32, first: u32, step: u32) -> PathMoments {
    var moment: PathMoments;
    let task = tasks[face_index];
    let normal = face_normal(task.face);
    let surface = vec3<f32>(task.voxel) + vec3(0.5) + normal * 0.5;
    let stream = hash_voxel(task.voxel) ^ pcg(task.face + 1u);
    for (var i = first; i < params.counts.y; i += step) {
        let sample = trace_receiver(surface, normal, stream, task.sequence + i);
        if !sample.known { continue; }
        let value = sample.energy.value;
        let radiation = sample.energy.radiation;
        moment.mean += value;
        moment.second += value * value;
        moment.radiation += vec2(radiation, radiation * radiation);
        moment.known += 1u;
    }
    return moment;
}

// Serial reference layout: one invocation per face.
@compute @workgroup_size(64)
fn trace_faces(@builtin(global_invocation_id) id: vec3<u32>) {
    // Unused GPU pick capacity keeps a face of 0xffffffff.
    if id.x >= params.counts.x || tasks[id.x].face == 0xffffffffu { return; }
    let moment = trace_moments(id.x, 0u, 1u);
    write_result(id.x, moment.mean, moment.second, moment.radiation, moment.known);
}

// Cooperative layout: PATH_LANES lanes share a receiver, each tracing an
// independent subsequence. Identical paths and moments, not sample reuse.
var<workgroup> path_moments: array<PathMoments, 64>;
@compute @workgroup_size(64)
fn trace_faces_parallel(@builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32) {
    let face_index = id.x / PATH_LANES;
    let valid = face_index < params.counts.x && tasks[face_index].face != 0xffffffffu;
    let lane = local % PATH_LANES;
    var moment: PathMoments;
    if valid { moment = trace_moments(face_index, lane, PATH_LANES); }
    path_moments[local] = moment;
    // All lanes, including the unused tail, participate in the barrier.
    workgroupBarrier();
    if lane != 0u || !valid { return; }
    var total: PathMoments;
    for (var i = 0u; i < PATH_LANES; i += 1u) {
        total.mean += path_moments[local + i].mean;
        total.second += path_moments[local + i].second;
        total.radiation += path_moments[local + i].radiation;
        total.known += path_moments[local + i].known;
    }
    write_result(face_index, total.mean, total.second, total.radiation, total.known);
}
