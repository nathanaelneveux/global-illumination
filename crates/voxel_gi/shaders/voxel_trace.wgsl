// World-space face consumers of the common transport kernel, and change footprints.
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
    second_extra: vec4<f32>, // reserved
    direct: vec4<f32>,
    direct_extra: vec4<f32>,
}
@group(0) @binding(6) var<storage, read> tasks: array<FaceTask>;
// Six float4s per face.
@group(0) @binding(7) var<storage, read_write> results: array<vec4<u32>>;

fn write_result(face_index: u32, mean: vec4<f32>, second: vec4<f32>, known: u32) {
    let task = tasks[face_index];
    let normal = face_normal(task.face);
    let direct = direct_at(vec3<f32>(task.voxel) + vec3(0.5) + normal * 0.5, normal);
    let inverse = 1.0 / f32(max(known, 1u));
    let offset = face_index * 6u;
    results[offset] = bitcast<vec4<u32>>(mean * inverse);
    results[offset + 1u] = bitcast<vec4<u32>>(vec4(0.0, f32(known), 0.0, 0.0));
    results[offset + 2u] = bitcast<vec4<u32>>(second * inverse);
    results[offset + 3u] = vec4(0u);
    results[offset + 4u] = bitcast<vec4<u32>>(direct.value);
    results[offset + 5u] = bitcast<vec4<u32>>(vec4(0.0, direct.custom_visible));
}

struct PathMoments {
    mean: vec4<f32>,
    second: vec4<f32>,
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
        moment.mean += value;
        moment.second += value * value;
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
    write_result(id.x, moment.mean, moment.second, moment.known);
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
        total.known += path_moments[local + i].known;
    }
    write_result(face_index, total.mean, total.second, total.known);
}

// Sunlight a change admits or blocks, traced in the new scene: tubes through
// a changed box along each light whose sun side is clear land on surfaces
// whose sunlight depends only on the box's content, which changed. However
// far below, that landing patch becomes a Light change there: a skylight 40
// voxels above a floor relights the floor and what it lights, which the
// edit's own distance rule would leave alone. One task per changed box, one
// thread per tube. Tasks: (box min, header record) (box max, bound record)
// (first thread, grid, 0, 0); a count first.
@group(0) @binding(20) var<storage, read> footprints: array<vec4<i32>>;
// Per task: landing min xyz, max xyz (biased by 2^30), reflected flux x 64.
@group(0) @binding(21) var<storage, read_write> footprint_sums: array<atomic<u32>>;
@group(0) @binding(22) var<storage, read_write> events: array<vec4<u32>>;

const FOOTPRINT_BIAS: i32 = 1073741824;
const FOOTPRINT_FLUX_SCALE: f32 = 64.0;

// The last task whose first thread is at most t (first threads ascend).
fn footprint_task(t: u32) -> u32 {
    var lo = 0u;
    var hi = u32(footprints[0].x);
    while hi - lo > 1u {
        let mid = (lo + hi) / 2u;
        if u32(footprints[1u + mid * 3u + 2u].x) <= t { lo = mid; } else { hi = mid; }
    }
    return lo;
}

@compute @workgroup_size(64)
fn trace_footprints(@builtin(global_invocation_id) id: vec3<u32>) {
    if u32(footprints[0].x) == 0u { return; }
    let e = footprint_task(id.x);
    let a = footprints[1u + e * 3u];
    let b = footprints[1u + e * 3u + 1u];
    let c = footprints[1u + e * 3u + 2u];
    let grid = u32(c.y);
    let local = id.x - u32(c.x);
    if local >= grid * grid { return; }
    let lo = vec3<f32>(a.xyz);
    let hi = vec3<f32>(b.xyz) + vec3(1.0);
    let center = (lo + hi) * 0.5;
    // A square grid across the box's diagonal covers its section from any
    // direction; tubes that miss the box are skipped below.
    let diagonal = length(hi - lo);
    let spacing = diagonal / f32(grid);
    let offset = (vec2(f32(local % grid), f32(local / grid)) + 0.5) * spacing - 0.5 * diagonal;
    let w = vec3(0.2126, 0.7152, 0.0722);
    let reach = params.extra.y;
    let base = e * 8u;
    var flux = 0.0;
    for (var i = 0u; i < u32(params.maximum.w); i += 1u) {
        let light = lights[i];
        let color = max(light.energy.rgb - light.extra.yzw, vec3(0.0));
        if all(color == vec3(0.0)) { continue; }
        let direction = normalize(light.position_range.xyz);
        let u = normalize(cross(direction, select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(direction.y) > 0.9)));
        let v = cross(direction, u);
        let point = center + u * offset.x + v * offset.y;
        // Exact axis directions would make 0 * inf; a tiny component is exact enough.
        let inverse = 1.0 / select(direction, vec3(1e-8), abs(direction) < vec3(1e-8));
        let near = (lo - point) * inverse;
        let far = (hi - point) * inverse;
        let enter = max(max(min(near.x, far.x), min(near.y, far.y)), min(near.z, far.z));
        let exit = min(min(max(near.x, far.x), max(near.y, far.y)), max(near.z, far.z));
        if enter > exit { continue; }
        // Shadowed beyond the box either way: the change cannot matter.
        if trace_domain(point + direction * (exit + 0.01), direction, reach, true).status != 0u { continue; }
        let hit = trace_domain(point + direction * (enter - 0.01), -direction, reach, true);
        if hit.status != 1u || dot(hit.normal, direction) <= 0.0 { continue; }
        // Reflected power through the tube: irradiance across its section
        // (perpendicular to the light, so no cosine) times the albedo.
        flux += dot(color, w) * dot(materials[hit.material].diffuse.rgb, w) * spacing * spacing;
        // Per component: naga's Metal output mis-sizes packed int3 arithmetic.
        let p = vec3(u32(hit.voxel.x + FOOTPRINT_BIAS), u32(hit.voxel.y + FOOTPRINT_BIAS),
            u32(hit.voxel.z + FOOTPRINT_BIAS));
        atomicMin(&footprint_sums[base], p.x);
        atomicMin(&footprint_sums[base + 1u], p.y);
        atomicMin(&footprint_sums[base + 2u], p.z);
        atomicMax(&footprint_sums[base + 3u], p.x);
        atomicMax(&footprint_sums[base + 4u], p.y);
        atomicMax(&footprint_sums[base + 5u], p.z);
    }
    if flux > 0.0 { atomicAdd(&footprint_sums[base + 6u], u32(flux * FOOTPRINT_FLUX_SCALE + 0.5)); }
}

// Each task's landing patch becomes its reserved Light event: the header's
// union and its one bound. A task that landed nowhere keeps its empty bound,
// out of every receiver's reach.
@compute @workgroup_size(64)
fn finish_footprints(@builtin(global_invocation_id) id: vec3<u32>) {
    let e = id.x;
    if e >= u32(footprints[0].x) { return; }
    let base = e * 8u;
    let x = atomicLoad(&footprint_sums[base]);
    if x == 0xffffffffu { return; }
    let minimum = vec3<i32>(vec3(x, atomicLoad(&footprint_sums[base + 1u]),
        atomicLoad(&footprint_sums[base + 2u]))) - vec3(FOOTPRINT_BIAS);
    let maximum = vec3<i32>(vec3(atomicLoad(&footprint_sums[base + 3u]),
        atomicLoad(&footprint_sums[base + 4u]), atomicLoad(&footprint_sums[base + 5u]))) - vec3(FOOTPRINT_BIAS);
    let flux = f32(atomicLoad(&footprint_sums[base + 6u])) / FOOTPRINT_FLUX_SCALE;
    let header = u32(footprints[1u + e * 3u].w);
    let bound = u32(footprints[1u + e * 3u + 1u].w);
    events[header] = vec4(bitcast<vec3<u32>>(minimum), events[header].w);
    events[header + 1u] = vec4(bitcast<vec3<u32>>(maximum), events[header + 1u].w);
    events[bound] = vec4(bitcast<vec3<u32>>(minimum), bitcast<u32>(flux));
    events[bound + 1u] = vec4(bitcast<vec3<u32>>(maximum), 4u);
}
