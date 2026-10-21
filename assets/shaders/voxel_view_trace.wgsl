// View-space receiver consumer. The view owns the receiver list and output
// lifetime; transport is exactly the world-space estimator. This unfiltered
// path stream is qualified independently of reprojection and reconstruction.
// #include voxel_path.wgsl

struct ViewReceiver {
    surface: vec3<f32>,
    face: u32,
    voxel: vec3<i32>,
    sequence: u32,
}

// One record per attempted path, including known black paths. Unknown paths
// keep known=0 rather than contributing black or silently recycling a bright
// predecessor. The owner must also track view/frame and scene/source epochs.
//
// These are path measurements, not a resampling reservoir: weights, represented
// candidate count, reprojection validity and reconnection visibility still have
// to be supplied by the reuse stage. In particular, emitter NEE cannot be
// shifted using the continuation's cosine PDF or its connection Jacobian.
struct ViewPath {
    surface: vec3<f32>,
    sequence: u32,
    voxel: vec3<i32>,
    face: u32,
    direction: vec3<f32>,
    proposal_pdf: f32,
    hit_position: vec3<f32>,
    hit_status: u32,
    hit_voxel: vec3<i32>,
    hit_face: u32,
    emitter: vec3<f32>,
    known: u32,
    continuation: vec3<f32>,
    hit_material: u32,
    direct: vec3<f32>,
    stream: u32,
}
@group(0) @binding(6) var<storage, read> receivers: array<ViewReceiver>;
@group(0) @binding(7) var<storage, read_write> paths: array<ViewPath>;

@compute @workgroup_size(64)
fn trace_view_paths(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.counts.x * params.counts.y { return; }
    let receiver = receivers[id.x / params.counts.y];
    var record: ViewPath;
    // Explicitly clear inactive view lanes; never leave a previous frame's
    // measurement behind. A face >=6 denotes no visible voxel receiver.
    if receiver.face >= 6u { paths[id.x] = record; return; }
    let normal = face_normal(receiver.face);
    let stream = hash_voxel(receiver.voxel) ^ pcg(receiver.face + 1u);
    let sequence = receiver.sequence + id.x % params.counts.y;
    let sample = sample_receiver_path(receiver.surface, normal, stream, sequence);
    let hit = sample.continuation.first_hit;
    record.surface = receiver.surface;
    record.sequence = sequence;
    record.voxel = receiver.voxel;
    record.face = receiver.face;
    record.direction = sample.direction;
    record.proposal_pdf = cosine_pdf(normal, sample.direction);
    record.hit_position = hit.position;
    record.hit_status = hit.status;
    record.hit_voxel = hit.voxel;
    record.hit_face = face_index_of(hit.normal);
    record.emitter = sample.emitter.value.rgb;
    record.known = select(1u, 0u, sample.continuation.unknown);
    record.continuation = sample.continuation.energy.value.rgb;
    record.hit_material = hit.material;
    record.direct = direct_at(receiver.surface, normal).value.rgb;
    record.stream = stream;
    paths[id.x] = record;
}

// Scratch only for in-flight tails, not another history allocation per sample.
// The view owner may compact active entries and retain completed ViewPaths.
// Five-band arithmetic remains identical to the shared kernel, while completed
// view records retain RGB only. No partial result is a completed measurement.
struct ViewContinuation {
    radiance: vec4<f32>,
    extra: vec4<f32>, // radiance ionizing, throughput ionizing, previous PDF, reserved
    throughput: vec4<f32>,
    origin: vec3<f32>,
    bounce: u32,
    direction: vec3<f32>,
    status: u32, // 0 new, 1 paused, 2 complete known, 3 complete unknown, 4 inactive
    previous_position: vec3<f32>,
    reserved0: u32,
    previous_normal: vec3<f32>,
    reserved1: u32,
    ticket: vec4<u32>,
}
struct ViewWave {
    range: vec4<u32>, // first path, count, max hit steps (0..8), finite depth (qualification only; 0 unlimited)
    ticket: vec4<u32>, // owner-controlled view/geometry/source/receiver generations
}
@group(1) @binding(0) var<storage, read_write> continuations: array<ViewContinuation>;
@group(1) @binding(1) var<uniform> wave: ViewWave;

@compute @workgroup_size(64)
fn advance_view_paths(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= wave.range.y || wave.range.z == 0u { return; }
    let index = wave.range.x + id.x;
    if index >= params.counts.x * params.counts.y { return; }
    let receiver = receivers[index / params.counts.y];
    let sequence = receiver.sequence + index % params.counts.y;
    var record = paths[index];
    var stored = continuations[index];
    if receiver.face >= 6u {
        var empty: ViewPath;
        paths[index] = empty;
        stored.status = 4u;
        stored.ticket = wave.ticket;
        continuations[index] = stored;
        return;
    }
    let normal = face_normal(receiver.face);
    let stream = hash_voxel(receiver.voxel) ^ pcg(receiver.face + 1u);
    let restart = stored.status == 0u || stored.status == 4u || any(stored.ticket != wave.ticket)
        || any(record.surface != receiver.surface) || any(record.voxel != receiver.voxel)
        || record.face != receiver.face || record.sequence != sequence;
    var state: PathState;
    if restart {
        var empty: ViewPath;
        record = empty;
        record.surface = receiver.surface;
        record.sequence = sequence;
        record.voxel = receiver.voxel;
        record.face = receiver.face;
        record.direction = hemisphere(normal, sample_1d(stream, sequence, 0u), sample_1d(stream, sequence, 1u));
        record.proposal_pdf = cosine_pdf(normal, record.direction);
        record.emitter = sample_emitter(receiver.surface, normal, stream, sequence, 0u).value.rgb;
        record.direct = direct_at(receiver.surface, normal).value.rgb;
        record.stream = stream;
        state = begin_path(receiver.surface + normal * 0.001, record.direction, normal);
    } else {
        if stored.status != 1u { return; }
        state.result.energy.value = stored.radiance;
        state.result.energy.radiation = stored.extra.x;
        state.result.first_hit = RayResult(record.hit_voxel, record.hit_material,
            record.hit_position, record.hit_status, vec3(0.0));
        if record.hit_face < 6u { state.result.first_hit.normal = face_normal(record.hit_face); }
        state.throughput = Energy(stored.throughput, stored.extra.y, vec3(0.0));
        state.origin = stored.origin;
        state.direction = stored.direction;
        state.previous_position = stored.previous_position;
        state.previous_normal = stored.previous_normal;
        state.previous_pdf = stored.extra.z;
        state.bounce = stored.bounce;
    }
    var steps = min(wave.range.z, 8u);
    var terminal = false;
    if wave.range.w != 0u {
        steps = min(steps, wave.range.w - min(state.bounce, wave.range.w));
        terminal = state.bounce + steps == wave.range.w;
    }
    state = advance_path(state, stream, sequence, steps, terminal);
    let hit = state.result.first_hit;
    record.hit_position = hit.position;
    record.hit_status = hit.status;
    record.hit_voxel = hit.voxel;
    record.hit_face = face_index_of(hit.normal);
    record.hit_material = hit.material;
    record.known = select(0u, 1u, state.done && !state.result.unknown);
    record.continuation = select(vec3(0.0), state.result.energy.value.rgb, state.done);
    stored.radiance = state.result.energy.value;
    stored.extra = vec4(state.result.energy.radiation, state.throughput.radiation, state.previous_pdf, 0.0);
    stored.throughput = state.throughput.value;
    stored.origin = state.origin;
    stored.bounce = state.bounce;
    stored.direction = state.direction;
    stored.status = select(1u, select(2u, 3u, state.result.unknown), state.done);
    stored.previous_position = state.previous_position;
    stored.previous_normal = state.previous_normal;
    stored.ticket = wave.ticket;
    paths[index] = record;
    continuations[index] = stored;
}
