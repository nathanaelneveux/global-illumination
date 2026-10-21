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
