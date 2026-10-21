// View-owned, face-identified path history. Reprojection copies an identical
// receiver and its unfinished ray, not a color at the previous screen pixel.
// Duplicate screen samples of a voxel face share one owner and one sequence.
// #include voxel_view_trace.wgsl

struct ViewControl {
    world_from_clip: mat4x4<f32>,
    camera: vec4<f32>, // xyz camera, w inner overlap distance
    viewport: vec4<u32>, // xy origin, zw full-resolution dimensions
    grid: vec4<u32>, // xy reduced dimensions, z count, w hash mask
    history: vec4<u32>, // previous grid count, previous hash mask, frame, event count
    display: vec4<f32>, // evidence goal, strength, temporal blend, measured retention
}
struct ViewEstimate {
    mean: vec4<f32>, // raw RGB, physical history weight
    moment: vec4<f32>, // raw second moment, pending predicted change
    shown: vec4<f32>, // reconstructed presentation, valid support
    identity: vec4<u32>, // next path sequence, material, reserved, reserved
}
struct FarPixel {
    surface: vec3<f32>,
    face: u32,
    irradiance: vec3<f32>,
    evidence: f32,
    sky: vec4<u32>,
}
struct FarOutput {
    viewport: vec4<u32>,
    shape: vec4<u32>,
    optical: array<vec4<f32>, 1024>,
    pixels: array<FarPixel>,
}
@group(2) @binding(0) var<uniform> view: ViewControl;
@group(2) @binding(1) var<storage, read> previous_receivers: array<ViewReceiver>;
@group(2) @binding(2) var<storage, read> previous_paths: array<ViewPath>;
@group(2) @binding(3) var<storage, read> previous_continuations: array<ViewContinuation>;
@group(2) @binding(4) var<storage, read> previous_estimates: array<ViewEstimate>;
@group(2) @binding(5) var<storage, read_write> estimates: array<ViewEstimate>;
@group(2) @binding(6) var<storage, read> previous_hash: array<u32>;
@group(2) @binding(7) var<storage, read_write> current_hash: array<atomic<u32>>;
@group(2) @binding(8) var depth: texture_depth_2d;
@group(2) @binding(9) var normals: texture_2d<f32>;
@group(2) @binding(10) var<storage, read_write> queue: array<atomic<u32>>;
@group(2) @binding(11) var<storage, read_write> far_output: FarOutput;
@group(2) @binding(12) var<storage, read> events: array<vec4<u32>>;
@group(2) @binding(13) var<uniform> event_control: vec4<u32>;

// #include voxel_retention.wgsl

fn receiver_hash(r: ViewReceiver) -> u32 {
    return hash_voxel(r.voxel) ^ pcg(r.face + 1u);
}
fn same_receiver(a: ViewReceiver, b: ViewReceiver) -> bool {
    return a.face == b.face && all(a.voxel == b.voxel);
}
fn owner_of(r: ViewReceiver) -> u32 {
    var slot = receiver_hash(r) & view.grid.w;
    for (var step = 0u; step < 64u; step++) {
        let value = atomicLoad(&current_hash[slot]);
        if value == 0u { return 0xffffffffu; }
        if same_receiver(receivers[value - 1u], r) { return value - 1u; }
        slot = (slot + 1u) & view.grid.w;
    }
    return 0xffffffffu;
}
fn previous_owner(r: ViewReceiver) -> u32 {
    if view.history.x == 0u { return 0xffffffffu; }
    var slot = receiver_hash(r) & view.history.y;
    for (var step = 0u; step < 64u; step++) {
        let value = previous_hash[slot];
        if value == 0u { return 0xffffffffu; }
        if value <= view.history.x && same_receiver(previous_receivers[value - 1u], r) { return value - 1u; }
        slot = (slot + 1u) & view.history.y;
    }
    return 0xffffffffu;
}

@compute @workgroup_size(64)
fn discover_receivers(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    let xy = vec2(id.x % view.grid.x, id.x / view.grid.x);
    let center_uv = (vec2<f32>(xy) + 0.5) / vec2<f32>(view.grid.xy);
    let pixel = vec2<i32>(view.viewport.xy + min(vec2<u32>(center_uv * vec2<f32>(view.viewport.zw)), view.viewport.zw - 1u));
    let uv = (vec2<f32>(pixel) + 0.5 - vec2<f32>(view.viewport.xy)) / vec2<f32>(view.viewport.zw);
    var receiver: ViewReceiver;
    receiver.face = 0xffffffffu;
    receivers[id.x] = receiver;
    var empty: ViewEstimate;
    estimates[id.x] = empty;
    let z = textureLoad(depth, pixel, 0);
    if z <= 0.0 { return; }
    let clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    let homogeneous = view.world_from_clip * clip;
    let position = homogeneous.xyz / homogeneous.w;
    if distance(position, view.camera.xyz) < view.camera.w { return; }
    let decoded = textureLoad(normals, pixel, 0).xyz * 2.0 - 1.0;
    if max(max(abs(decoded.x), abs(decoded.y)), abs(decoded.z)) < 0.99 { return; }
    // RGB10A2 cannot represent the two zero components exactly. Traversal's
    // face_index_of expects an exact axis normal; quantized raster normals
    // must instead select their dominant component.
    receiver.face = select(4u, 5u, decoded.z < 0.0);
    if abs(decoded.x) > 0.99 { receiver.face = select(0u, 1u, decoded.x < 0.0); }
    else if abs(decoded.y) > 0.99 { receiver.face = select(2u, 3u, decoded.y < 0.0); }
    let normal = face_normal(receiver.face);
    // MSAA depth is evaluated at a subpixel sample, while unprojection above
    // uses the pixel center. Snap the normal axis to the exact voxel plane;
    // otherwise that small drift can select a buried voxel on grazing faces.
    var surface_position = position;
    surface_position[receiver.face / 2u] = round(surface_position[receiver.face / 2u]);
    receiver.voxel = vec3<i32>(floor(surface_position - normal * 0.01));
    receiver.surface = vec3<f32>(receiver.voxel) + 0.5 + normal * 0.5;
    // Reject nonvoxel meshes and mesh/transport publication skew. This is a
    // short geometry identity check, not another primary camera ray.
    let material = voxel_material(receiver.voxel);
    if material == 256u || voxel_material(receiver.voxel + vec3<i32>(normal)) != 256u { return; }
    receivers[id.x] = receiver;
    estimates[id.x].identity.y = material;
}

// Receiver keys are fully written by discover before this pass. CAS publishes
// only an index, never a partially initialized identity. Colliding keys probe;
// duplicate keys all resolve to the same winner regardless of dispatch order.
@compute @workgroup_size(64)
fn index_receivers(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    let receiver = receivers[id.x];
    if receiver.face >= 6u { return; }
    var slot = receiver_hash(receiver) & view.grid.w;
    for (var step = 0u; step < 64u; step++) {
        let claim = atomicCompareExchangeWeak(&current_hash[slot], 0u, id.x + 1u);
        if claim.exchanged { return; }
        if claim.old_value == 0u { continue; } // weak CAS may fail spuriously
        if same_receiver(receivers[claim.old_value - 1u], receiver) { return; }
        slot = (slot + 1u) & view.grid.w;
    }
    atomicAdd(&queue[1], 1u); // explicit overflow diagnostic, never alias keys
}

@compute @workgroup_size(64)
fn reproject_paths(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    var receiver = receivers[id.x];
    if receiver.face >= 6u || owner_of(receiver) != id.x { return; }
    let material = estimates[id.x].identity.y;
    let previous = previous_owner(receiver);
    var estimate: ViewEstimate;
    var record: ViewPath;
    var state: ViewContinuation;
    if previous != 0xffffffffu && previous_estimates[previous].identity.y == material {
        estimate = previous_estimates[previous];
        record = previous_paths[previous];
        state = previous_continuations[previous];
    }
    estimate.identity.y = material;
    let normal = face_normal(receiver.face);
    var lit_face = false;
    for (var light = 0u; light < u32(params.maximum.w); light++) {
        lit_face = lit_face || dot(normal, lights[light].position_range.xyz) > 0.0;
    }
    estimate.identity.z = 0x80808080u;
    if lit_face {
        var u = vec3(0.0, 1.0, 0.0);
        var v = vec3(0.0, 0.0, 1.0);
        if receiver.face / 2u == 1u { u = vec3(1.0, 0.0, 0.0); }
        else if receiver.face / 2u == 2u { u = vec3(1.0, 0.0, 0.0); v = vec3(0.0, 1.0, 0.0); }
        let base = vec3<f32>(receiver.voxel) + max(normal, vec3(0.0));
        var packed = 0u;
        for (var corner = 0u; corner < 4u; corner++) {
            let position = base + u * f32(corner & 1u) + v * f32(corner >> 1u);
            let word = sky_corner_visibility(position);
            packed |= ((word & 15u) | ((word >> 24u) & 128u)) << (corner * 8u);
        }
        estimate.identity.z = packed;
    }
    receiver.sequence = estimate.identity.x;
    receivers[id.x].sequence = receiver.sequence;
    paths[id.x] = record;
    continuations[id.x] = state;
    estimates[id.x] = estimate;
    let entry = atomicAdd(&queue[0], 1u);
    atomicStore(&queue[12u + entry], id.x);
}

@compute @workgroup_size(64)
fn trace_visible_paths(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = atomicLoad(&queue[0]);
    if id.x >= min(count, wave.range.y) || wave.range.z == 0u { return; }
    let index = atomicLoad(&queue[12u + (wave.range.x + id.x) % count]);
    advance_view_path(index);
    let state = continuations[index];
    if state.status != 2u && state.status != 3u { return; }
    atomicAdd(&queue[2], 1u);
    var estimate = estimates[index];
    // Unknown is not black. Every attempted sequence advances, including a
    // rejected path, so geometry gaps cannot lock a face to one direction.
    estimate.identity.x += 1u;
    if paths[index].known != 0u {
        atomicAdd(&queue[3], 1u);
        let value = paths[index].emitter + paths[index].continuation;
        if estimate.moment.w > 0.0 {
            let change = estimate.moment.w;
            let variance = max(estimate.moment.rgb - estimate.mean.rgb * estimate.mean.rgb, vec3(0.0));
            let keep = view.display.w * dot(variance, vec3(0.2126, 0.7152, 0.0722)) / (change * change);
            estimate.mean.w = min(estimate.mean.w, floor(keep));
            estimate.moment.w = 0.0;
        }
        let count = min(estimate.mean.w + 1.0, 65536.0);
        estimate.mean = vec4(mix(estimate.mean.rgb, value, 1.0 / count), count);
        estimate.moment = vec4(mix(estimate.moment.rgb, value * value, 1.0 / count), estimate.moment.w);
    }
    estimates[index] = estimate;
}

@compute @workgroup_size(64)
fn retain_view_history(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    let receiver = receivers[id.x];
    if receiver.face >= 6u || owner_of(receiver) != id.x { return; }
    let change = resolve_retention(receiver.voxel, event_control.x, event_control.y,
        view.display.w > 0.0, 0u, 0u);
    if change.limit != NO_LIMIT { estimates[id.x].mean.w = min(estimates[id.x].mean.w, f32(change.limit)); }
    if change.measured { estimates[id.x].moment.w += change.change; }
    // A pending tail through changed geometry is not a sample of either
    // scene. Discard only that attempt, preserving accumulated measurements,
    // its display prior and all unrelated receivers' in-flight paths.
    if change.woken && continuations[id.x].status == 1u {
        continuations[id.x].status = 0u;
    }
}

// Presentation only: raw measurements and unfinished rays above never read
// this output. Only unique, coplanar owners contribute support; duplicate
// screen pixels do not masquerade as additional independent evidence.
@compute @workgroup_size(64)
fn reconstruct_view(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    let receiver = receivers[id.x];
    if receiver.face >= 6u || owner_of(receiver) != id.x { return; }
    let center = estimates[id.x];
    let normal = face_normal(receiver.face);
    let xy = vec2<i32>(i32(id.x % view.grid.x), i32(id.x / view.grid.x));
    let radius = i32(clamp(ceil(sqrt(view.display.x / max(center.mean.w, 1.0) / 3.14159)), 1.0, 8.0));
    var sum = center.mean.rgb * center.mean.w;
    var weight = center.mean.w;
    if center.mean.w < view.display.x && view.display.y > 0.0 {
        for (var y = -radius; y <= radius; y++) {
            for (var x = -radius; x <= radius; x++) {
                if x == 0 && y == 0 { continue; }
                let q = xy + vec2(x, y);
                if any(q < vec2(0)) || any(q >= vec2<i32>(view.grid.xy)) { continue; }
                let candidate = u32(q.y) * view.grid.x + u32(q.x);
                let donor = receivers[candidate];
                if donor.face != receiver.face { continue; }
                let delta = donor.surface - receiver.surface;
                let d2 = dot(delta, delta);
                if abs(dot(delta, normal)) > 0.05 || d2 > 256.0 || owner_of(donor) != candidate { continue; }
                let value = estimates[candidate].mean;
                let w = min(value.w, 1024.0) / (1.0 + d2 * 0.0625);
                sum += value.rgb * w;
                weight += w;
            }
        }
    }
    if weight <= 0.0 { return; }
    let desired = mix(sum / weight, center.mean.rgb, select(0.0, 1.0 - view.display.y, center.mean.w > 0.0));
    let blend = select(1.0, view.display.z, center.shown.w > 0.0);
    estimates[id.x].shown = vec4(mix(center.shown.rgb, desired, blend), 1.0);
}

@compute @workgroup_size(64)
fn publish_view(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    if id.x == 0u {
        far_output.viewport = view.viewport;
        far_output.shape = vec4(view.grid.xy, 1u, atomicLoad(&queue[1]));
    }
    let receiver = receivers[id.x];
    var output: FarPixel;
    output.face = 0xffffffffu;
    if receiver.face < 6u {
        let owner = owner_of(receiver);
        if owner != 0xffffffffu {
            let value = estimates[owner];
            output.surface = receiver.surface;
            output.face = receiver.face;
            output.irradiance = value.shown.rgb;
            output.evidence = select(0.0, max(value.mean.w, 0.001), value.shown.w > 0.0);
            for (var corner = 0u; corner < 4u; corner++) {
                let byte = (value.identity.z >> (corner * 8u)) & 255u;
                output.sky[corner] = (byte & 15u) | ((byte & 128u) << 24u);
            }
            if owner == id.x {
                atomicAdd(&queue[4], u32(value.mean.w));
                atomicMax(&queue[5], u32(value.mean.w));
                if value.mean.w == 0.0 { atomicAdd(&queue[6], 1u); }
                if output.evidence > 0.0 { atomicAdd(&queue[7], 1u); }
                if continuations[owner].status == 1u { atomicAdd(&queue[8], 1u); }
            }
        }
    }
    far_output.pixels[id.x] = output;
}
