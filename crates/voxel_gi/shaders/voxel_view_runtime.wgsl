// View-owned, face-identified path history. Reprojection copies an identical
// receiver and its unfinished ray, not a color at the previous screen pixel.
// Duplicate screen samples of a voxel face share one owner and one sequence.
// #include voxel_view_trace.wgsl

struct ViewControl {
    world_from_clip: mat4x4<f32>,
    clip_from_world: mat4x4<f32>,
    camera: vec4<f32>, // xyz camera, w inner overlap distance
    viewport: vec4<u32>, // xy origin, zw full-resolution dimensions
    grid: vec4<u32>, // xy reduced dimensions, z count, w hash mask
    history: vec4<u32>, // previous grid count, previous hash mask, frame, view-first
    display: vec4<f32>, // evidence goal, strength, temporal blend, measured retention
    cache: vec4<u32>, // writable epoch, worker-clock milliseconds, enabled, reserved
    discovery: vec4<u32>, // stationary, compatible previous viewport/overlap/MSAA; reserved
    pixel_axis: vec4<f32>, // far tier (voxel_view_pixels.wgsl): view forward, pixels per unit at depth 1
    pixel_grid: vec4<u32>, // cells x, y, pixels per cell side, enabled (bit 0) and measure (bits 1-2)
    pixel_tune: vec4<f32>, // voxel-size threshold px, max history, trace rate, previous valid
    previous_clip_from_world: mat4x4<f32>,
}
struct ViewEstimate {
    mean: vec4<f32>, // raw RGB, effective history weight (may include a one-unit topology prior)
    moment: vec4<f32>, // raw second moment, pending predicted change
    shown: vec4<f32>, // reconstructed presentation, valid support
    identity: vec4<u32>, // next path sequence, material, sunlight, resident incarnation
}
// Raster geometry can publish before the asynchronous transport scene. Keep a
// face-owned prior during that interval, but never trace or export it against
// incompatible geometry. This bit is not a material-table index.
const PENDING_GEOMETRY: u32 = 0x80000000u;
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
    keys: array<u32, 262144>, // visible face -> owner pixel + 1; zero is empty
    heads: array<u32, 65536>, // projected 2x2-cell tile -> first pool slot + 1
    links: array<u32, 131072>, // spatial list, independent of physical slot order
    pixels: array<FarPixel>,
}
// Words of the `queue` buffer (Rust: runtime::queue_words): counters read
// back with the timing receipt, then from QUEUE_PREFIX the visible-owner
// queue (grid.z), each owner's queue rank (grid.z) and the sun flips.
const QUEUE_OWNERS: u32 = 0u;
const QUEUE_OVERFLOW: u32 = 1u;
const QUEUE_COMPLETED: u32 = 2u;
const QUEUE_KNOWN: u32 = 3u;
const QUEUE_HISTORY: u32 = 4u;
const QUEUE_MAX_HISTORY: u32 = 5u;
const QUEUE_UNCOVERED: u32 = 6u;
const QUEUE_DISPLAYED: u32 = 7u;
const QUEUE_PAUSED: u32 = 8u;
const QUEUE_ATTEMPTED: u32 = 9u;
const QUEUE_CACHE_LOADS: u32 = 10u;
const QUEUE_CACHE_PROMOTIONS: u32 = 11u;
const QUEUE_PREFIX: u32 = 12u;
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
@group(2) @binding(14) var<storage, read_write> shared_state: array<vec4<u32>>;
@group(2) @binding(15) var<storage, read_write> shared_meta: array<vec4<u32>>;
@group(2) @binding(16) var<storage, read> cache_guard: array<u32>;

// #include voxel_retention.wgsl
// #include voxel_view_cache.wgsl

fn receiver_hash(r: ViewReceiver) -> u32 {
    return hash_voxel(r.voxel) ^ pcg(r.face + 1u);
}
fn same_receiver(a: ViewReceiver, b: ViewReceiver) -> bool {
    return a.face == b.face && all(a.voxel == b.voxel) && all(a.surface == b.surface);
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

// #include voxel_view_discovery.wgsl
// #include voxel_view_pixels.wgsl

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
    atomicAdd(&queue[QUEUE_OVERFLOW], 1u); // explicit overflow diagnostic, never alias keys
}

/// Frames between a retained face's corner sun refreshes (shadow edges may
/// trail a moving sun or an edit by up to this many frames minus one).
const SUN_REFRESH_FRAMES: u32 = 4u;

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
    let pending = (material & PENDING_GEOMETRY) != 0u;
    if previous != 0xffffffffu {
        let old = previous_estimates[previous];
        if old.identity.y == material || pending || (old.identity.y & PENDING_GEOMETRY) != 0u {
            estimate = old;
            record = previous_paths[previous];
            state = previous_continuations[previous];
        }
    }
    estimate.identity.y = material;
    let normal = face_normal(receiver.face);
    var lit_face = false;
    for (var light = 0u; light < u32(params.maximum.w); light++) {
        lit_face = lit_face || dot(normal, lights[light].position_range.xyz) > 0.0;
    }
    // A new face awaiting GI geometry has unknown corners (0), not known-unlit
    // ones: the material traces unknown corners itself instead of going black.
    let old_corners = estimate.identity.z;
    if !pending { estimate.identity.z = 0x80808080u; }
    else if previous == 0xffffffffu { estimate.identity.z = 0u; }
    // Corner sun visibility is refreshed every SUN_REFRESH_FRAMES frames per
    // face (staggered by slot); a new, re-identified or pending face traces
    // at once. Four corner rays per face per frame was most of reprojection.
    let carried = previous != 0xffffffffu && !pending && old_corners != 0u
        && (view.history.z + id.x) % SUN_REFRESH_FRAMES != 0u;
    if carried { estimate.identity.z = old_corners; }
    if lit_face && !pending && !carried {
        var u = vec3(0.0, 1.0, 0.0);
        var v = vec3(0.0, 0.0, 1.0);
        if receiver.face / 2u == 1u { u = vec3(1.0, 0.0, 0.0); }
        else if receiver.face / 2u == 2u { u = vec3(1.0, 0.0, 0.0); v = vec3(0.0, 1.0, 0.0); }
        let s = receiver_stride(receiver);
        u *= s;
        v *= s;
        let base = vec3<f32>(receiver.voxel) + max(normal, vec3(0.0)) * s;
        var packed = 0u;
        for (var corner = 0u; corner < 4u; corner++) {
            let position = base + u * f32(corner & 1u) + v * f32(corner >> 1u);
            let word = sky_corner_visibility(position);
            packed |= ((word & 15u) | ((word >> 24u) & 128u)) << (corner * 8u);
        }
        estimate.identity.z = packed;
        if previous != 0xffffffffu && old_corners != 0u { report_sun_flips(receiver, normal, material, old_corners, packed); }
    }
    receiver.sequence = estimate.identity.x;
    receivers[id.x].sequence = receiver.sequence;
    if pending {
        // A paused ray of the old scene is not a path through the new face.
        // Keep accumulated RGB, not its unfinished continuation or lease.
        record = ViewPath();
        state = ViewContinuation();
        estimate.identity.w = 0u;
    }
    paths[id.x] = record;
    continuations[id.x] = state;
    estimates[id.x] = estimate;
    if pending { return; }
    let entry = atomicAdd(&queue[QUEUE_OWNERS], 1u);
    atomicStore(&queue[QUEUE_PREFIX + entry], id.x);
    atomicStore(&queue[QUEUE_PREFIX + view.grid.z + id.x], entry);
}

// A face whose corner sunlight flipped (both measurements known) reports the
// reflected change, a quarter face per corner, for Light retention events:
// a moving sun's shadow borders cut the bounce history around them. The CPU
// bins these (`sun_changes.rs`); edits' shadows are traced as footprints.
// Flips follow the owner queue: a count, (voxel, flux) records, then one
// overflow record that bounds the rest. They persist until a timing receipt
// copies them (only that copy clears them), so a frame without a free
// receipt or past the record limit loses no change. Every slot reads zero
// when empty: the overflow box keeps (2^30 - min) and (max + 2^30).
const MAX_SUN_FLIPS: u32 = 2048u;
const SUN_FLIP_BIAS: i32 = 1073741824;
fn report_sun_flips(receiver: ViewReceiver, normal: vec3<f32>, material: u32, before: u32, after: u32) {
    let w = vec3(0.2126, 0.7152, 0.0722);
    var flux = 0.0;
    var bit = 0u;
    for (var i = 0u; i < u32(params.maximum.w) && bit < 4u; i += 1u) {
        let light = lights[i];
        let color = max(light.energy.rgb - light.extra.yzw, vec3(0.0));
        if all(color == vec3(0.0)) { continue; }
        let cosine = max(dot(normal, normalize(light.position_range.xyz)), 0.0);
        for (var corner = 0u; corner < 4u; corner++) {
            let a = (before >> (corner * 8u)) & 255u;
            let b = (after >> (corner * 8u)) & 255u;
            if (a & b & 128u) != 0u && ((a ^ b) & (1u << bit)) != 0u {
                flux += 0.25 * dot(color, w) * cosine;
            }
        }
        bit += 1u;
    }
    if flux <= 0.0 { return; }
    flux *= dot(materials[material & 255u].diffuse.rgb, w);
    let base = QUEUE_PREFIX + 2u * view.grid.z;
    let slot = atomicAdd(&queue[base], 1u);
    if slot >= MAX_SUN_FLIPS {
        let overflow = base + 4u + MAX_SUN_FLIPS * 4u;
        let v = vec3(receiver.voxel.x, receiver.voxel.y, receiver.voxel.z);
        atomicMax(&queue[overflow], u32(SUN_FLIP_BIAS - v.x));
        atomicMax(&queue[overflow + 1u], u32(SUN_FLIP_BIAS - v.y));
        atomicMax(&queue[overflow + 2u], u32(SUN_FLIP_BIAS - v.z));
        atomicMax(&queue[overflow + 3u], u32(v.x + SUN_FLIP_BIAS));
        atomicMax(&queue[overflow + 4u], u32(v.y + SUN_FLIP_BIAS));
        atomicMax(&queue[overflow + 5u], u32(v.z + SUN_FLIP_BIAS));
        atomicAdd(&queue[overflow + 6u], u32(flux * 64.0 + 0.5));
        return;
    }
    let record = base + 4u + slot * 4u;
    atomicStore(&queue[record], bitcast<u32>(receiver.voxel.x));
    atomicStore(&queue[record + 1u], bitcast<u32>(receiver.voxel.y));
    atomicStore(&queue[record + 2u], bitcast<u32>(receiver.voxel.z));
    atomicStore(&queue[record + 3u], bitcast<u32>(flux));
}

@compute @workgroup_size(64)
fn trace_visible_paths(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = atomicLoad(&queue[QUEUE_OWNERS]);
    if id.x >= min(count, wave.range.y) || wave.range.z == 0u { return; }
    let index = atomicLoad(&queue[QUEUE_PREFIX + (wave.range.x + id.x) % count]);
    // One invocation exclusively owns this receiver, including sequential
    // visits. Extra allowance can advance several three-hit waves without
    // concurrent writes to a face or a permanent one-path-per-frame ceiling.
    // Hybrid keeps its existing one-visit policy. Neither policy truncates a
    // path or publishes its partial energy as a complete measurement.
    let visits = min(select(1u, 32u, view.history.w != 0u), (wave.range.y - id.x + count - 1u) / count);
    let reserved_sequence = view.cache.z != 0u && view.cache.x == cache_guard[0] && estimates[index].identity.w != 0u;
    // Cache admission reserved this dispatch's sequence range before tracing.
    // An older unfinished path keeps its own sequence while background work
    // is free to consume later, disjoint ranges.
    var next = atomicLoad(&queue[QUEUE_PREFIX + view.grid.z + index]);
    if continuations[index].status != 1u { next += 1u; }
    for (var visit = 0u; visit < visits; visit++) {
        atomicAdd(&queue[QUEUE_ATTEMPTED], 1u);
        advance_view_path(index);
        let state = continuations[index];
        if state.status != 2u && state.status != 3u { continue; }
        atomicAdd(&queue[QUEUE_COMPLETED], 1u);
        var estimate = estimates[index];
        // Unknown is not black. Advance every attempted sequence, including
        // rejected paths; the next visit may begin immediately in this pass.
        if reserved_sequence {
            estimate.identity.x = next;
            next += 1u;
        } else { estimate.identity.x += 1u; }
        receivers[index].sequence = estimate.identity.x;
        if paths[index].known != 0u {
            atomicAdd(&queue[QUEUE_KNOWN], 1u);
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
    if reserved_sequence {
        // Unfinished work owns its current index. A completed path's successor
        // is not owned until started. Return only the unstarted suffix, without
        // rewinding past ranges already consumed by the background scheduler.
        let first = atomicLoad(&queue[QUEUE_PREFIX + view.grid.z + index]);
        let used = select(estimates[index].identity.x, paths[index].sequence + 1u,
            continuations[index].status == 1u);
        atomicStore(&queue[QUEUE_PREFIX + view.grid.z + index], max(first, used));
    }
}

@compute @workgroup_size(64)
fn retain_view_history(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    let receiver = receivers[id.x];
    if receiver.face >= 6u || owner_of(receiver) != id.x { return; }
    let change = resolve_retention(receiver.voxel, event_control.x, event_control.y,
        view.display.w > 0.0, 0u, 0u);
    // Resident events already changed the authority. Its snapshot is loaded
    // after this pass; applying them again would halve history twice.
    if estimates[id.x].identity.w == 0u || shared_slot(receiver, estimates[id.x].identity.y) == 0xffffffffu {
        if change.limit != NO_LIMIT { estimates[id.x].mean.w = min(estimates[id.x].mean.w, f32(change.limit)); }
        if change.measured { estimates[id.x].moment.w += change.change; }
    }
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
// Spatial pooling is MSE-optimal rather than count-targeted: each estimate
// is weighted by 1 / (variance of its mean + squared bias against this
// face), the spatial counterpart of the temporal history rule c sigma^2/D^2.
// A neighbour's bias is what its difference exceeds the two estimates'
// noise by, so light gradients and other surfaces fall away by themselves,
// and a lower-variance sampler needs less help without retuning. A face is
// shown alone once its relative standard error is under the noise threshold
// (`view.display.x`, `spatial_noise_threshold`).
//
// Per-sample variance from few samples is unreliable, and heavy-tailed light
// (emitters) hides its variance until rare bright paths arrive: it is shrunk
// toward relative sigma one, the Monte Carlo prior, with this many pseudo
// samples (4 let cave faces stop borrowing too early; a hard floor at one
// would stop a quieter sampler from ever needing less).
//
// Moments are per channel, so the luminance deviation is the weighted sum of
// the channel deviations: exact when the channels move together (one
// light's colour), an upper bound otherwise. Subtracting squared luminance
// from weighted squared channels instead read identical saturated samples
// as noise (blue: 11% relative error at 1,024 samples).
//
// A prefix that has seen no light is black so far, not certainly black: the
// prior scale is zero there. Its mean's standard error is floored at the
// brightest nearby estimate over n (`reference`, a rule-of-three bound on
// light it has not hit yet), which matters for a few samples and vanishes
// by a few hundred; a dark neighbourhood stays confidently dark.
const VARIANCE_PRIOR_SAMPLES: f32 = 64.0;
const LUMINANCE = vec3(0.2126, 0.7152, 0.0722);

fn mean_variance(estimate: ViewEstimate, reference: f32) -> f32 {
    let n = max(estimate.mean.w, 1.0);
    let mean = dot(estimate.mean.rgb, LUMINANCE);
    let channel = max(estimate.moment.rgb - estimate.mean.rgb * estimate.mean.rgb, vec3(0.0));
    let deviation = dot(sqrt(channel), LUMINANCE);
    let prior = mean * mean;
    let shrunk = (deviation * deviation * max(n - 1.0, 0.0) + prior * VARIANCE_PRIOR_SAMPLES)
        / (max(n - 1.0, 0.0) + VARIANCE_PRIOR_SAMPLES);
    let unseen = reference / n;
    return shrunk / n + unseen * unseen;
}

fn coplanar_donor(receiver: ViewReceiver, normal: vec3<f32>, candidate: u32) -> bool {
    let donor = receivers[candidate];
    if donor.face != receiver.face { return false; }
    // Sixteen voxels, or sixteen blocks of a coarse page: block faces are
    // `s` apart, and a fixed radius would leave far blocks no neighbours.
    let delta = donor.surface - receiver.surface;
    let reach = 16.0 * receiver_stride(receiver);
    return abs(dot(delta, normal)) <= 0.05 && dot(delta, delta) <= reach * reach && owner_of(donor) == candidate;
}

@compute @workgroup_size(64)
fn reconstruct_view(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    let receiver = receivers[id.x];
    if receiver.face >= 6u || owner_of(receiver) != id.x { return; }
    let center = estimates[id.x];
    let normal = face_normal(receiver.face);
    let tiles = tile_shape();
    let tile = pool.slots[id.x].tile;
    let xy = vec2<i32>(i32(tile % tiles.x), i32(tile / tiles.x));
    let center_luminance = dot(center.mean.rgb, LUMINANCE);
    // The brightest sampled coplanar estimate nearby (3x3 tiles).
    var reference = center_luminance;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = xy + vec2(x, y);
            if any(q < vec2(0)) || any(q >= vec2<i32>(tiles)) { continue; }
            var entry = atomicLoad(&pool.tiles[u32(q.y) * tiles.x + u32(q.x)].head);
            for (var visited = 0u; entry != 0u && visited < view.grid.z; visited++) {
                let candidate = entry - 1u;
                entry = pool.slots[candidate].next;
                if candidate == id.x || !coplanar_donor(receiver, normal, candidate) { continue; }
                let value = estimates[candidate];
                if value.mean.w > 0.0 { reference = max(reference, dot(value.mean.rgb, LUMINANCE)); }
            }
        }
    }
    let center_variance = mean_variance(center, reference);
    let tolerance = view.display.x * center_luminance;
    // How many times more evidence this face needs, as a gather radius.
    let deficit = center_variance / max(tolerance * tolerance, 1e-12);
    let radius = i32(clamp(ceil(sqrt(deficit / 3.14159) * 0.5), 1.0, 4.0));
    let unsampled = center.mean.w <= 0.0;
    var sum = vec3(0.0);
    var weight = 0.0;
    if !unsampled {
        let w = 1.0 / max(center_variance, 1e-12);
        sum = center.mean.rgb * w;
        weight = w;
    }
    if (unsampled || center_variance > tolerance * tolerance) && view.display.y > 0.0 {
        for (var y = -radius; y <= radius; y++) {
            for (var x = -radius; x <= radius; x++) {
                let q = xy + vec2(x, y);
                if any(q < vec2(0)) || any(q >= vec2<i32>(tiles)) { continue; }
                var entry = atomicLoad(&pool.tiles[u32(q.y) * tiles.x + u32(q.x)].head);
                for (var visited = 0u; entry != 0u && visited < view.grid.z; visited++) {
                    let candidate = entry - 1u;
                    entry = pool.slots[candidate].next;
                    if candidate == id.x || !coplanar_donor(receiver, normal, candidate) { continue; }
                    let value = estimates[candidate];
                    if value.mean.w <= 0.0 { continue; }
                    let variance = mean_variance(value, reference);
                    var bias2 = 0.0;
                    if !unsampled {
                        let difference = dot(value.mean.rgb, LUMINANCE) - center_luminance;
                        bias2 = max(difference * difference - variance - center_variance, 0.0);
                    }
                    let w = 1.0 / max(variance + bias2, 1e-12);
                    sum += value.mean.rgb * w;
                    weight += w;
                }
            }
        }
    }
    if weight <= 0.0 { return; }
    let desired = mix(sum / weight, center.mean.rgb, select(0.0, 1.0 - view.display.y, !unsampled));
    let blend = select(1.0, view.display.z, center.shown.w > 0.0);
    estimates[id.x].shown = vec4(mix(center.shown.rgb, desired, blend), 1.0);
}

@compute @workgroup_size(64)
fn publish_view(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    if id.x == 0u {
        far_output.viewport = view.viewport;
        far_output.shape = vec4(view.grid.xy, 1u | (view.history.w << 1u), view.grid.w);
    }
    let tiles = tile_shape();
    if id.x < tiles.x * tiles.y { far_output.heads[id.x] = atomicLoad(&pool.tiles[id.x].head); }
    far_output.links[id.x] = pool.slots[id.x].next;
    // Publish the already-built GPU index, not a second CPU world directory.
    // Striding covers its power-of-two padding even for non-power-of-two grids.
    for (var slot = id.x; slot <= view.grid.w; slot += view.grid.z) {
        far_output.keys[slot] = atomicLoad(&current_hash[slot]);
    }
    let receiver = receivers[id.x];
    var output: FarPixel;
    output.face = 0xffffffffu;
    if receiver.face < 6u {
        let owner = owner_of(receiver);
        if owner == id.x {
            let value = estimates[owner];
            output.surface = receiver.surface;
            output.face = receiver.face;
            output.irradiance = value.shown.rgb;
            // Custom direct RGB has no Bevy light. Keep it out of raw GI
            // moments and evaluate the same direct term as near presentation.
            // Ordinary raster lights and voxel emitters skip this work.
            if params.transport.y > 0.0 {
                output.irradiance += direct_at(receiver.surface, face_normal(receiver.face)).custom_visible;
            }
            output.evidence = select(0.0, max(value.mean.w, 0.001),
                value.shown.w > 0.0 || any(output.irradiance > vec3(0.0)));
            for (var corner = 0u; corner < 4u; corner++) {
                let byte = (value.identity.z >> (corner * 8u)) & 255u;
                output.sky[corner] = (byte & 15u) | ((byte & 128u) << 24u);
            }
            atomicAdd(&queue[QUEUE_HISTORY], u32(value.mean.w));
            atomicMax(&queue[QUEUE_MAX_HISTORY], u32(value.mean.w));
            if value.mean.w == 0.0 { atomicAdd(&queue[QUEUE_UNCOVERED], 1u); }
            if output.evidence > 0.0 { atomicAdd(&queue[QUEUE_DISPLAYED], 1u); }
            if continuations[owner].status == 1u { atomicAdd(&queue[QUEUE_PAUSED], 1u); }
        }
    }
    far_output.pixels[id.x] = output;
}
