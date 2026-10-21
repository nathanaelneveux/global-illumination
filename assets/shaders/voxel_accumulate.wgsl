// Persistent measurement storage, independent of the ray estimator. Each face
// owns nine uint4s: six result records, temporal RGB, metadata, render transition.
// Readback is a snapshot of this state, never the authority for advancing rays.
// schedule: (transport revision tag, stable error (f32), stable samples,
// audit interval ms); focus: (voxel.xyz, GPU pick budget); batch: (CPU
// traced faces, batch id (28 bits, never 0), first and end slot this
// batch's rotating classification covers); edits: (positions to measure,
// measure entries, urgent radius^2 for a measured / an unknown edit);
// mirror: (statistics-mirror records per batch, ...).
struct Controls {
    counts: vec4<u32>, options: vec4<u32>, display: vec4<u32>, limits: vec4<u32>,
    schedule: vec4<u32>, focus: vec4<u32>, batch: vec4<u32>, edits: vec4<u32>,
    mirror: vec4<u32>,
}
@group(0) @binding(0) var<uniform> settings: Controls;
@group(0) @binding(1) var<storage, read_write> tasks: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> state: array<vec4<u32>>;
// Per batch: CPU traced faces, then GPU pick records (written by
// `select_emit`; unused ones stay 0xffffffff), then cap and seed records.
@group(0) @binding(4) var<storage, read_write> inputs: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> seeds: array<vec4<u32>>;
// Persistent evidence donors: per-slot (start, len) into an arena of slots.
@group(0) @binding(6) var<storage, read> donor_lists: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> donor_arena: array<u32>;
@group(0) @binding(8) var<storage, read_write> face_index: array<vec4<u32>>;
@group(0) @binding(9) var<storage, read> index_scatter: array<u32>;

// Changed face-index entries from the worker's authoritative table. The table
// is hashed, so changes are scattered: one record upload per batch replaces
// one buffer write per changed run. Word 0 is the count, then (index, entry).
// Per slot: statistics before its last merge (mean, second moment, then
// [aux x, count, aux x, unused]). When the CPU rejects a merged batch (it was
// selected before an edit and still in flight then), the next selection asks
// for a rollback, so stale pre-edit light never stays in the GPU state.
@group(0) @binding(12) var<storage, read_write> checkpoint: array<vec4<u32>>;
@group(0) @binding(10) var<storage, read_write> scatter_target: array<u32>;
@group(0) @binding(11) var<storage, read> word_scatter: array<u32>;
// Per slot, three uint4s: (voxel.xyz, face | LIVE); (pending predicted
// change (f32), schedule, revision tag, transport tag); and the ranking
// record (noise per sample (f32), last merge ms, brightest visible band
// (f32), measured (bit 31) | urgent (bit 30) | mirror stale (bit 29) | count). Schedule: bits 0-2 stable batches, bit 3 fresh,
// bits 4-31 the batch of the last CPU job. Selection reads these and one
// state word per slot, not the 144-byte state. See resident.rs META_BYTES.
@group(0) @binding(13) var<storage, read_write> slot_meta: array<vec4<u32>>;
// Retention events: limits.y headers of three uint4s, then bounds records of
// two uint4s. See resident.rs `encode_events`.
@group(0) @binding(14) var<storage, read_write> events: array<vec4<u32>>;
// Luminance albedo and emission per material id (slot identity bits 16+).
@group(0) @binding(19) var<storage, read> material_light: array<vec2<f32>>;
// Edit bounds whose flux the GPU measures, three uint4s each: (bound record,
// first position, positions, changed voxels), (scan min, base flux), (scan
// size, 0). measure_max holds each entry's brightest outgoing luminance.
@group(0) @binding(20) var<storage, read> measures: array<vec4<u32>>;
@group(0) @binding(21) var<storage, read_write> measure_max: array<atomic<u32>>;
// Selection scratch, cleared per batch: three 256-bucket histograms, then
// per-class threshold buckets, remainders and tie counters, then the total.
@group(0) @binding(15) var<storage, read_write> selection: array<atomic<u32>>;
// Per slot: its class and key bucket this batch (group << 8 | bucket), or
// 0xffffffff when not a candidate. Written by classify, read by emit.
@group(0) @binding(16) var<storage, read_write> slot_keys: array<u32>;
// The CPU's statistics mirror for faces the GPU picked: per record (slot,
// incarnation, revision tag, 0) and the face's six result uint4s; the
// record count is selection[SEL_MIRRORS]. Classification emits faces whose mirror is
// stale, up to mirror.x per batch; the CPU merges only these.
@group(0) @binding(17) var<storage, read_write> mirror_out: array<vec4<u32>>;

// Word updates to a persistent buffer (donor arena or list table): word 0 is
// the count, then (word index, value) pairs. Indices are unique per batch.
@compute @workgroup_size(64)
fn scatter_words(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= word_scatter[0] { return; }
    scatter_target[word_scatter[1u + id.x * 2u]] = word_scatter[2u + id.x * 2u];
}

@compute @workgroup_size(64)
fn scatter_face_index(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= index_scatter[0] { return; }
    let r = 1u + id.x * 5u;
    face_index[index_scatter[r]] = vec4(
        index_scatter[r + 1u], index_scatter[r + 2u], index_scatter[r + 3u], index_scatter[r + 4u]);
}

@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) id: vec3<u32>) {
    let control = settings.counts;
    let i = id.x;
    if i >= settings.options.y { return; }
    let input = inputs[i];
    if input.x == 0xffffffffu { return; }
    let base = input.x * 9u;
    // Word y: a seed index (below 2^31) for a new slot. Otherwise bit 30
    // asks to undo the last merge (the CPU rejected it) and the low 30 bits
    // are a hard history cap (0x3fffffff: none). GPU state is authoritative:
    // an existing slot is never re-seeded from the CPU.
    let cp = input.x * 3u;
    if input.y < 0x80000000u {
        for (var word = 0u; word < 9u; word += 1u) {
            state[base + word] = seeds[input.y * 9u + word];
        }
    } else {
        if (input.y & 0x40000000u) != 0u {
            let saved = checkpoint[cp + 2u];
            state[base] = checkpoint[cp];
            state[base + 2u] = checkpoint[cp + 1u];
            state[base + 1u].x = saved.x;
            state[base + 1u].y = saved.y;
            state[base + 3u].x = saved.z;
        }
        state[base + 1u].y = min(state[base + 1u].y, input.y & 0x3fffffffu);
    }
    // Word z: the source-change cap (none: 0xffffffff). Retention from
    // edits and streaming is applied by `apply_events`, to every face.
    state[base + 1u].y = min(state[base + 1u].y, input.z);
    mirror_count(input.x);
    if input.w != 0xffffffffu {
        begin_trace(base, input.w);
        // A CPU job owns the face for its in-flight window: no GPU pick.
        let index = input.x * 3u + 1u;
        slot_meta[index].y = (slot_meta[index].y & 0xfu) | (settings.batch.y << 4u);
        slot_keys[input.x] = 0xffffffffu;
    }
}

// Selection reads a slot's count from its ranking record, never the state.
fn mirror_count(slot: u32) {
    let index = slot * 3u + 2u;
    slot_meta[index].w = (slot_meta[index].w & 0xe0000000u) | (state[slot * 9u + 1u].y & 0x1fffffffu);
}

// A face about to be traced: its task takes the current GPU sequence, and
// the render transition continues from what the renderer would show now,
// including an interrupted previous transition (never a raw/noisy donor).
fn begin_trace(base: u32, task: u32) {
    tasks[task * 2u + 1u].x = state[base + 1u].z;
    let next_rgb = bitcast<vec3<f32>>(vec3(state[base + 1u].w, state[base + 3u].yz));
    let previous = bitcast<vec4<f32>>(state[base + 8u]);
    let duration = state[base + 7u].z;
    let age = settings.display.x - state[base + 8u].w;
    // A render frame may precede a pipelined submission's counter. Treat
    // that as age zero; ordinary u32 wrap still advances the transition.
    let elapsed = select(age, 0u, age > 0x7fffffffu);
    let blend = select(1.0, min(f32(elapsed) / f32(max(duration, 1u)), 1.0), duration > 0u && duration <= 32u);
    state[base + 8u] = vec4(bitcast<vec3<u32>>(mix(previous.rgb, next_rgb, blend)), settings.display.x);
}

@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let control = settings.counts;
    let i = id.x;
    if i >= control.x || inputs[i].x == 0xffffffffu { return; }
    let base = inputs[i].x * 9u;
    let result = i * 6u;
    let cp = inputs[i].x * 3u;
    checkpoint[cp] = state[base];
    checkpoint[cp + 1u] = state[base + 2u];
    checkpoint[cp + 2u] = vec4(state[base + 1u].x, state[base + 1u].y & 0x7fffffffu, state[base + 3u].x, 0u);
    // Samples that reached unloaded space carry no information and were not
    // averaged; only the known count advances history.
    let fresh_count = u32(bitcast<f32>(output[result + 1u].y));
    // Retention marked by `mark_changed`: the predicted change accumulated
    // since this face was last traced. Consumed whether or not it binds, but
    // only by a batch with known paths: an all-unknown batch defers it
    // (`mark_changed` flags the face again on its next trace).
    let meta_index = inputs[i].x * 3u + 1u;
    let pending = bitcast<f32>(slot_meta[meta_index].x);
    if fresh_count > 0u { slot_meta[meta_index].x = 0u; }
    // Physical retention: keep H = c * sigma^2 / D^2 samples, the MSE optimum
    // for a change of size D (predicted by `apply_events`). sigma^2 is
    // the retained per-sample variance. A face that was dark relative to the
    // change has no retained noise to go on, so it uses the fresh batch's.
    // Lit faces never trust the fresh batch: right after an edit it still reads
    // neighbours' stale cached light. Measuring costs no extra rays.
    var previous_count = state[base + 1u].y & 0x7fffffffu;
    let physical = bitcast<f32>(settings.display.w);
    let change = select(0.0, pending, (state[base + 1u].y & 0x80000000u) != 0u);
    if physical > 0.0 && fresh_count > 0u && change > 0.0 {
        let w = vec3(0.2126, 0.7152, 0.0722);
        let old_mean = bitcast<vec4<f32>>(state[base]).rgb;
        let old_variance = max(bitcast<vec4<f32>>(state[base + 2u]).rgb - old_mean * old_mean, vec3(0.0));
        let fresh_mean = bitcast<vec4<f32>>(output[result]).rgb;
        let fresh_variance = max(bitcast<vec4<f32>>(output[result + 2u]).rgb - fresh_mean * fresh_mean, vec3(0.0));
        let variance = select(dot(w, old_variance), dot(w, fresh_variance), dot(w, old_mean) < 0.05 * change);
        let keep = physical * variance / (change * change);
        previous_count = min(previous_count, u32(min(keep, 65536.0)));
        // The change is consumed now: a rollback of this merge restores the
        // retained history, never the pre-change count.
        checkpoint[cp + 2u].y = min(checkpoint[cp + 2u].y, previous_count);
    }
    let count = min(previous_count + fresh_count, control.w);
    // Recency k >= 1 weights sample i of n by (i/n)^(k-1); k = 1 is the mean.
    let recency = bitcast<f32>(settings.display.z);
    let alpha = select(0.0, min(recency * f32(fresh_count) / f32(max(count, 1u)), 1.0), fresh_count > 0u);
    let mean = bitcast<vec4<f32>>(output[result]);
    let second = bitcast<vec4<f32>>(output[result + 2u]);
    let previous_mean = bitcast<vec4<f32>>(state[base]);
    let previous_second = bitcast<vec4<f32>>(state[base + 2u]);
    let previous_ionizing = state[base + 1u].x;
    state[base] = bitcast<vec4<u32>>(previous_mean + (mean - previous_mean) * alpha);
    state[base + 2u] = bitcast<vec4<u32>>(previous_second + (second - previous_second) * alpha);
    for (var word = 1u; word <= 3u; word += 2u) {
        let old = bitcast<f32>(state[base + word].x);
        let fresh = bitcast<f32>(output[result + word].x);
        state[base + word].x = bitcast<u32>(old + (fresh - old) * alpha);
    }
    state[base + 1u].y = count;
    state[base + 1u].z += control.y;
    state[base + 4u] = output[result + 4u];
    state[base + 5u] = output[result + 5u];
    let radiance = bitcast<vec4<f32>>(state[base]).rgb;
    let moment = bitcast<vec4<f32>>(state[base + 2u]).rgb;
    let error = sqrt(max(moment - radiance * radiance, vec3(0.0)) / max(f32(count), 1.0));
    let uncertainty = select(min(8.0, max3(error / (vec3(0.02) + abs(radiance)))), 1.0, count < 2u);
    let noise = select(min(8.0, max3(error / (vec3(0.002) + abs(radiance)))), 1.0, count < 2u);
    let elapsed = select(1.0 / 60.0, f32(control.z - state[base + 7u].y) * 0.001, state[base + 7u].z != 0u);
    let time_alpha = 1.0 - exp(-elapsed / bitcast<f32>(settings.options.x));
    let evidence = f32(count) / (f32(count) + 32.0);
    let blend = min(time_alpha, evidence / (1.0 + 16.0 * noise * noise));
    let previous = bitcast<vec4<f32>>(state[base + 6u]);
    let old_weight = (1.0 - blend) / (0.25 + max3(previous.rgb));
    let new_weight = blend / (0.25 + max3(radiance));
    let alpha_display = min(blend, new_weight / max(old_weight + new_weight, 0.0000001));
    let confidence = clamp(f32(count) / (f32(count) + 8.0) / (1.0 + uncertainty), 0.125, 1.0);
    state[base + 6u] = bitcast<vec4<u32>>(vec4(
        mix(previous.rgb, radiance, alpha_display),
        mix(previous.a, confidence, blend),
    ));
    state[base + 7u].y = control.z;
    // UINT_MAX denotes initialized, instantaneous publication.
    // Its nonzero bits also preserve the existing temporal-clock validity flag.
    state[base + 7u].z = select(0xffffffffu, settings.display.y, settings.display.y > 0u);
    // Spare result words carry the resolved presentation and its measurement
    // time without enlarging the five-band simulation snapshot.
    state[base + 1u].w = state[base + 6u].x;
    state[base + 3u] = vec4(state[base + 3u].x, state[base + 6u].yzw);
    for (var word = 0u; word < 6u; word += 1u) {
        output[result + word] = state[base + word];
    }
    // A GPU pick's result is not read back: its CPU mirror is now stale.
    // Its record reports the merged count instead (after any retention cap),
    // so the CPU plans its donor list from the GPU's history.
    if i >= settings.batch.x { inputs[i].z = count; }
    record_stability(inputs[i].x, previous_mean, bitcast<f32>(previous_ionizing), i >= settings.batch.x);
    // Readback-only flag: the resident count stays untouched. Slim CPU
    // receivers need the GPU verdict, not a duplicate of all its moments.
    output[result + 1u].y |= settled_flag(inputs[i].x);
}

fn settled_flag(slot: u32) -> u32 {
    return select(0u, 0x80000000u, (slot_meta[slot * 3u + 1u].y & 15u) == 12u);
}

// Per-face stability after a merge (scheduling.rs record_face_batch): stable
// batches advance while the face stays in the same transport revision, its
// display has settled on its mean, it has its required samples, and both its
// relative error and its drift in this batch are small. Then fresh.
fn record_stability(slot: u32, before: vec4<f32>, before_ionizing: f32, picked: bool) {
    let base = slot * 9u;
    let index = slot * 3u + 1u;
    let schedule = slot_meta[index];
    let same_revision = (schedule.y & 8u) != 0u && schedule.w == settings.schedule.x;
    let count = state[base + 1u].y & 0x7fffffffu;
    let after = bitcast<vec4<f32>>(state[base]);
    let after_ionizing = bitcast<f32>(state[base + 1u].x);
    let tolerance = bitcast<f32>(settings.schedule.y);
    let drift = max(
        max3(abs(before.xyz - after.xyz) / (vec3(0.002) + max(abs(before.xyz), abs(after.xyz)))),
        max(
            abs(before.w - after.w) / (0.002 + max(abs(before.w), abs(after.w))),
            abs(before_ionizing - after_ionizing) / (0.002 + max(abs(before_ionizing), abs(after_ionizing))),
        ),
    );
    let display = bitcast<vec4<f32>>(state[base + 6u]).rgb;
    let settled = max3(abs(display - after.xyz)) <= 0.001 + max3(after.xyz) * tolerance * 0.5;
    let stable = select(
        0u,
        min((schedule.y & 7u) + 1u, 4u),
        same_revision && settled && count >= required_samples(after.xyz)
            && refinement_error(base, count) <= tolerance && drift <= tolerance * 0.5,
    );
    slot_meta[index].y = (schedule.y & 0xfffffff0u) | 8u | stable;
    slot_meta[index].w = settings.schedule.x;
    slot_meta[slot * 3u + 2u] = vec4(
        bitcast<u32>(sample_noise(base)),
        settings.counts.z,
        bitcast<u32>(max3(after.xyz)),
        0x80000000u | count | select(slot_meta[slot * 3u + 2u].w & MIRROR_STALE, MIRROR_STALE, picked),
    );
}

// Worst per-band relative standard deviation of one sample (all five bands,
// 0.002 floor): refinement_error is this over sqrt(count), at most 8.
fn sample_noise(base: u32) -> f32 {
    let mean = bitcast<vec4<f32>>(state[base]);
    let second = bitcast<vec4<f32>>(state[base + 2u]);
    let mean_i = bitcast<f32>(state[base + 1u].x);
    let second_i = bitcast<f32>(state[base + 3u].x);
    let bands = sqrt(max(second - mean * mean, vec4(0.0))) / (vec4(0.002) + abs(mean));
    let ionizing = sqrt(max(second_i - mean_i * mean_i, 0.0)) / (0.002 + abs(mean_i));
    return max(max(max(bands.x, bands.y), max(bands.z, bands.w)), ionizing);
}

fn required_samples(mean: vec3<f32>) -> u32 {
    // Dark faces need more evidence before resting (scheduling.rs).
    return select(settings.schedule.z, max(settings.schedule.z, 256u), max3(mean) < 0.02);
}

// scheduling.rs refinement_error: worst per-band relative standard error over
// all five bands, with a 0.002 energy floor, at most 8.
fn refinement_error(base: u32, count: u32) -> f32 {
    if count < 2u { return 1.0; }
    let n = f32(count);
    let mean = bitcast<vec4<f32>>(state[base]);
    let second = bitcast<vec4<f32>>(state[base + 2u]);
    let mean_i = bitcast<f32>(state[base + 1u].x);
    let second_i = bitcast<f32>(state[base + 3u].x);
    let bands = sqrt(max(second - mean * mean, vec4(0.0)) / n) / (vec4(0.002) + abs(mean));
    let ionizing = sqrt(max(second_i - mean_i * mean_i, 0.0) / n) / (0.002 + abs(mean_i));
    return min(max(max(max(bands.x, bands.y), max(bands.z, bands.w)), ionizing), 8.0);
}

fn max3(value: vec3<f32>) -> f32 { return max(value.x, max(value.y, value.z)); }

// Standard error of a face's raw mean, from its moments and retained count.
fn standard_error(base: u32) -> vec3<f32> {
    let mean = bitcast<vec4<f32>>(state[base]).rgb;
    let second = bitcast<vec4<f32>>(state[base + 2u]).rgb;
    let count = state[base + 1u].y;
    return sqrt(max(second - mean * mean, vec3(0.0)) / max(f32(count), 1.0));
}

// Adaptive display support. Each header owns a disjoint range of unique,
// connected candidates; means/moments and temporal RGB are read-only inputs.
// Counts are a support heuristic, not independently traced rays at this face.
fn unpack_position_tag(tag: u32) -> vec3<u32> {
    return vec3(tag, tag >> 6u, tag >> 12u) & vec3(63u);
}

@compute @workgroup_size(64)
fn reconstruct_evidence(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= settings.counts.x || inputs[id.x].x == 0xffffffffu { return; }
    let slot = inputs[id.x].x;
    let base = slot * 9u;
    let count = state[base + 1u].y;
    let evidence_goal = f32(settings.options.w);
    if f32(count) >= evidence_goal { return; }
    // The walk's donors in order; this face needs a prefix sized by its own
    // current history (2 goal / history sites including itself, bounded).
    let list = donor_lists[slot];
    let sites = clamp(2u * ((settings.options.w + max(count, 1u) - 1u) / max(count, 1u)), 1u, settings.limits.x);
    let donors = min(list.y, sites - 1u);
    if donors == 0u { return; }
    let center = bitcast<vec4<f32>>(state[base + 6u]);
    // Donors are at most 16 voxels away. Modulo-64 coordinate differences
    // therefore recover exact distances even across negative/world boundaries.
    // Identity still uses the full slot/incarnation, not these periodic tags.
    let position = unpack_position_tag(state[base + 7u].w);
    let mean = bitcast<vec4<f32>>(state[base]).rgb;
    let error = standard_error(base);
    let n = max(f32(count), 1.0);
    let bandwidth2 = max(evidence_goal / n / 3.14159265359, 1.0);
    var sum = center.rgb * n;
    var weight = n;
    var squares = n;
    for (var i = 0u; i < donors; i += 1u) {
        let donor = donor_arena[list.x + i] * 9u;
        if donor + 7u >= arrayLength(&state) { continue; }
        let donor_count = f32(state[donor + 1u].y);
        if donor_count == 0.0 { continue; }
        let donor_mean = bitcast<vec4<f32>>(state[donor]).rgb;
        let tolerance = vec3(0.002) + 0.1 * max(abs(mean), abs(donor_mean))
            + 2.0 * (error + standard_error(donor));
        let difference = max3(abs(mean - donor_mean) / tolerance);
        let donor_position = unpack_position_tag(state[donor + 7u].w);
        let delta = vec3<i32>((donor_position + vec3(96u) - position) & vec3(63u)) - vec3(32);
        let distance2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        var w = exp(-0.5 * f32(distance2) / bandwidth2 - 0.5 * difference * difference) * donor_count;
        if w <= 0.00000001 { continue; }
        let next_effective = (weight + w) * (weight + w) / (squares + w * w / donor_count);
        let reached = next_effective >= evidence_goal;
        if reached {
            let c = weight * weight - evidence_goal * squares;
            let a = 1.0 - evidence_goal / donor_count;
            w = clamp(-c / (weight + sqrt(max(weight * weight - a * c, 0.0))), 0.0, w);
        }
        sum += bitcast<vec4<f32>>(state[donor + 6u]).rgb * w;
        weight += w;
        squares += w * w / donor_count;
        if reached { break; }
    }
    let result = mix(center.rgb, sum / weight, bitcast<f32>(settings.options.z));
    state[base + 1u].w = bitcast<u32>(result.r);
    state[base + 3u].y = bitcast<u32>(result.g);
    state[base + 3u].z = bitcast<u32>(result.b);
    state[base + 3u].w = bitcast<u32>(center.a);
}

const NO_LIMIT: u32 = 0xffffffffu;
const PI: f32 = 3.14159265358979;
const REGION_EDGE: i32 = 16;

// H(d) = round((d / 12)^4) from squared distance, exactly in integers
// (retention.rs `limit`). At d^2 >= 192^2 it reaches the history ceiling.
fn curve_limit(d2: u32) -> u32 {
    if d2 >= 36864u { return NO_LIMIT; }
    return (d2 * d2 + 10368u) / 20736u;
}

// geometry.rs Bounds::distance_squared, saturating each axis at 4096.
fn box_distance2(lo: vec3<i32>, hi: vec3<i32>, p: vec3<i32>) -> u32 {
    let separation = vec3<u32>(min(max(max(lo - p, p - hi), vec3(0)), vec3(4096)));
    return dot(separation, separation);
}

fn load_height(top: i32, y: i32) -> f32 {
    return f32(clamp(top + 1 - y, 0, REGION_EDGE));
}

// Retention for every resident face, at publication of each change, from
// the face's own statistics (retention.rs `resolve` is the CPU statement).
// Unknown flux, or the curve rule, caps the count now. A known measured
// change accumulates; `mark_changed` decides whether it binds when the face
// is next traced, and accumulate then applies it.
@compute @workgroup_size(64)
fn apply_events(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x;
    if slot >= settings.limits.z { return; }
    let identity = slot_meta[slot * 3u];
    if (identity.w & 8u) == 0u { return; }
    let p = bitcast<vec3<i32>>(identity.xyz);
    let reach = settings.limits.w;
    let event_count = settings.limits.y;
    var cap = NO_LIMIT;
    var change = 0.0;
    var known = true;
    // Source changes cap every face (sources.rs history_limit): the base cap
    // outside the changed lights' spheres, 4 within. Any event in reach, or
    // any source change, revokes stability: the face is due for an audit.
    var source_cap = NO_LIMIT;
    var woken = false;
    // Player edits (bounds flagged bit 3): faces within the urgent radius
    // whose history the change binds (or that were dark relative to it) are
    // urgent; with unknown flux or the curve rule, every face in the smaller
    // radius is (publication.rs used to decide this from CPU statistics).
    let measured_mode = bitcast<f32>(settings.display.w) > 0.0;
    var urgent_if_binding = false;
    var urgent = false;
    for (var e = 0u; e < event_count; e += 1u) {
        let h0 = events[e * 3u];
        let h1 = events[e * 3u + 1u];
        let h2 = events[e * 3u + 2u];
        if (h0.w & 2u) != 0u {
            var limit = h2.x;
            for (var b = 0u; b < h2.w; b += 1u) {
                let sphere = bitcast<vec4<f32>>(events[event_count * 3u + (h2.z + b) * 2u]);
                let d = vec3<f32>(p) - sphere.xyz;
                if dot(d, d) <= sphere.w { limit = min(limit, 4u); }
            }
            source_cap = min(source_cap, limit);
            woken = true;
            continue;
        }
        let outside = box_distance2(bitcast<vec3<i32>>(h0.xyz), bitcast<vec3<i32>>(h1.xyz), p);
        if outside > reach { continue; }
        woken = true;
        var exact = known;
        var nearest = NO_LIMIT;
        for (var b = 0u; b < h2.w; b += 1u) {
            let record = event_count * 3u + (h2.z + b) * 2u;
            let b0 = events[record];
            let b1 = events[record + 1u];
            let d = box_distance2(bitcast<vec3<i32>>(b0.xyz), bitcast<vec3<i32>>(b1.xyz), p);
            if d > reach { continue; }
            nearest = min(nearest, d);
            let edit_kind = b1.w & 7u;
            if (b1.w & 8u) != 0u && (edit_kind == 0u || edit_kind == 3u) {
                if edit_kind == 3u && measured_mode {
                    urgent_if_binding = urgent_if_binding || d <= settings.edits.z;
                } else {
                    urgent = urgent || d <= settings.edits.w;
                }
            }
            if exact {
                let kind = b1.w & 7u;
                if kind == 0u {
                    known = false;
                    exact = false;
                } else if kind != 1u {
                    let area = select(1.0, load_height(bitcast<i32>(b1.w) >> 3u, p.y), kind == 2u);
                    change += bitcast<f32>(b0.w) * area / (PI * f32(max(d, 1u)));
                }
            }
        }
        if nearest != NO_LIMIT { cap = min(cap, curve_limit(nearest)); }
    }
    let measured = bitcast<f32>(settings.display.w) > 0.0 && known;
    let limit = min(source_cap, select(cap, NO_LIMIT, measured));
    if limit != NO_LIMIT {
        let base = slot * 9u;
        state[base + 1u].y = min(state[base + 1u].y, limit);
        mirror_count(slot);
        // A later rollback restores the pre-merge statistics: cap them too.
        checkpoint[slot * 3u + 2u].y = min(checkpoint[slot * 3u + 2u].y, limit);
    }
    if measured && change > 0.0 {
        let index = slot * 3u + 1u;
        slot_meta[index].x = bitcast<u32>(bitcast<f32>(slot_meta[index].x) + change);
    }
    if urgent_if_binding {
        let pending = bitcast<f32>(slot_meta[slot * 3u + 1u].x);
        urgent = urgent || !measured || (pending > 0.0 && binds(slot, pending));
    }
    if woken || urgent {
        slot_meta[slot * 3u + 1u].y &= 0xfffffff0u;
    }
    if urgent {
        slot_meta[slot * 3u + 2u].w |= URGENT;
    }
}

// Measured retention: does a change of this size cut the face's history
// (c sigma^2 / D^2 below its count), or was the face dark relative to it?
fn binds(slot: u32, change: f32) -> bool {
    let base = slot * 9u;
    let w = vec3(0.2126, 0.7152, 0.0722);
    let mean = bitcast<vec4<f32>>(state[base]).rgb;
    let variance = max(bitcast<vec4<f32>>(state[base + 2u]).rgb - mean * mean, vec3(0.0));
    let lit = dot(w, mean) >= 0.05 * change;
    let keep = bitcast<f32>(settings.display.w) * dot(w, variance) / max(change * change, 1e-30);
    return !lit || keep < f32(state[base + 1u].y & 0x7fffffffu);
}

fn pcg(value: u32) -> u32 {
    let mixed = value * 747796405u + 2891336453u;
    let word = ((mixed >> ((mixed >> 28u) + 4u)) ^ mixed) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_voxel(voxel: vec3<i32>) -> u32 {
    return pcg(pcg(pcg(bitcast<u32>(voxel.x)) ^ bitcast<u32>(voxel.y)) ^ bitcast<u32>(voxel.z));
}

// The face index (voxel_trace.wgsl cached_energy): a face's slot or none.
fn face_slot(voxel: vec3<i32>, face: u32) -> u32 {
    let header = face_index[0];
    var probe = (hash_voxel(voxel) ^ pcg(face + 1u)) & header.x;
    for (var attempt = 0u; attempt < header.y; attempt += 1u) {
        let entry = face_index[probe + 1u];
        if entry.w == 0xffffffffu { return NO_LIMIT; }
        if all(bitcast<vec3<i32>>(entry.xyz) == voxel) && (entry.w >> 29u) == face {
            return entry.w & 0x1fffffffu;
        }
        probe = (probe + 1u) & header.x;
    }
    return NO_LIMIT;
}

// An edit's flux, measured from the GPU's own statistics (publication.rs
// brightest_outgoing did this from the CPU mirror): the brightest outgoing
// luminance (albedo times incident, plus emission) of any face within one
// voxel of the edit, one position per thread.
@compute @workgroup_size(64)
fn measure_edits(@builtin(global_invocation_id) id: vec3<u32>) {
    let t = id.x;
    if t >= settings.edits.x { return; }
    var e = 0u;
    for (var i = 0u; i < settings.edits.y; i += 1u) {
        let a = measures[i * 3u];
        if t >= a.y && t < a.y + a.z { e = i; break; }
    }
    let a = measures[e * 3u];
    let size = measures[e * 3u + 2u].xyz;
    let local = t - a.y;
    let offset = vec3(local % size.x, (local / size.x) % size.y, local / (size.x * size.y));
    let p = bitcast<vec3<i32>>(measures[e * 3u + 1u].xyz) + vec3<i32>(offset);
    let w = vec3(0.2126, 0.7152, 0.0722);
    var brightest = 0.0;
    for (var face = 0u; face < 6u; face += 1u) {
        let slot = face_slot(p, face);
        if slot == NO_LIMIT { continue; }
        let identity = slot_meta[slot * 3u];
        if (identity.w & 8u) == 0u { continue; }
        let base = slot * 9u;
        var incident = 0.0;
        if (state[base + 1u].y & 0x7fffffffu) > 0u {
            incident = dot(w, bitcast<vec4<f32>>(state[base]).rgb) + dot(w, bitcast<vec4<f32>>(state[base + 4u]).rgb);
        }
        let light = material_light[identity.w >> 16u];
        brightest = max(brightest, light.x * incident + light.y);
    }
    if brightest > 0.0 { atomicMax(&measure_max[e], bitcast<u32>(brightest)); }
}

// Each measured edit's flux: base (the changed voxels' own emission) plus
// the brightest outgoing luminance times two faces per changed voxel.
@compute @workgroup_size(64)
fn finish_edits(@builtin(global_invocation_id) id: vec3<u32>) {
    let e = id.x;
    if e >= settings.edits.y { return; }
    let a = measures[e * 3u];
    let base_flux = bitcast<f32>(measures[e * 3u + 1u].w);
    let brightest = bitcast<f32>(atomicLoad(&measure_max[e]));
    events[a.x].w = bitcast<u32>(base_flux + brightest * 2.0 * f32(a.w));
    events[a.x + 1u].w = (events[a.x + 1u].w & 0xfffffff8u) | 3u;
}

// A traced face with a pending measured change: does the change bind (would
// keep fewer samples than it has), or was the face dark relative to it? Then
// its stale cached light must not serve this batch's paths (the high count
// bit makes them continue explicitly) and accumulate applies the change.
// Otherwise the change is consumed without effect.
@compute @workgroup_size(64)
fn mark_changed(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= settings.counts.x || inputs[i].x == 0xffffffffu { return; }
    let slot = inputs[i].x;
    let index = slot * 3u + 1u;
    let pending = bitcast<f32>(slot_meta[index].x);
    if pending <= 0.0 { return; }
    let base = slot * 9u;
    if binds(slot, pending) {
        state[base + 1u].y |= 0x80000000u;
    } else {
        slot_meta[index].x = 0u;
    }
}

// GPU selection of bulk work (scheduling.rs states the CPU rules it
// mirrors). Every live, due face competes in one class: coverage (fewer
// than BOOTSTRAP samples; nearest the focus first), refinement (within two
// pages of the focus; highest error + age priority first) or sweep
// (elsewhere; oldest first). Quotas of the pick budget: 1/4, 1/2, 1/4;
// unused share is lent to refinement, then coverage, then sweep. A histogram
// over each class's 256 key buckets yields exact per-class thresholds.
const BOOTSTRAP: u32 = 8u;
// Classes: 0 coverage, 1 refinement, 2 sweep, 3 urgent (an edit's change
// binds the face's history, or the face was dark relative to it).
const CLASSES: u32 = 4u;
const NO_CLASS: u32 = 4u;
const URGENT: u32 = 0x40000000u;
const MIRROR_STALE: u32 = 0x20000000u;
const SEL_THRESHOLD: u32 = 1024u;
const SEL_REMAINDER: u32 = 1028u;
const SEL_TIES: u32 = 1032u;
const SEL_TOTAL: u32 = 1036u;
const SEL_MIRRORS: u32 = 1037u;

struct Candidate { group: u32, bucket: u32 }

fn page_of(p: vec3<i32>) -> vec3<i32> {
    return (p - select(vec3(0), vec3(31), p < vec3(0))) / 32;
}

fn candidate(slot: u32) -> Candidate {
    let none = Candidate(NO_CLASS, 0u);
    let identity = slot_meta[slot * 3u];
    if (identity.w & 8u) == 0u { return none; }
    let schedule = slot_meta[slot * 3u + 1u];
    // A CPU job's face stays the CPU's for its readback window (at most
    // three batches in flight), so its merge is never raced by a pick.
    let last_cpu = schedule.y >> 4u;
    if last_cpu != 0u && ((settings.batch.y - last_cpu) & 0x0fffffffu) < 4u { return none; }
    let rank = slot_meta[slot * 3u + 2u];
    let count = rank.w & 0x1fffffffu;
    let fresh = (schedule.y & 8u) != 0u && schedule.w == settings.schedule.x;
    let since = settings.counts.z - rank.y;
    let age_ms = select(select(since, 0u, since > 0x7fffffffu), 0x7fffffffu, rank.w < 0x80000000u);
    let required = select(settings.schedule.z, max(settings.schedule.z, 256u), bitcast<f32>(rank.z) < 0.02);
    let due = !fresh || count < required || (schedule.y & 7u) < 4u || age_ms >= settings.schedule.w
        || (rank.w & URGENT) != 0u;
    if !due { return none; }
    let p = bitcast<vec3<i32>>(identity.xyz);
    let focus = bitcast<vec3<i32>>(settings.focus.xyz);
    if (rank.w & URGENT) != 0u {
        let d = vec3<f32>(p - focus);
        return Candidate(3u, min(u32(log2(dot(d, d) + 1.0) * 8.0), 255u));
    }
    if count < BOOTSTRAP {
        let d = vec3<f32>(p - focus);
        return Candidate(0u, min(u32(log2(dot(d, d) + 1.0) * 8.0), 255u));
    }
    let pages = abs(page_of(p) - page_of(focus));
    let page_distance = max(pages.x, max(pages.y, pages.z));
    if page_distance <= 2 {
        let age = min(f32(age_ms) * 0.001, 10.0);
        let error = select(min(bitcast<f32>(rank.x) / sqrt(f32(count)), 8.0), 1.0, count < 2u);
        let priority = error + age * 0.1 + select(0.0, 0.25, page_distance <= 1);
        // Log buckets: about 4% relative resolution from 0.004 to 50.
        return Candidate(1u, 255u - min(u32(max(log2(priority) + 8.0, 0.0) * 16.0), 255u));
    }
    return Candidate(2u, 255u - min(u32(log2(f32(age_ms) + 1.0) * 8.0), 255u));
}

// Classification rotates: each batch re-ranks one contiguous quarter of the
// slots (selection is memory-bound; ~1 ms for all ~800k on an M4). Keys
// persist; a picked or CPU-owned face's key is cleared until re-ranked.
var<workgroup> group_mirrors: atomic<u32>;
var<workgroup> group_mirror_base: u32;

@compute @workgroup_size(256)
fn select_classify(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    if local == 0u { atomicStore(&group_mirrors, 0u); }
    workgroupBarrier();
    let slot = settings.batch.z + id.x;
    var stale = false;
    var mirror_index = 0u;
    if slot < settings.batch.w {
        let c = candidate(slot);
        slot_keys[slot] = select(0xffffffffu, (c.group << 8u) | c.bucket, c.group != NO_CLASS);
        stale = (slot_meta[slot * 3u + 2u].w & MIRROR_STALE) != 0u
            && (slot_meta[slot * 3u].w & 8u) != 0u;
        if stale { mirror_index = atomicAdd(&group_mirrors, 1u); }
    }
    workgroupBarrier();
    if local == 0u {
        let n = atomicLoad(&group_mirrors);
        group_mirror_base = select(0u, atomicAdd(&selection[SEL_MIRRORS], n), n > 0u);
    }
    workgroupBarrier();
    let record = group_mirror_base + mirror_index;
    if !stale || record >= settings.mirror.x { return; }
    // Emitted: the mirror is current once the CPU merges this record.
    slot_meta[slot * 3u + 2u].w &= ~MIRROR_STALE;
    let base = slot * 9u;
    let out = record * 7u;
    mirror_out[out] = vec4(slot, state[base + 7u].x, slot_meta[slot * 3u + 1u].z, 0u);
    for (var word = 0u; word < 6u; word += 1u) {
        mirror_out[out + 1u + word] = state[base + word];
    }
    mirror_out[out + 2u].y |= settled_flag(slot);
}

// Histograms of all keys build in workgroup memory, then only touched bins
// reach the global ones.
var<workgroup> local_histogram: array<atomic<u32>, 1024>;

@compute @workgroup_size(256)
fn select_histogram(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    for (var b = local; b < 1024u; b += 256u) { atomicStore(&local_histogram[b], 0u); }
    workgroupBarrier();
    if id.x < settings.limits.z {
        let key = slot_keys[id.x];
        if key != 0xffffffffu { atomicAdd(&local_histogram[min(key >> 8u, 3u) * 256u + (key & 255u)], 1u); }
    }
    workgroupBarrier();
    for (var b = local; b < 1024u; b += 256u) {
        let n = atomicLoad(&local_histogram[b]);
        if n != 0u { atomicAdd(&selection[b], n); }
    }
}

// One workgroup: the three histograms load into workgroup memory in
// parallel and an inclusive prefix scan per class finds each threshold
// bucket. (A single thread walking them made ~1,500 dependent global loads,
// about 1.5 ms.)
var<workgroup> scan: array<u32, 1024>;
var<workgroup> taken: array<u32, 4>;

@compute @workgroup_size(256)
fn select_threshold(@builtin(local_invocation_index) t: u32) {
    for (var c = 0u; c < CLASSES; c += 1u) {
        scan[c * 256u + t] = atomicLoad(&selection[c * 256u + t]);
    }
    workgroupBarrier();
    for (var step = 1u; step < 256u; step <<= 1u) {
        var add = array<u32, 4>(0u, 0u, 0u, 0u);
        for (var c = 0u; c < CLASSES; c += 1u) {
            if t >= step { add[c] = scan[c * 256u + t - step]; }
        }
        workgroupBarrier();
        for (var c = 0u; c < CLASSES; c += 1u) { scan[c * 256u + t] += add[c]; }
        workgroupBarrier();
    }
    if t == 0u {
        // Urgent faces first, up to three quarters of the budget (the CPU's
        // urgent queue preempted three of every four slots); the rest by
        // quota, unused share lent to refinement, coverage, sweep, urgent.
        let budget = settings.focus.w;
        var take = array<u32, 4>(0u, 0u, 0u, 0u);
        take[3] = min(scan[3u * 256u + 255u], budget - budget / 4u);
        let rest = budget - take[3];
        take[0] = min(rest / 4u, scan[255u]);
        take[1] = min(rest / 2u, scan[256u + 255u]);
        take[2] = min(rest - rest / 4u - rest / 2u, scan[512u + 255u]);
        var spare = budget - take[0] - take[1] - take[2] - take[3];
        let lend = array<u32, 4>(1u, 0u, 2u, 3u);
        for (var o = 0u; o < CLASSES; o += 1u) {
            let c = lend[o];
            let extra = min(spare, scan[c * 256u + 255u] - take[c]);
            take[c] += extra;
            spare -= extra;
        }
        for (var c = 0u; c < CLASSES; c += 1u) {
            taken[c] = take[c];
            atomicStore(&selection[SEL_THRESHOLD + c], 256u);
            atomicStore(&selection[SEL_REMAINDER + c], 0u);
        }
    }
    workgroupBarrier();
    for (var c = 0u; c < CLASSES; c += 1u) {
        let below = select(0u, scan[c * 256u + max(t, 1u) - 1u], t > 0u);
        if taken[c] > 0u && below < taken[c] && scan[c * 256u + t] >= taken[c] {
            atomicStore(&selection[SEL_THRESHOLD + c], t);
            atomicStore(&selection[SEL_REMAINDER + c], taken[c] - below);
        }
    }
}

// Picks append after the CPU's traced faces: a task (voxel, face, current
// sequence) and an input record (slot, revision tag, count before the
// merge, incarnation) the readback returns.
// Ties at a threshold bucket and the pick list are reserved per workgroup:
// one global atomic per workgroup, not per face (most faces can share one
// bucket, and per-face atomics on one counter serialized ~1.5 ms).
var<workgroup> group_ties: array<atomic<u32>, 4>;
var<workgroup> group_tie_base: array<u32, 4>;
var<workgroup> group_picks: atomic<u32>;
var<workgroup> group_pick_base: u32;

@compute @workgroup_size(256)
fn select_emit(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    if local < CLASSES { atomicStore(&group_ties[local], 0u); }
    if local == 0u { atomicStore(&group_picks, 0u); }
    workgroupBarrier();
    let slot = id.x;
    var key = 0xffffffffu;
    if slot < settings.limits.z { key = slot_keys[slot]; }
    let group = min(key >> 8u, 3u);
    let bucket = key & 255u;
    let threshold = atomicLoad(&selection[SEL_THRESHOLD + group]);
    let candidate = key != 0xffffffffu && bucket <= threshold;
    let tie = candidate && bucket == threshold;
    var tie_index = 0u;
    if tie { tie_index = atomicAdd(&group_ties[group], 1u); }
    workgroupBarrier();
    if local < CLASSES {
        let n = atomicLoad(&group_ties[local]);
        group_tie_base[local] = select(0u, atomicAdd(&selection[SEL_TIES + local], n), n > 0u);
    }
    workgroupBarrier();
    let accept = candidate
        && (!tie || group_tie_base[group] + tie_index < atomicLoad(&selection[SEL_REMAINDER + group]));
    var pick_index = 0u;
    if accept { pick_index = atomicAdd(&group_picks, 1u); }
    workgroupBarrier();
    if local == 0u {
        let n = atomicLoad(&group_picks);
        group_pick_base = select(0u, atomicAdd(&selection[SEL_TOTAL], n), n > 0u);
    }
    workgroupBarrier();
    let pick = group_pick_base + pick_index;
    if !accept || pick >= settings.focus.w { return; }
    // Re-ranked when its quarter comes round; until then not picked again.
    slot_keys[slot] = 0xffffffffu;
    let identity = slot_meta[slot * 3u];
    // A stale key of a retired slot: a hole in the pick list (skipped).
    if (identity.w & 8u) == 0u { return; }
    let task = settings.batch.x + pick;
    let base = slot * 9u;
    tasks[task * 2u] = vec4(identity.xyz, identity.w & 7u);
    begin_trace(base, task);
    // Read back: the slot's revision tag (stale picks are undone by the
    // probe's next CPU job) and, once `accumulate` replaces it, the merged
    // count (donor lists). No pass reads y or z of a pick record.
    inputs[task] = vec4(slot, slot_meta[slot * 3u + 1u].z, state[base + 1u].y & 0x3fffffffu, state[base + 7u].x);
}
