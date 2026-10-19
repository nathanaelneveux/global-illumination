// Persistent measurement storage, independent of the ray estimator. Each face
// owns nine uint4s: six result records, temporal RGB, metadata, render transition.
// Readback is a snapshot of this state, never the authority for advancing rays.
struct Controls { counts: vec4<u32>, options: vec4<u32>, display: vec4<u32>, limits: vec4<u32> }
@group(0) @binding(0) var<uniform> settings: Controls;
@group(0) @binding(1) var<storage, read_write> tasks: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> state: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> inputs: array<vec4<u32>>;
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
    // Physical retention passes a predicted change here instead of a cap;
    // accumulate applies it once the fresh batch's noise is known. Until then
    // a changed face must not serve other faces' paths its stale cached light:
    // the high count bit makes this batch's rays continue explicitly instead.
    // Cap-only records (unselected faces, input.w = MAX) always carry a cap.
    // A negative value is a fallback cap for a change of unknown flux.
    let measured_input = bitcast<f32>(input.z);
    if bitcast<f32>(settings.display.w) <= 0.0 || input.w == 0xffffffffu {
        state[base + 1u].y = min(state[base + 1u].y, input.z);
    } else if measured_input < 0.0 {
        state[base + 1u].y = min(state[base + 1u].y, u32(-measured_input));
    } else if measured_input > 0.0 {
        state[base + 1u].y |= 0x80000000u;
    }
    if input.w != 0xffffffffu {
        tasks[input.w * 2u + 1u].x = state[base + 1u].z;
        // Continue from what the renderer would show now, including an
        // interrupted previous transition. Never blend from a raw/noisy donor.
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
}

@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let control = settings.counts;
    let i = id.x;
    if i >= control.x { return; }
    let base = inputs[i].x * 9u;
    let result = i * 6u;
    let cp = inputs[i].x * 3u;
    checkpoint[cp] = state[base];
    checkpoint[cp + 1u] = state[base + 2u];
    checkpoint[cp + 2u] = vec4(state[base + 1u].x, state[base + 1u].y & 0x7fffffffu, state[base + 3u].x, 0u);
    // Samples that reached unloaded space carry no information and were not
    // averaged; only the known count advances history.
    let fresh_count = u32(bitcast<f32>(output[result + 1u].y));
    // Physical retention: keep H = c * sigma^2 / D^2 samples, the MSE optimum
    // for a change of size D (predicted by the CPU from the edit). sigma^2 is
    // the retained per-sample variance. A face that was dark relative to the
    // change has no retained noise to go on, so it uses the fresh batch's.
    // Lit faces never trust the fresh batch: right after an edit it still reads
    // neighbours' stale cached light. Measuring costs no extra rays.
    var previous_count = state[base + 1u].y & 0x7fffffffu;
    let physical = bitcast<f32>(settings.display.w);
    let change = bitcast<f32>(inputs[i].z);
    if physical > 0.0 && fresh_count > 0u && change > 0.0 {
        let w = vec3(0.2126, 0.7152, 0.0722);
        let old_mean = bitcast<vec4<f32>>(state[base]).rgb;
        let old_variance = max(bitcast<vec4<f32>>(state[base + 2u]).rgb - old_mean * old_mean, vec3(0.0));
        let fresh_mean = bitcast<vec4<f32>>(output[result]).rgb;
        let fresh_variance = max(bitcast<vec4<f32>>(output[result + 2u]).rgb - fresh_mean * fresh_mean, vec3(0.0));
        let variance = select(dot(w, old_variance), dot(w, fresh_variance), dot(w, old_mean) < 0.05 * change);
        let keep = physical * variance / (change * change);
        previous_count = min(previous_count, u32(min(keep, 65536.0)));
    }
    let count = min(previous_count + fresh_count, control.w);
    // Recency k >= 1 weights sample i of n by (i/n)^(k-1); k = 1 is the mean.
    let recency = bitcast<f32>(settings.display.z);
    let alpha = select(0.0, min(recency * f32(fresh_count) / f32(max(count, 1u)), 1.0), fresh_count > 0u);
    let mean = bitcast<vec4<f32>>(output[result]);
    let second = bitcast<vec4<f32>>(output[result + 2u]);
    let previous_mean = bitcast<vec4<f32>>(state[base]);
    let previous_second = bitcast<vec4<f32>>(state[base + 2u]);
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
    if id.x >= settings.counts.x { return; }
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
