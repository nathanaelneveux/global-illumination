// Persistent measurement storage, independent of the ray estimator. Each face
// owns eight uint4s: the six existing result records, presentation, then metadata.
// Readback is a snapshot of this state, never the authority for advancing rays.
struct Controls { counts: vec4<u32>, options: vec4<u32> }
@group(0) @binding(0) var<uniform> settings: Controls;
@group(0) @binding(1) var<storage, read_write> tasks: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> state: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> inputs: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> seeds: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> spatial_jobs: array<vec4<u32>>;

@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) id: vec3<u32>) {
    let control = settings.counts;
    let i = id.x;
    if i >= settings.options.y { return; }
    let input = inputs[i];
    let base = input.x * 8u;
    if input.y != 0xffffffffu {
        for (var word = 0u; word < 8u; word += 1u) {
            state[base + word] = seeds[input.y * 8u + word];
        }
    }
    state[base + 1u].y = min(state[base + 1u].y, input.z);
    if input.w != 0xffffffffu {
        tasks[input.w * 2u + 1u].x = state[base + 1u].z;
    }
}

@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let control = settings.counts;
    let i = id.x;
    if i >= control.x { return; }
    let base = inputs[i].x * 8u;
    let result = i * 6u;
    let count = min(state[base + 1u].y + control.y, control.w);
    let alpha = min(f32(control.y) / f32(count), 1.0);
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
    state[base + 7u].z = 1u;
    // Spare result words carry the resolved presentation and its measurement
    // time without enlarging the five-band simulation snapshot.
    state[base + 1u].w = state[base + 6u].x;
    state[base + 3u] = vec4(state[base + 3u].x, state[base + 6u].yzw);
    for (var word = 0u; word < 6u; word += 1u) {
        output[result + word] = state[base + word];
    }
}

fn max3(value: vec3<f32>) -> f32 { return max(value.x, max(value.y, value.z)); }

// One bounded surface stencil per selected face, built on the worker from
// stable ownership. Four connected axial walks of length four, then the four
// immediate diagonals with an edge connection. No search through missing faces.
// Read only raw statistics and the ordinary temporal resolve; never another
// spatial result. The final presentation occupies existing snapshot words.
fn standard_error(base: u32) -> vec3<f32> {
    let mean = bitcast<vec4<f32>>(state[base]).rgb;
    let second = bitcast<vec4<f32>>(state[base + 2u]).rgb;
    let count = state[base + 1u].y;
    return sqrt(max(second - mean * mean, vec3(0.0)) / max(f32(count), 1.0));
}

@compute @workgroup_size(64)
fn reconstruct_spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= settings.counts.x { return; }
    let job = spatial_jobs[i * 11u];
    let base = job.x * 8u;
    if state[base + 7u].x != job.y { return; }
    let center = bitcast<vec4<f32>>(state[base + 6u]);
    let mean = bitcast<vec4<f32>>(state[base]).rgb;
    let error = standard_error(base);
    let noise = max3(error / (vec3(0.002) + abs(mean)));
    let blend = bitcast<f32>(settings.options.z) * noise * noise / (noise * noise + 0.15 * 0.15);
    var result = center.rgb;
    if blend > 0.001 {
        var sum = center.rgb;
        var weight = 1.0;
        for (var n = 0u; n < 20u; n += 1u) {
            let packed = spatial_jobs[i * 11u + 1u + n / 2u];
            let address = select(packed.xy, packed.zw, n % 2u != 0u);
            let donor = address.x * 8u;
            if address.y == 0u || donor + 7u >= arrayLength(&state) { continue; }
            if state[donor + 7u].x != address.y { continue; }
            let count = state[donor + 1u].y;
            if count == 0u { continue; }
            let donor_mean = bitcast<vec4<f32>>(state[donor]).rgb;
            let tolerance = vec3(0.002) + 0.1 * max(abs(mean), abs(donor_mean))
                + 2.0 * (error + standard_error(donor));
            let difference = max3(abs(mean - donor_mean) / tolerance);
            let distance = f32(n % 4u + 1u);
            let distance_squared = select(distance * distance, 2.0, n >= 16u);
            let w = exp(-0.125 * distance_squared - 0.5 * difference * difference)
                * f32(count) / (f32(count) + 8.0);
            sum += bitcast<vec4<f32>>(state[donor + 6u]).rgb * w;
            weight += w;
        }
        result = mix(center.rgb, sum / weight, blend);
    }
    // Keep the receiver's original confidence. Borrowing does not manufacture
    // independent samples or modify physical energy, moments, or temporal state.
    state[base + 1u].w = bitcast<u32>(result.r);
    state[base + 3u].y = bitcast<u32>(result.g);
    state[base + 3u].z = bitcast<u32>(result.b);
    state[base + 3u].w = bitcast<u32>(center.a);
    // CPU snapshots intentionally retain the temporal resolve written by
    // accumulate. Scheduling and simulation never consume spatially borrowed
    // values. End-to-end spatial quality is measured from rendered captures.
}
