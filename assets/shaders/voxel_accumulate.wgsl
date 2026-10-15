// Persistent measurement storage, independent of the ray estimator. Each face
// owns eight uint4s: the six existing result records, presentation, then metadata.
// Readback is a snapshot of this state, never the authority for advancing rays.
@group(0) @binding(0) var<uniform> control: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> tasks: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> state: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> inputs: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> seeds: array<vec4<u32>>;

@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= control.x { return; }
    let input = inputs[i];
    let base = input.x * 8u;
    if input.y != 0xffffffffu {
        for (var word = 0u; word < 8u; word += 1u) {
            state[base + word] = seeds[input.y * 8u + word];
        }
    }
    state[base + 1u].y = min(state[base + 1u].y, input.z);
    tasks[i * 2u + 1u].x = state[base + 1u].z;
}

@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
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
    state[base + 7u].y = control.z;
    for (var word = 0u; word < 6u; word += 1u) {
        output[result + word] = state[base + word];
    }
}
