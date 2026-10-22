// RGB face ownership handoff. A resident face has ONE authority, shared by
// worker and view command buffers in queue order. A view loads it before rays
// and stores it after rays; no pass reads bounce data while writing it.
// Presentation is a separate warm-start value, never transport evidence.
fn shared_slot(receiver: ViewReceiver, material: u32) -> u32 {
    if view.cache.z == 0u || view.cache.x != cache_guard[0] { return 0xffffffffu; }
    let header = cache_index[0];
    var bucket = receiver_hash(receiver) & header.x;
    for (var attempt = 0u; attempt < header.y; attempt++) {
        let entry = cache_index[bucket + 1u];
        if entry.w == 0xffffffffu { return 0xffffffffu; }
        if all(bitcast<vec3<i32>>(entry.xyz) == receiver.voxel) && (entry.w >> 29u) == receiver.face {
            let slot = entry.w & 0x1fffffffu;
            let key = shared_meta[slot * 3u];
            if (key.w & 8u) == 0u || (key.w >> 16u) != material { return 0xffffffffu; }
            return slot;
        }
        bucket = (bucket + 1u) & header.x;
    }
    return 0xffffffffu;
}

@compute @workgroup_size(64)
fn load_shared_faces(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    let receiver = receivers[id.x];
    if receiver.face >= 6u || owner_of(receiver) != id.x { return; }
    // A stale descriptor is not cache eviction. Keep provenance so the next
    // current descriptor loads the authority even if an edit lowered its count.
    if view.cache.z != 0u && view.cache.x != cache_guard[0] { return; }
    var estimate = estimates[id.x];
    let slot = shared_slot(receiver, estimate.identity.y);
    if slot == 0xffffffffu {
        // Keep valid local history across cache retirement or a stale render
        // generation. Re-admission transfers an estimate, never adds a copy.
        estimates[id.x].identity.w = 0u;
        return;
    }
    let base = slot * 9u;
    let incarnation = shared_state[base + 7u].x;
    let count = shared_state[base + 1u].y & 0x7fffffffu;
    let next = shared_state[base + 1u].z;
    atomicAdd(&queue[10], 1u);
    // On first admission retain the better-supported valid estimate. Histories
    // may share path prefixes: do NOT sum their counts or mix them as independent
    // measurements. Once admitted, always load the resident authority, including
    // a smaller retained count following an edit.
    if estimate.identity.w == incarnation || estimate.mean.w <= f32(count) {
        estimate.mean = vec4(bitcast<vec4<f32>>(shared_state[base]).rgb, f32(count));
        estimate.moment = vec4(bitcast<vec4<f32>>(shared_state[base + 2u]).rgb,
            bitcast<f32>(shared_meta[slot * 3u + 1u].x));
        estimate.identity.x = next;
    } else {
        atomicAdd(&queue[11], 1u);
        estimate.identity.x = max(estimate.identity.x, next);
    }
    if estimate.shown.w == 0.0 && count > 0u {
        let shown = bitcast<vec4<f32>>(shared_state[base + 6u]);
        estimate.shown = select(vec4(estimate.mean.rgb, 1.0), shown, shown.w > 0.0);
    }
    estimate.identity.w = incarnation;
    estimates[id.x] = estimate;
    receivers[id.x].sequence = estimate.identity.x;
    // An unfinished view path is speculative until completion. If background
    // work advanced this face, advance_view_path restarts at its new sequence.
}

@compute @workgroup_size(64)
fn store_shared_faces(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    let receiver = receivers[id.x];
    if receiver.face >= 6u || owner_of(receiver) != id.x { return; }
    let estimate = estimates[id.x];
    let slot = shared_slot(receiver, estimate.identity.y);
    if slot == 0xffffffffu { return; }
    let base = slot * 9u;
    if estimate.identity.w != shared_state[base + 7u].x { return; }
    let count = u32(estimate.mean.w);
    let advanced = shared_state[base + 1u].z != estimate.identity.x;
    let changed_count = (shared_state[base + 1u].y & 0x7fffffffu) != count;
    shared_state[base] = vec4(bitcast<vec3<u32>>(estimate.mean.rgb), 0u);
    shared_state[base + 2u] = vec4(bitcast<vec3<u32>>(estimate.moment.rgb), 0u);
    shared_state[base + 1u].x = 0u;
    shared_state[base + 3u].x = 0u;
    shared_state[base + 1u].y = count | select(0u, 0x80000000u, estimate.moment.w > 0.0);
    shared_state[base + 1u].z = estimate.identity.x;
    shared_meta[slot * 3u + 1u].x = bitcast<u32>(estimate.moment.w);
    // The presentation prior can contain borrowed support, but never enters
    // raw mean/moments or the physical count read by cached_energy.
    shared_state[base + 6u] = bitcast<vec4<u32>>(estimate.shown);
    shared_state[base + 1u].w = bitcast<u32>(estimate.shown.x);
    shared_state[base + 3u].y = bitcast<u32>(estimate.shown.y);
    shared_state[base + 3u].z = bitcast<u32>(estimate.shown.z);
    shared_state[base + 3u].w = bitcast<u32>(estimate.shown.w);
    // Refresh the scheduler's compact rank, not CPU mirror data. No extra
    // readback is introduced by visible sampling.
    if advanced || changed_count {
        let noise = sqrt(max(estimate.moment.rgb - estimate.mean.rgb * estimate.mean.rgb, vec3(0.0)))
            / (vec3(0.002) + abs(estimate.mean.rgb));
        let at = select(shared_meta[slot * 3u + 2u].y, view.cache.y, advanced);
        shared_meta[slot * 3u + 2u] = vec4(bitcast<u32>(max(noise.x, max(noise.y, noise.z))), at,
            bitcast<u32>(max(estimate.mean.x, max(estimate.mean.y, estimate.mean.z))), 0xa0000000u | count);
    }
}
