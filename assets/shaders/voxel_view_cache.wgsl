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
    let pending = continuations[id.x].status == 1u;
    let owner_count = atomicLoad(&queue[0]);
    let rank = atomicLoad(&queue[12u + view.grid.z + id.x]);
    let ordinal = (rank + owner_count - wave.range.x % max(owner_count, 1u)) % max(owner_count, 1u);
    var visits = 0u;
    if ordinal < min(owner_count, wave.range.y) && wave.range.z > 0u {
        visits = min(select(1u, 32u, view.history.w != 0u),
            (wave.range.y - ordinal + owner_count - 1u) / owner_count);
    }
    let first = max(next, estimate.identity.x);
    // Reserve attempts, not completed energy. A pending tail already owns its
    // original sequence. Trace/store release the unstarted suffix in this same
    // command buffer, before another scheduler can consume the cursor. Keeping
    // it would skip QMC indices according to path length and bias the prefix.
    let starts = visits - select(0u, 1u, pending && visits > 0u);
    shared_state[base + 1u].z = first + starts;
    atomicStore(&queue[12u + view.grid.z + id.x], first);
    estimate.identity.x = select(first, paths[id.x].sequence, pending);
    estimates[id.x] = estimate;
    receivers[id.x].sequence = estimate.identity.x;
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
    let changed_count = (shared_state[base + 1u].y & 0x7fffffffu) != count;
    let changed_mean = any(shared_state[base].rgb != bitcast<vec3<u32>>(estimate.mean.rgb));
    shared_state[base] = vec4(bitcast<vec3<u32>>(estimate.mean.rgb), 0u);
    shared_state[base + 2u] = vec4(bitcast<vec3<u32>>(estimate.moment.rgb), 0u);
    shared_state[base + 1u].x = 0u;
    shared_state[base + 3u].x = 0u;
    shared_state[base + 1u].y = count | select(0u, 0x80000000u, estimate.moment.w > 0.0);
    shared_state[base + 1u].z = atomicLoad(&queue[12u + view.grid.z + id.x]);
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
    if changed_mean || changed_count {
        let noise = sqrt(max(estimate.moment.rgb - estimate.mean.rgb * estimate.mean.rgb, vec3(0.0)))
            / (vec3(0.002) + abs(estimate.mean.rgb));
        shared_meta[slot * 3u + 2u] = vec4(bitcast<u32>(max(noise.x, max(noise.y, noise.z))), view.cache.y,
            bitcast<u32>(max(estimate.mean.x, max(estimate.mean.y, estimate.mean.z))), 0xa0000000u | count);
    }
}
