// RGB face ownership handoff. A resident face has ONE authority, shared by
// worker and view command buffers in queue order. A view loads it before rays
// and stores it after rays; no pass reads bounce data while writing it.
// Display reconstruction is separate from transport. New identities may use
// the explicitly one-weight raw topology prior below, never filtered colors.
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

fn load_shared_face(index: u32, receiver: ViewReceiver) {
    // A stale descriptor is not cache eviction. Keep provenance so the next
    // current descriptor loads the authority even if an edit lowered its count.
    if view.cache.z != 0u && view.cache.x != cache_guard[0] { return; }
    var estimate = estimates[index];
    let slot = shared_slot(receiver, estimate.identity.y);
    if slot == 0xffffffffu {
        // Keep valid local history across cache retirement or a stale render
        // generation. Re-admission transfers an estimate, never adds a copy.
        estimates[index].identity.w = 0u;
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
        if estimate.identity.w != incarnation {
            // This independent local tail may overlap the imported prefix. It
            // was not reserved against this authority; start a fresh attempt.
            continuations[index].status = 0u;
        }
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
    let pending = continuations[index].status == 1u;
    let owner_count = atomicLoad(&queue[0]);
    let rank = atomicLoad(&queue[12u + view.grid.z + index]);
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
    atomicStore(&queue[12u + view.grid.z + index], first);
    estimate.identity.x = select(first, paths[index].sequence, pending);
    estimates[index] = estimate;
    receivers[index].sequence = estimate.identity.x;
}

// A topology prior is incident RGB, not conserved flux or a traced path. It
// contributes ONE unit of effective history and no QMC sequence advance.
// The 16-weight donor gate prevents unmeasured priors from copying each other;
// the ordinary transport-cache gate likewise requires fresh supporting rays.
// Read only last-frame estimates and the resident snapshot before store: never
// another invocation's newly seeded estimate in this dispatch.
struct FacePrior { mean: vec3<f32>, moment: vec3<f32>, weight: f32 }

fn prior_donor(voxel: vec3<i32>, face: u32) -> FacePrior {
    var key: ViewReceiver;
    key.voxel = voxel;
    key.face = face;
    var donor: FacePrior;
    let old = previous_owner(key);
    if old != 0xffffffffu {
        let value = previous_estimates[old];
        if value.mean.w >= f32(CACHE_MIN_SAMPLES) {
            donor = FacePrior(value.mean.rgb, value.moment.rgb, value.mean.w);
        }
    }
    let slot = shared_slot(key, voxel_material(voxel));
    if slot != 0xffffffffu {
        let base = slot * 9u;
        let count = shared_state[base + 1u].y & 0x7fffffffu;
        if count >= CACHE_MIN_SAMPLES && f32(count) > donor.weight {
            donor = FacePrior(bitcast<vec4<f32>>(shared_state[base]).rgb,
                bitcast<vec4<f32>>(shared_state[base + 2u]).rgb, f32(count));
        }
    }
    return donor;
}

fn add_prior(sum: FacePrior, donor: FacePrior, weight: f32) -> FacePrior {
    if donor.weight == 0.0 { return sum; }
    return FacePrior(sum.mean + donor.mean * weight, sum.moment + donor.moment * weight,
        sum.weight + weight);
}

fn prior_air(voxel: vec3<i32>) -> bool {
    let region = vec3<i32>(floor(vec3<f32>(voxel) / 16.0));
    return in_bounds(region) && lookup_region(region) != 0xffffffffu && voxel_material(voxel) == 256u;
}

fn bootstrap_face(receiver: ViewReceiver) -> FacePrior {
    // Explicit components avoid Naga/Metal signed arithmetic on packed int3.
    let origin = vec3<i32>(receiver.voxel.x, receiver.voxel.y, receiver.voxel.z);
    let normal = vec3<i32>(face_normal(receiver.face));
    let air = origin + normal;
    var prior: FacePrior;
    // Unknown geometry is not empty space. Never borrow through a wall.
    if voxel_material(origin) == 256u || !prior_air(air) { return prior; }
    for (var face = 0u; face < 6u; face++) {
        let offset = vec3<i32>(face_normal(face));
        let voxel = origin + offset;
        // The covered face proves this receiver's voxel WAS air. Other
        // matching normals must still connect through the new air pocket.
        if all(offset == -normal) || prior_air(voxel + normal) {
            prior = add_prior(prior, prior_donor(voxel, receiver.face), 1.0);
        }
    }
    if prior.weight == 0.0 {
        for (var face = 0u; face < 6u; face++) {
            let offset = vec3<i32>(face_normal(face));
            let alignment = dot(vec3<f32>(normal), vec3<f32>(offset));
            let weight = select(0.25, 1.0, alignment > 0.0);
            // Removed voxel's old faces; faces covered by placement; and
            // surviving faces bordering the receiver's known-empty cell.
            prior = add_prior(prior, prior_donor(air, face), weight);
            let covered = origin - offset;
            if any(covered != air) {
                prior = add_prior(prior, prior_donor(covered, face), weight);
            }
            prior = add_prior(prior, prior_donor(air - offset, face), weight);
        }
    }
    if prior.weight > 0.0 {
        prior.mean /= prior.weight;
        prior.moment = max(prior.moment / prior.weight, prior.mean * prior.mean);
        prior.weight = 1.0;
    }
    return prior;
}

@compute @workgroup_size(64)
fn load_shared_faces(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    let receiver = receivers[id.x];
    if receiver.face >= 6u || owner_of(receiver) != id.x { return; }
    load_shared_face(id.x, receiver);
    var estimate = estimates[id.x];
    // Never restart a sampled face after a retention cap. This initializes
    // only a new, unmeasured identity, after retention and cache admission.
    if estimate.mean.w == 0.0 && estimate.identity.x == 0u {
        let prior = bootstrap_face(receiver);
        if prior.weight > 0.0 {
            estimate.mean = vec4(prior.mean, prior.weight);
            // This is already a post-edit initialization, not old history to
            // invalidate on the first measured path.
            estimate.moment = vec4(prior.moment, 0.0);
            estimate.shown = vec4(prior.mean, 1.0);
            estimates[id.x] = estimate;
        }
    }
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
