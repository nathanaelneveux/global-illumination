// Surface-aware discovery backed by a shared receiver pool. Screen tiles index
// projected faces but do not reserve physical slots. Receiver and ray budgets
// remain bounded independently of the number of planes in a particular tile.
fn prepass_face(pixel: vec2<i32>) -> u32 {
    let decoded = textureLoad(normals, pixel, 0).xyz * 2.0 - 1.0;
    if max(max(abs(decoded.x), abs(decoded.y)), abs(decoded.z)) < 0.99 { return 0xffffffffu; }
    // RGB10A2 cannot represent axis-normal zero components exactly.
    if abs(decoded.x) > 0.99 { return select(0u, 1u, decoded.x < 0.0); }
    if abs(decoded.y) > 0.99 { return select(2u, 3u, decoded.y < 0.0); }
    return select(4u, 5u, decoded.z < 0.0);
}

fn prepass_position(pixel: vec2<i32>, z: f32) -> vec3<f32> {
    let uv = (vec2<f32>(pixel) + 0.5 - vec2<f32>(view.viewport.xy)) / vec2<f32>(view.viewport.zw);
    let clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    let homogeneous = view.world_from_clip * clip;
    return homogeneous.xyz / homogeneous.w;
}

fn prepass_receiver(pixel: vec2<i32>) -> ViewReceiver {
    return block_receiver(prepass_voxel_receiver(pixel));
}

// A face of a coarse block is one receiver: the block's minimum voxel and its
// face centre. Unit-stride faces are unchanged.
fn block_receiver(r: ViewReceiver) -> ViewReceiver {
    if r.face >= 6u { return r; }
    let s = voxel_stride(r.voxel);
    if s == 1 { return r; }
    var block = r;
    block.voxel = r.voxel & vec3(-s);
    block.surface = vec3<f32>(block.voxel) + f32(s) * (0.5 + 0.5 * face_normal(r.face));
    return block;
}
fn receiver_stride(r: ViewReceiver) -> f32 {
    let u = (r.face / 2u + 1u) % 3u;
    return 2.0 * (r.surface[u] - f32(r.voxel[u]));
}

fn prepass_voxel_receiver(pixel: vec2<i32>) -> ViewReceiver {
    var receiver: ViewReceiver;
    receiver.face = 0xffffffffu;
    let z = textureLoad(depth, pixel, 0);
    if z <= 0.0 { return receiver; }
    let position = prepass_position(pixel, z);
    if view.camera.w > 0.0 && distance(position, view.camera.xyz) < view.camera.w { return receiver; }
    receiver.face = prepass_face(pixel);
    if receiver.face >= 6u { return receiver; }
    let normal = face_normal(receiver.face);
    // MSAA depth belongs to a subpixel sample. Snap the normal axis before
    // selecting the voxel so grazing faces do not become buried receivers.
    var surface = position;
    surface[receiver.face / 2u] = round(surface[receiver.face / 2u]);
    receiver.voxel = vec3<i32>(floor(surface - normal * 0.01));
    receiver.surface = vec3<f32>(receiver.voxel) + 0.5 + normal * 0.5;
    if receiver_exists(receiver) { return receiver; }
    // Depth and subpixel-sample error can put a distant face's decoded point
    // across an in-plane voxel boundary, onto air or a buried face. Within one
    // pixel's world footprint of that boundary, take the exposed neighbour
    // instead; a one-pixel ledge side would otherwise never get an owner.
    // Genuine mesh/transport skew decodes well inside its voxel and stays.
    let footprint = distance(prepass_position(pixel + vec2(1, 0), z), position);
    let axis = receiver.face / 2u;
    let u = (axis + 1u) % 3u;
    let v = (axis + 2u) % 3u;
    let local = surface - floor(surface);
    let gap = vec2(min(local[u], 1.0 - local[u]), min(local[v], 1.0 - local[v]));
    let step = vec2<i32>(select(1, -1, local[u] < 0.5), select(1, -1, local[v] < 0.5));
    var best = receiver;
    var best_gap = 2.0;
    for (var candidate = 1u; candidate < 4u; candidate++) {
        var offset = vec3<i32>(0);
        var needed = 0.0;
        if (candidate & 1u) != 0u { offset[u] = step.x; needed = max(needed, gap.x); }
        if (candidate & 2u) != 0u { offset[v] = step.y; needed = max(needed, gap.y); }
        if needed > footprint || needed >= best_gap { continue; }
        var neighbour = receiver;
        neighbour.voxel = receiver.voxel + offset;
        if receiver_exists(neighbour) {
            neighbour.surface = vec3<f32>(neighbour.voxel) + 0.5 + normal * 0.5;
            best = neighbour;
            best_gap = needed;
        }
    }
    if receiver_exists(best) { return best; }
    // Geometry not yet published keeps its raster receiver (pending); the
    // traced fallback needs GI's own.
    if !region_loaded(receiver.voxel) { return best; }
    return traced_receiver(position, footprint, best);
}

/// A host mesh that is not GI's geometry (BVW LOD chunks, impostor or
/// foreign meshes) puts its depth beside the true voxel surface, by up to a
/// LOD stride. Trace this pixel's view ray through GI's own geometry around
/// that depth, as NAADF's primary rays do, and take the first exposed face:
/// identity then comes from the structure GI lights, not from the raster.
/// The window grows with the pixel's world footprint (LOD strides grow with
/// distance); the step budget is bounded like NAADF's primary rays.
const TRACED_DISCOVERY_STEPS: u32 = 64u;
/// Footprint-search rounds (new planes) per tile per frame.
const DISCOVERY_ROUNDS: u32 = 8u;

fn region_loaded(voxel: vec3<i32>) -> bool {
    let region = vec3<i32>(floor(vec3<f32>(voxel) / 16.0));
    return in_bounds(region) && lookup_region(region) != 0xffffffffu;
}
fn traced_receiver(position: vec3<f32>, footprint: f32, fallback: ViewReceiver) -> ViewReceiver {
    let direction = normalize(position - view.camera.xyz);
    let margin = clamp(32.0 * footprint, 2.0, 64.0);
    let hit = trace_domain_limited(position - direction * margin, direction, 2.0 * margin, true,
        TRACED_DISCOVERY_STEPS);
    // Loaded geometry with no exposed face along this pixel's ray: not
    // pending geometry, just no GI surface here. Returning the raster face
    // would mark the tile pending and retrace it every frame.
    var none: ViewReceiver;
    none.face = 0xffffffffu;
    if hit.status != 1u { return none; }
    var receiver: ViewReceiver;
    receiver.face = face_index_of(hit.normal);
    if receiver.face >= 6u { return none; }
    receiver.voxel = hit.voxel;
    receiver.surface = vec3<f32>(hit.voxel) + 0.5 + face_normal(receiver.face) * 0.5;
    receiver.sequence = fallback.sequence;
    if !receiver_exists(receiver) { return none; }
    return receiver;
}

fn same_plane(a: ViewReceiver, b: ViewReceiver) -> bool {
    if a.face >= 6u || a.face != b.face { return false; }
    return a.surface[a.face / 2u] == b.surface[b.face / 2u];
}

// Receiver slots are global; screen tiles only own linked spatial indices.
// A separate dispatch validates retained witnesses and publishes the free list
// before any tile allocates. No cross-workgroup spin or partially written key.
struct PoolTile {
    head: atomic<u32>, dirty: atomic<u32>,
    fingerprint_xor: u32, fingerprint_sum: u32, pending: atomic<u32>,
}
struct PoolSlot { witness: u32, tile: u32, next: u32, free: u32, }
struct ReceiverPool {
    free_count: atomic<u32>, claimed: atomic<u32>,
    exhausted: atomic<u32>, retained: atomic<u32>,
    tiles: array<PoolTile, 65536>,
    slots: array<PoolSlot>,
}
@group(2) @binding(17) var<storage, read_write> pool: ReceiverPool;
@group(2) @binding(18) var<storage, read_write> previous_pool: ReceiverPool;

fn tile_shape() -> vec2<u32> { return (view.grid.xy + 1u) / 2u; }
fn tile_of_pixel(pixel: vec2<i32>) -> u32 {
    let cell = (vec2<u32>(pixel) - view.viewport.xy) * view.grid.xy / view.viewport.zw;
    let tile = min(cell, view.grid.xy - 1u) / 2u;
    return tile.y * tile_shape().x + tile.x;
}

// Reproject the visible point, not the voxel centre: a partially occluded or
// grazing face's centre may never have been visible. Face-local coordinates
// keep the physical point fixed; snapping to each new pixel centre would let
// it drift across the voxel and eventually discard an otherwise valid face.
fn reproject_witness(receiver: ViewReceiver, packed: u32) -> vec2<i32> {
    let axis = receiver.face / 2u;
    let uv = unpack2x16unorm(packed);
    let s = receiver_stride(receiver);
    var point = vec3<f32>(receiver.voxel);
    point[axis] = receiver.surface[axis];
    point[(axis + 1u) % 3u] += uv.x * s;
    point[(axis + 2u) % 3u] += uv.y * s;
    let clip = view.clip_from_world * vec4(point, 1.0);
    if clip.w <= 0.0 { return vec2(-1); }
    let projected = clip.xy / clip.w * vec2(0.5, -0.5) + 0.5;
    return vec2<i32>(floor(vec2<f32>(view.viewport.xy) + projected * vec2<f32>(view.viewport.zw)));
}

fn pack_witness(receiver: ViewReceiver, pixel: vec2<i32>) -> u32 {
    let z = textureLoad(depth, pixel, 0);
    let local = (prepass_position(pixel, z) - vec3<f32>(receiver.voxel)) / receiver_stride(receiver);
    let axis = receiver.face / 2u;
    let uv = vec2(local[(axis + 1u) % 3u], local[(axis + 2u) % 3u]);
    return pack2x16unorm(clamp(uv, vec2(1.0 / 65535.0), vec2(1.0 - 1.0 / 65535.0)));
}

fn receiver_exists(receiver: ViewReceiver) -> bool {
    if receiver.face >= 6u { return false; }
    // The voxel just outside the face; for a coarse block that is past the
    // block (any voxel of it, the block's minimum once canonical).
    let cell = voxel_cell(receiver.voxel);
    if cell.x == 256u { return false; }
    let s = 1 << cell.y;
    let axis = receiver.face / 2u;
    let block = receiver.voxel[axis] & -s;
    var outside = vec3<i32>(receiver.voxel.x, receiver.voxel.y, receiver.voxel.z);
    outside[axis] = select(block - 1, block + s, (receiver.face & 1u) == 0u);
    return voxel_material(outside) == 256u;
}


fn publish_receiver(index: u32, receiver: ViewReceiver) {
    var estimate: ViewEstimate;
    if receiver.face < 6u {
        estimate.identity.y = voxel_material(receiver.voxel);
        if view.history.w != 0u && !receiver_exists(receiver) {
            estimate.identity.y |= PENDING_GEOMETRY;
        }
    }
    receivers[index] = receiver;
    estimates[index] = estimate;
}

// Only the canonical previous owner can retain a face. Its key is unique, so
// insertion needs no reads of concurrently initialized current receivers.
// Discovery reads this index only after the whole preparation dispatch ends.
fn index_retained_receiver(index: u32, receiver: ViewReceiver) {
    var slot = receiver_hash(receiver) & view.grid.w;
    for (var step = 0u; step < 64u; step++) {
        let claim = atomicCompareExchangeWeak(&current_hash[slot], 0u, index + 1u);
        if claim.exchanged { return; }
        if claim.old_value != 0u { slot = (slot + 1u) & view.grid.w; }
    }
    atomicAdd(&queue[1], 1u);
}

// All old slots are classified before discovery. Surviving faces keep their
// physical slot even when they cross tile boundaries; face-keyed reprojection
// still owns raw moments, QMC indices and unfinished paths.
@compute @workgroup_size(64)
fn prepare_receiver_pool(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.grid.z { return; }
    let tiles = tile_shape();
    if id.x < tiles.x * tiles.y && view.discovery.y != 0u {
        pool.tiles[id.x].fingerprint_xor = previous_pool.tiles[id.x].fingerprint_xor;
        pool.tiles[id.x].fingerprint_sum = previous_pool.tiles[id.x].fingerprint_sum;
        if atomicLoad(&previous_pool.tiles[id.x].pending) != 0u {
            atomicOr(&pool.tiles[id.x].dirty, 1u);
        }
    }
    var receiver: ViewReceiver;
    receiver.face = 0xffffffffu;
    if view.history.x != 0u && view.discovery.y != 0u {
        let old = previous_receivers[id.x];
        if old.face < 6u && previous_owner(old) != id.x {
            // Removing a duplicate also changes its old spatial bin. The
            // surviving face key remains globally addressable, but that bin
            // may need another representative for neighbouring surfaces.
            atomicOr(&pool.tiles[previous_pool.slots[id.x].tile].dirty, 1u);
        }
        if old.face < 6u && previous_owner(old) == id.x {
            let pixel = reproject_witness(old, previous_pool.slots[id.x].witness);
            let in_view = all(pixel >= vec2<i32>(view.viewport.xy))
                && all(pixel < vec2<i32>(view.viewport.xy + view.viewport.zw));
            var valid = false;
            if in_view {
                valid = same_receiver(old, prepass_receiver(pixel))
                    && (view.history.w != 0u || receiver_exists(old));
            }
            let old_tile = previous_pool.slots[id.x].tile;
            if valid {
                receiver = old;
                let tile = tile_of_pixel(pixel);
                pool.slots[id.x].witness = previous_pool.slots[id.x].witness;
                pool.slots[id.x].tile = tile;
                pool.slots[id.x].next = atomicExchange(&pool.tiles[tile].head, id.x + 1u);
                index_retained_receiver(id.x, old);
                atomicAdd(&pool.retained, 1u);
                // Crossing a spatial-bin boundary is not lost visibility.
                // Both bins keep the normal four-phase discovery cadence;
                // an empty bin checks its centres below. Only an actually
                // rejected witness forces an immediate footprint search.
            } else { atomicOr(&pool.tiles[old_tile].dirty, 1u); }
        }
    }
    publish_receiver(id.x, receiver);
    if receiver.face >= 6u {
        let free = atomicAdd(&pool.free_count, 1u);
        pool.slots[free].free = id.x;
    }
}

fn pool_covers(tile: u32, receiver: ViewReceiver) -> bool {
    // Presentation's reach: sixteen voxels, or sixteen blocks.
    let reach = 16.0 * receiver_stride(receiver);
    var entry = atomicLoad(&pool.tiles[tile].head);
    for (var visited = 0u; entry != 0u && visited < view.grid.z; visited++) {
        let other = receivers[entry - 1u];
        let delta = other.surface - receiver.surface;
        if same_plane(other, receiver) && dot(delta, delta) <= reach * reach { return true; }
        entry = pool.slots[entry - 1u].next;
    }
    // Only a missing local plane needs a global lookup. This immutable index
    // contains fully prepared retained faces, never concurrent allocations.
    return owner_of(receiver) != 0xffffffffu;
}

fn occupied_cell(tile: u32, cell: vec2<u32>) -> bool {
    var entry = atomicLoad(&pool.tiles[tile].head);
    for (var visited = 0u; entry != 0u && visited < view.grid.z; visited++) {
        let index = entry - 1u;
        let pixel = reproject_witness(receivers[index], pool.slots[index].witness);
        let q = (vec2<u32>(pixel) - view.viewport.xy) * view.grid.xy / view.viewport.zw;
        if all(q == cell) { return true; }
        entry = pool.slots[index].next;
    }
    return false;
}

// Called by one invocation per tile. Free slots are from the completed prepare
// pass; the slot's identity and links are visible to this tile after a barrier.
fn allocate_receiver(tile: u32, pixel: vec2<i32>, receiver: ViewReceiver) -> bool {
    let claim = atomicAdd(&pool.claimed, 1u);
    if claim >= atomicLoad(&pool.free_count) {
        atomicAdd(&pool.exhausted, 1u);
        return false;
    }
    let index = pool.slots[claim].free;
    publish_receiver(index, receiver);
    pool.slots[index].witness = pack_witness(receiver, pixel);
    pool.slots[index].tile = tile;
    pool.slots[index].next = atomicLoad(&pool.tiles[tile].head);
    atomicStore(&pool.tiles[tile].head, index + 1u);
    return true;
}

var<workgroup> pool_winner: atomic<u32>;
var<workgroup> pool_hash_xor: atomic<u32>;
var<workgroup> pool_hash_sum: atomic<u32>;
var<workgroup> pool_done: u32;
var<workgroup> pool_dirty: u32;

@compute @workgroup_size(32)
fn discover_receivers(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) thread: u32) {
    let tiles = tile_shape();
    if id.x >= tiles.x * tiles.y { return; }
    let cell = vec2(id.x % tiles.x, id.x / tiles.x) * 2u;
    let begin = view.viewport.xy + cell * view.viewport.zw / view.grid.xy;
    let end = view.viewport.xy + min(cell + 2u, view.grid.xy) * view.viewport.zw / view.grid.xy;
    let extent = end - begin;
    let pixels = extent.x * extent.y;
    let check = (view.history.z + id.x) % 4u == 0u;
    if thread == 0u {
        pool_dirty = u32(atomicLoad(&pool.tiles[id.x].dirty) != 0u);
        if view.discovery.x == 0u && atomicLoad(&pool.tiles[id.x].head) == 0u {
            for (var lane = 0u; lane < 4u; lane++) {
                let q = cell + vec2(lane & 1u, lane >> 1u);
                if any(q >= view.grid.xy) { continue; }
                let uv = (vec2<f32>(q) + 0.5) / vec2<f32>(view.grid.xy);
                let pixel = vec2<i32>(view.viewport.xy + min(vec2<u32>(uv * vec2<f32>(view.viewport.zw)), view.viewport.zw - 1u));
                if prepass_receiver(pixel).face < 6u { pool_dirty = 1u; break; }
            }
        }
        pool_done = u32(view.history.x != 0u && view.discovery.y != 0u && pool_dirty == 0u && !check);
    }
    if workgroupUniformLoad(&pool_done) != 0u { return; }
    let dirty = workgroupUniformLoad(&pool_dirty) != 0u;

    if thread == 0u {
        atomicStore(&pool_hash_xor, 0u);
        atomicStore(&pool_hash_sum, 0u);
    }
    workgroupBarrier();
    if view.discovery.x != 0u {
        var xor = 0u;
        var sum = 0u;
        for (var offset = thread; offset < pixels; offset += 32u) {
            let pixel = vec2<i32>(begin + vec2(offset % extent.x, offset / extent.x));
            let z = textureLoad(depth, pixel, 0);
            let face = select(0xffffffffu, prepass_face(pixel), z > 0.0);
            let value = pcg(bitcast<u32>(z) ^ pcg(offset * 7u + face));
            xor ^= value;
            sum += pcg(value ^ 0x9e3779b9u);
        }
        atomicXor(&pool_hash_xor, xor);
        atomicAdd(&pool_hash_sum, sum);
    }
    workgroupBarrier();
    if thread == 0u {
        let matched = pool.tiles[id.x].fingerprint_xor == atomicLoad(&pool_hash_xor)
            && pool.tiles[id.x].fingerprint_sum == atomicLoad(&pool_hash_sum);
        pool_done = u32(view.discovery.x != 0u && !dirty && matched
            && (view.history.z + id.x) % 64u != 0u);
        pool.tiles[id.x].fingerprint_xor = atomicLoad(&pool_hash_xor);
        pool.tiles[id.x].fingerprint_sum = atomicLoad(&pool_hash_sum);
    }
    if workgroupUniformLoad(&pool_done) != 0u { return; }

    // Keep the original coarse density where useful, without tying storage to
    // those cells. Existing projected witnesses satisfy their current cell.
    if thread == 0u {
        pool_done = 0u;
        for (var lane = 0u; lane < 4u; lane++) {
            let q = cell + vec2(lane & 1u, lane >> 1u);
            if any(q >= view.grid.xy) || occupied_cell(id.x, q) { continue; }
            let uv = (vec2<f32>(q) + 0.5) / vec2<f32>(view.grid.xy);
            let pixel = vec2<i32>(view.viewport.xy + min(vec2<u32>(uv * vec2<f32>(view.viewport.zw)), view.viewport.zw - 1u));
            let receiver = prepass_receiver(pixel);
            if receiver.face >= 6u { continue; }
            if owner_of(receiver) != 0xffffffffu { continue; }
            if view.history.w != 0u || receiver_exists(receiver) {
                if !allocate_receiver(id.x, pixel, receiver) { pool_done = 1u; break; }
            } else { atomicStore(&pool.tiles[id.x].pending, 1u); }
        }
    }
    storageBarrier();
    if workgroupUniformLoad(&pool_done) != 0u { return; }

    // Every newly represented plane removes its candidates from the next round.
    // The bound is the actual footprint, not a per-tile receiver quota.
    // A crowded tile (distant terrain: many one-pixel planes) needs a round
    // per new plane; at most DISCOVERY_ROUNDS run per frame and an unfinished
    // tile stays pending, finishing over the next frames.
    for (var round = 0u; round < pixels; round++) {
        if round >= DISCOVERY_ROUNDS {
            if thread == 0u { atomicStore(&pool.tiles[id.x].pending, 1u); }
            break;
        }
        if thread == 0u { atomicStore(&pool_winner, 0xffffffffu); }
        workgroupBarrier();
        for (var offset = thread; offset < pixels; offset += 32u) {
            let pixel = vec2<i32>(begin + vec2(offset % extent.x, offset / extent.x));
            let receiver = prepass_receiver(pixel);
            if receiver.face >= 6u || pool_covers(id.x, receiver) { continue; }
            if view.history.w != 0u || receiver_exists(receiver) { atomicMin(&pool_winner, offset); }
            else { atomicStore(&pool.tiles[id.x].pending, 1u); }
        }
        workgroupBarrier();
        let winner = atomicLoad(&pool_winner);
        if winner == 0xffffffffu { break; }
        if thread == 0u {
            let pixel = vec2<i32>(begin + vec2(winner % extent.x, winner / extent.x));
            pool_done = u32(!allocate_receiver(id.x, pixel, prepass_receiver(pixel)));
        }
        storageBarrier();
        if workgroupUniformLoad(&pool_done) != 0u { break; }
    }
}
