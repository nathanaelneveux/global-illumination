// Surface-aware downsampling. Four neighbouring screen cells keep their normal
// centre samples, but redundant coplanar samples make room for thin surfaces in
// their shared footprint. The receiver/storage/ray budgets do not grow.
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

// Coverage needs only orientation and plane, not a complete receiver or a
// transport lookup. Cache four compact keys per round in invocation registers.
fn receiver_plane(receiver: ViewReceiver) -> vec2<i32> {
    if receiver.face >= 6u { return vec2(-1); }
    return vec2(i32(receiver.face), receiver.voxel[receiver.face / 2u]);
}

fn prepass_plane(pixel: vec2<i32>) -> vec2<i32> {
    let z = textureLoad(depth, pixel, 0);
    if z <= 0.0 { return vec2(-1); }
    let face = prepass_face(pixel);
    if face >= 6u { return vec2(-1); }
    let position = prepass_position(pixel, z);
    if view.camera.w > 0.0 && distance(position, view.camera.xyz) < view.camera.w { return vec2(-1); }
    let plane = i32(floor(round(position[face / 2u]) - select(-0.01, 0.01, (face & 1u) == 0u)));
    return vec2(i32(face), plane);
}

fn prepass_receiver(pixel: vec2<i32>) -> ViewReceiver {
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
    return receiver;
}

fn same_plane(a: ViewReceiver, b: ViewReceiver) -> bool {
    if a.face >= 6u || a.face != b.face { return false; }
    return a.voxel[a.face / 2u] == b.voxel[b.face / 2u];
}

var<workgroup> discovery_chosen: array<ViewReceiver, 4>;
var<workgroup> discovery_centers: array<vec2<i32>, 4>;
var<workgroup> discovery_winner: atomic<u32>;
var<workgroup> discovery_hash_xor: atomic<u32>;
var<workgroup> discovery_hash_sum: atomic<u32>;
var<workgroup> discovery_pending: atomic<u32>;
var<workgroup> discovery_reuse: u32;

fn receiver_exists(receiver: ViewReceiver) -> bool {
    if receiver.face >= 6u { return false; }
    let voxel = vec3<i32>(receiver.voxel.x, receiver.voxel.y, receiver.voxel.z);
    return voxel_material(receiver.voxel) != 256u
        && voxel_material(voxel + vec3<i32>(face_normal(receiver.face))) == 256u;
}

fn publish_discovery(index: u32, receiver: ViewReceiver) {
    var estimate: ViewEstimate;
    if receiver.face < 6u { estimate.identity.y = voxel_material(receiver.voxel); }
    receivers[index] = receiver;
    estimates[index] = estimate;
}

@compute @workgroup_size(32)
fn discover_receivers(@builtin(workgroup_id) id: vec3<u32>, @builtin(local_invocation_index) thread: u32) {
    let tiles = (view.grid.xy + 1u) / 2u;
    if id.x >= tiles.x * tiles.y { return; }
    let cell = vec2(id.x % tiles.x, id.x / tiles.x) * 2u;
    let begin = view.viewport.xy + cell * view.viewport.zw / view.grid.xy;
    let end = view.viewport.xy + min(cell + 2u, view.grid.xy) * view.viewport.zw / view.grid.xy;
    let extent = end - begin;
    let pixels = extent.x * extent.y;

    // Reuse selection, never stale lighting. Every pixel still participates in
    // a compact depth/normal fingerprint; changed footprints scan immediately.
    // Moving cameras use the full scan without this additional fingerprint pass.
    if view.discovery.x != 0u {
        if thread == 0u {
            atomicStore(&discovery_hash_xor, 0u);
            atomicStore(&discovery_hash_sum, 0u);
            atomicStore(&discovery_pending, 0u);
        }
        workgroupBarrier();
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
        atomicXor(&discovery_hash_xor, xor);
        atomicAdd(&discovery_hash_sum, sum);
        workgroupBarrier();
        if thread == 0u {
            let cached = discovery_tiles[id.x];
            discovery_reuse = u32(cached.z != 0u
                && cached.x == atomicLoad(&discovery_hash_xor)
                && cached.y == atomicLoad(&discovery_hash_sum)
                && (view.history.z + id.x) % 64u != 0u);
        }
        if workgroupUniformLoad(&discovery_reuse) != 0u {
            if thread < 4u {
                let q = cell + vec2(thread & 1u, thread >> 1u);
                if all(q < view.grid.xy) {
                    let receiver = previous_receivers[q.y * view.grid.x + q.x];
                    if receiver.face < 6u && !receiver_exists(receiver) {
                        atomicOr(&discovery_pending, 1u);
                    }
                }
            }
            workgroupBarrier();
            if thread == 0u { discovery_reuse = u32(atomicLoad(&discovery_pending) == 0u); }
            if workgroupUniformLoad(&discovery_reuse) != 0u {
                if thread < 4u {
                    let q = cell + vec2(thread & 1u, thread >> 1u);
                    if all(q < view.grid.xy) {
                        let index = q.y * view.grid.x + q.x;
                        publish_discovery(index, previous_receivers[index]);
                    }
                }
                return;
            }
        }
    }

    if thread == 0u { atomicStore(&discovery_pending, 0u); }
    workgroupBarrier();
    if thread < 4u {
        let q = cell + vec2(thread & 1u, thread >> 1u);
        discovery_chosen[thread].face = 0xffffffffu;
        if all(q < view.grid.xy) {
            let uv = (vec2<f32>(q) + 0.5) / vec2<f32>(view.grid.xy);
            discovery_centers[thread] = vec2<i32>(view.viewport.xy + min(vec2<u32>(uv * vec2<f32>(view.viewport.zw)), view.viewport.zw - 1u));
            let receiver = prepass_receiver(discovery_centers[thread]);
            if receiver_exists(receiver) { discovery_chosen[thread] = receiver; }
            else if receiver.face < 6u { atomicOr(&discovery_pending, 1u); }
        }
    }
    workgroupBarrier();
    // Parallel prepass classification, with one deterministic replacement per
    // round. Coordinated selection cannot evict every copy of the background
    // for the same thin plane. Flat tiles finish after one scan.
    for (var round = 0u; round < 4u; round++) {
        if thread == 0u { atomicStore(&discovery_winner, 0xffffffffu); }
        workgroupBarrier();
        let p0 = receiver_plane(discovery_chosen[0]);
        let p1 = receiver_plane(discovery_chosen[1]);
        let p2 = receiver_plane(discovery_chosen[2]);
        let p3 = receiver_plane(discovery_chosen[3]);
        for (var offset = thread; offset < pixels; offset += 32u) {
            let pixel = vec2<i32>(begin + vec2(offset % extent.x, offset / extent.x));
            let plane = prepass_plane(pixel);
            if plane.x < 0 || all(p0 == plane) || all(p1 == plane) || all(p2 == plane) || all(p3 == plane) { continue; }
            let orientation = any(vec4(p0.x, p1.x, p2.x, p3.x) == vec4(plane.x));
            var replace = 4u;
            var distance = 0x7fffffffu;
            for (var lane = 0u; lane < 4u; lane++) {
                if any(cell + vec2(lane & 1u, lane >> 1u) >= view.grid.xy) { continue; }
                var redundant = discovery_chosen[lane].face >= 6u;
                for (var other = 0u; other < 4u; other++) {
                    if other == lane { continue; }
                    redundant = redundant || same_plane(discovery_chosen[lane], discovery_chosen[other])
                        || (!orientation && discovery_chosen[lane].face < 6u
                            && discovery_chosen[lane].face == discovery_chosen[other].face);
                }
                let delta = pixel - discovery_centers[lane];
                let d2 = u32(dot(delta, delta));
                if redundant && d2 < distance { replace = lane; distance = d2; }
            }
            // A raster/transport geometry lag must not displace a valid owner.
            if replace < 4u {
                if receiver_exists(prepass_receiver(pixel)) {
                    // Lexicographic distance/slot/pixel ordering, independent of
                    // thread arrival order. The rank fits u32 for 16K viewports.
                    atomicMin(&discovery_winner, (distance * 4u + replace) * pixels + offset);
                } else {
                    // Geometry uploads can catch up without changing the
                    // prepass. Do not memoize a selection that excluded them.
                    atomicOr(&discovery_pending, 1u);
                }
            }
        }
        workgroupBarrier();
        let winner = atomicLoad(&discovery_winner);
        if winner == 0xffffffffu { break; }
        if thread == 0u {
            let offset = winner % pixels;
            let lane = (winner / pixels) % 4u;
            discovery_chosen[lane] = prepass_receiver(vec2<i32>(begin + vec2(offset % extent.x, offset / extent.x)));
        }
        workgroupBarrier();
    }
    if thread == 0u {
        var cached = vec4(0u);
        if view.discovery.x != 0u {
            cached = vec4(atomicLoad(&discovery_hash_xor), atomicLoad(&discovery_hash_sum),
                u32(atomicLoad(&discovery_pending) == 0u), view.history.z + 1u);
        }
        discovery_tiles[id.x] = cached;
    }
    if thread < 4u {
        let q = cell + vec2(thread & 1u, thread >> 1u);
        if any(q >= view.grid.xy) { return; }
        let index = q.y * view.grid.x + q.x;
        let receiver = discovery_chosen[thread];
        publish_discovery(index, receiver);
    }
}
