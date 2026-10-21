// Voxel traversal shared by world-space transport (voxel_trace.wgsl) and
// the terrain material (voxel_transport.wgsl), included textually so near and
// far lighting traverse the same geometry with the same code. Each includer
// declares `params` (with `counts.z` the directory mask and `minimum`/`maximum`
// the region bounds), `directory`, `regions` and `cells`.

struct RayResult {
    voxel: vec3<i32>,
    material: u32,
    position: vec3<f32>,
    status: u32, // 0 escaped, 1 hit, 2 unknown, 3 cut at the diffuse reach
    normal: vec3<f32>,
}

fn pcg(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn hash_voxel(voxel: vec3<i32>) -> u32 {
    return pcg(pcg(pcg(bitcast<u32>(voxel.x)) ^ bitcast<u32>(voxel.y)) ^ bitcast<u32>(voxel.z));
}
fn lookup_region(coordinate: vec3<i32>) -> u32 {
    var slot = hash_voxel(coordinate) & params.counts.z;
    for (var attempt = 0u; attempt < 32u; attempt += 1u) {
        let entry = directory[slot];
        if entry.w == -1 { return 0xffffffffu; }
        if all(entry.xyz == coordinate) { return u32(entry.w); }
        slot = (slot + 1u) & params.counts.z;
    }
    return 0xffffffffu;
}
fn in_bounds(region: vec3<i32>) -> bool {
    return all(region >= params.minimum.xyz) && all(region <= params.maximum.xyz);
}

// Point lookup for raster receiver identity and emission. 256 denotes air or
// absent geometry; transport rays retain their separate unknown-space policy.
fn voxel_material(voxel: vec3<i32>) -> u32 {
    let region = vec3<i32>(floor(vec3<f32>(voxel) / 16.0));
    if !in_bounds(region) { return 256u; }
    let index = lookup_region(region);
    if index == 0xffffffffu { return 256u; }
    let metadata = regions[index];
    if metadata.x == 1u { return 256u; }
    if metadata.x != 3u { return metadata.y; }
    let local = vec3<u32>(voxel & vec3<i32>(15));
    let brick = local / 4u;
    let word = cells[metadata.y + (brick.z * 4u + brick.y) * 4u + brick.x];
    if (word & 3u) == 0u { return 256u; }
    if (word & 3u) == 1u { return word >> 2u; }
    let inner = local & vec3<u32>(3u);
    let cell_index = (inner.z * 4u + inner.y) * 4u + inner.x;
    let cell = (cells[(word >> 2u) + cell_index / 2u] >> ((cell_index & 1u) * 16u)) & 65535u;
    return select(256u, cell >> 1u, (cell & 1u) != 0u);
}
fn unpack_bytes(value: u32) -> vec3<i32> {
    return vec3<i32>(i32(value & 255u), i32((value >> 8u) & 255u), i32((value >> 16u) & 255u));
}
fn negative_extents(bits: u32) -> vec3<i32> {
    return vec3<i32>(i32(bits & 3u), i32((bits >> 4u) & 3u), i32((bits >> 8u) & 3u));
}
fn positive_extents(bits: u32) -> vec3<i32> {
    return vec3<i32>(i32((bits >> 2u) & 3u), i32((bits >> 6u) & 3u), i32((bits >> 10u) & 3u));
}
fn advance_box(position: vec3<f32>, direction: vec3<f32>, cell: vec3<i32>, negative: vec3<i32>, positive: vec3<i32>, size: f32) -> vec4<f32> {
    let minimum = vec3<f32>(cell - negative) * size;
    let maximum = vec3<f32>(cell + positive + vec3(1)) * size;
    let moving = abs(direction) > vec3(1e-8);
    let plane = select(minimum, maximum, direction > vec3(0.0));
    let raw = (plane - position) / select(vec3(1.0), direction, moving);
    let travel = select(vec3(1e30), raw, moving & (raw >= vec3(-0.001)) & (raw < vec3(1e30)));
    // Match the scalar XYZ ordering, including a tiny negative crossing after
    // roundoff: each accepted crossing is clamped before considering the next.
    let choose_y = travel.y < max(travel.x, 0.0);
    let distance_xy = max(select(travel.x, travel.y, choose_y), 0.0);
    let choose_z = travel.z < distance_xy;
    let distance = max(select(distance_xy, travel.z, choose_z), 0.0);
    let axis_xy = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), choose_y);
    let axis = select(axis_xy, vec3(0.0, 0.0, 1.0), choose_z);
    let normal = select(vec3(0.0), axis * select(vec3(1.0), vec3(-1.0), direction > vec3(0.0)), distance < 1e30);
    return vec4(normal, max(distance + 0.001, 0.001));
}
fn geometry_ahead(start: vec3<f32>, direction: vec3<f32>, max_distance: f32) -> bool {
    var position = start;
    var remaining = max_distance;
    for (var step = 0u; step < 2048u; step += 1u) {
        let region = vec3<i32>(floor(position / 16.0));
        if remaining < 0.0 || !in_bounds(region) { return false; }
        let index = lookup_region(region);
        if index != 0xffffffffu && regions[index].x != 1u { return true; }
        let travel = advance_box(position, direction, region, vec3(0), vec3(0), 16.0).w;
        position += direction * travel;
        remaining -= travel;
    }
    return true;
}
fn trace_domain(origin: vec3<f32>, direction: vec3<f32>, max_distance: f32, visible_geometry: bool) -> RayResult {
    var result: RayResult;
    var near = -1e30;
    var far = 1e30;
    var normal = vec3(0.0);
    let minimum = vec3<f32>(params.minimum.xyz) * 16.0;
    let maximum = vec3<f32>(params.maximum.xyz + vec3(1)) * 16.0;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        if abs(direction[axis]) <= 1e-8 {
            if origin[axis] < minimum[axis] || origin[axis] > maximum[axis] { return result; }
            continue;
        }
        let first = (minimum[axis] - origin[axis]) / direction[axis];
        let second = (maximum[axis] - origin[axis]) / direction[axis];
        if min(first, second) > near {
            near = min(first, second);
            normal = vec3(0.0);
            normal[axis] = select(1.0, -1.0, first < second);
        }
        far = min(far, max(first, second));
    }
    if far < max(near, 0.0) { return result; }
    var distance = max(near, 0.0) + 0.001;
    var position = origin + direction * distance;
    var last_region = vec3<i32>(0x7fffffff);
    var metadata = vec4<u32>(0u);
    for (var step = 0u; step < 2048u; step += 1u) {
        if distance > max_distance { return result; }
        let voxel = vec3<i32>(floor(position));
        let region = voxel >> vec3<u32>(4u);
        if !in_bounds(region) { return result; }
        if any(last_region != region) {
            let index = lookup_region(region);
            if index == 0xffffffffu {
                if visible_geometry {
                    let advance = advance_box(position, direction, region, vec3(0), vec3(0), 16.0);
                    distance += advance.w;
                    position += direction * advance.w;
                    normal = advance.xyz;
                    continue;
                }
                result.status = select(0u, 2u, geometry_ahead(position, direction, max_distance - distance));
                return result;
            }
            metadata = regions[index];
            last_region = region;
        }
        var advance = vec4(0.0);
        if metadata.x == 1u {
            advance = advance_box(position, direction, region, unpack_bytes(metadata.z), unpack_bytes(metadata.w), 16.0);
        } else {
            var material = metadata.y;
            if metadata.x == 3u {
                let local = vec3<u32>(voxel & vec3<i32>(15));
                let brick = local / 4u;
                let brick_data = cells[metadata.y + (brick.z * 4u + brick.y) * 4u + brick.x];
                let brick_state = brick_data & 3u;
                if brick_state == 0u {
                    let extents = brick_data >> 2u;
                    advance = advance_box(position, direction, region * 4 + vec3<i32>(brick), negative_extents(extents), positive_extents(extents), 4.0);
                    material = 256u;
                } else if brick_state == 1u {
                    material = brick_data >> 2u;
                } else {
                    let inner = local & vec3<u32>(3u);
                    let index = (inner.z * 4u + inner.y) * 4u + inner.x;
                    let word = cells[(brick_data >> 2u) + index / 2u];
                    let cell = (word >> ((index & 1u) * 16u)) & 65535u;
                    material = select(256u, cell >> 1u, (cell & 1u) != 0u);
                    if material == 256u {
                        let extents = cell >> 1u;
                        advance = advance_box(position, direction, voxel, negative_extents(extents), positive_extents(extents), 1.0);
                    }
                }
            }
            if material != 256u {
                result.voxel = voxel;
                result.material = material;
                result.position = position;
                result.normal = normal;
                result.status = 1u;
                return result;
            }
        }
        distance += advance.w;
        position += direction * advance.w;
        normal = advance.xyz;
    }
    result.status = 2u;
    return result;
}
