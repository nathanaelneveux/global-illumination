// EXPERIMENT: per-pixel lighting for faces smaller than `pixel_tune.x`
// pixels, in NAADF's manner. No face identity: each cell (`pixel_grid.z`
// pixels per side) traces one path per frame from one of its pixels (a
// different one each frame), keeps a screen-space history reprojected by
// world position (up to `pixel_tune.y` samples), and writes a packed cell
// the material filters. Cost follows the screen. Three dispatches:
// classify (reproject every cell, queue those to trace), size the queue's
// indirect dispatch, trace the queue densely.

struct PixelHistory {
    mean: vec4<f32>, // RGB, sample count
    surface: vec3<f32>,
    face: u32, // 6: none
}
struct PixelOutput {
    camera: vec4<f32>, // xyz
    axis: vec4<f32>, // view forward, pixels per unit at depth 1
    grid: vec4<u32>, // cells x, y, pixels per cell side, enabled
    origin: vec4<u32>, // viewport origin, threshold px (bits), reserved
    cells: array<vec4<u32>>, // pack2x16float(rg), pack2x16float(b, count), face, plane (bits)
}
@group(2) @binding(19) var<storage, read> previous_pixels: array<PixelHistory>;
@group(2) @binding(20) var<storage, read_write> current_pixels: array<PixelHistory>;
@group(2) @binding(21) var<storage, read_write> pixel_stats: array<atomic<u32>, 4>;
@group(2) @binding(22) var<storage, read_write> pixel_output: PixelOutput;
// Cells to trace this frame: a count, then cell indices (dense, so the
// trace dispatch runs full waves however scattered the owned cells are).
@group(2) @binding(23) var<storage, read_write> pixel_queue: array<atomic<u32>>;
@group(2) @binding(24) var<storage, read_write> pixel_args: array<u32, 3>;

/// The same test the material makes: a full-resolution voxel at this
/// voxel's view depth projects under the threshold's pixels across. A pure
/// distance band, whatever the LoD stride or viewing angle.
fn pixel_owned(voxel: vec3<i32>) -> bool {
    if view.pixel_grid.w == 0u { return false; }
    let depth = dot(vec3<f32>(voxel) + 0.5 - view.camera.xyz, view.pixel_axis.xyz);
    return depth > 0.0 && view.pixel_axis.w < view.pixel_tune.x * depth;
}

fn write_pixel(index: u32, value: PixelHistory) {
    current_pixels[index] = value;
    let axis = min(value.face, 5u) / 2u;
    pixel_output.cells[index] = vec4(pack2x16float(value.mean.rg), pack2x16float(vec2(value.mean.b, value.mean.w)),
        value.face, bitcast<u32>(value.surface[axis]));
}

@compute @workgroup_size(64)
fn classify_pixels(@builtin(global_invocation_id) id: vec3<u32>) {
    let grid = view.pixel_grid;
    if id.x == 0u {
        pixel_output.camera = vec4(view.camera.xyz, 0.0);
        pixel_output.axis = view.pixel_axis;
        pixel_output.grid = grid;
        pixel_output.origin = vec4(view.viewport.xy, bitcast<u32>(view.pixel_tune.x), 0u);
    }
    if id.x >= grid.x * grid.y { return; }
    var none: PixelHistory;
    none.face = 6u;
    let size = grid.z;
    let cell = vec2<u32>(id.x % grid.x, id.x / grid.x);
    // One pixel of the cell per frame, cycling through all of them.
    let k = (view.history.z * 3u + (cell.x ^ cell.y)) % (size * size);
    let local = vec2<u32>(k % size, k / size);
    let pixel = vec2<i32>(view.viewport.xy + cell * size + local);
    if any(vec2<u32>(pixel) >= view.viewport.xy + view.viewport.zw) { write_pixel(id.x, none); return; }
    let z = textureLoad(depth, pixel, 0);
    if z <= 0.0 { write_pixel(id.x, none); return; }
    let face = prepass_face(pixel);
    if face >= 6u { write_pixel(id.x, none); return; }
    let normal = face_normal(face);
    var surface = prepass_position(pixel, z);
    surface[face / 2u] = round(surface[face / 2u]);
    let voxel = vec3<i32>(floor(surface - normal * 0.01));
    if !pixel_owned(voxel) { write_pixel(id.x, none); return; }
    atomicAdd(&pixel_stats[0], 1u);

    // Reproject: bilinear over the previous cells around this point's old
    // screen position, keeping the same face orientation within four blocks
    // along its normal and two cells' world footprint across (NAADF's
    // depth/normal test: a jittered sample of sub-pixel terraces lands on a
    // neighbouring plane most frames).
    var mean = vec3(0.0);
    var count = 0.0;
    if view.pixel_tune.w != 0.0 {
        let clip = view.previous_clip_from_world * vec4(surface, 1.0);
        if clip.w > 0.0 {
            let uv = clip.xy / clip.w * vec2(0.5, -0.5) + 0.5;
            let p = uv * vec2<f32>(view.viewport.zw) / f32(size) - 0.5;
            let base = vec2<i32>(floor(p));
            let f = p - floor(p);
            let depth = dot(surface - view.camera.xyz, view.pixel_axis.xyz);
            let reach = 2.0 * f32(size) * depth / view.pixel_axis.w;
            let tolerance = 4.0 * f32(voxel_stride(voxel)) + 0.01;
            var weight = 0.0;
            var sum = vec3(0.0);
            var samples = 0.0;
            for (var y = 0; y <= 1; y++) {
                for (var x = 0; x <= 1; x++) {
                    let q = base + vec2(x, y);
                    if any(q < vec2(0)) || any(q >= vec2<i32>(grid.xy)) { continue; }
                    let old = previous_pixels[u32(q.y) * grid.x + u32(q.x)];
                    if old.face != face || old.mean.w == 0.0 { continue; }
                    let delta = old.surface - surface;
                    if abs(dot(delta, normal)) > tolerance || dot(delta, delta) > reach * reach { continue; }
                    let w = select(1.0 - f.x, f.x, x == 1) * select(1.0 - f.y, f.y, y == 1) + 1e-4;
                    sum += old.mean.rgb * w;
                    samples += old.mean.w * w;
                    weight += w;
                }
            }
            if weight > 0.0 {
                mean = sum / weight;
                count = min(samples / weight, view.pixel_tune.y);
                atomicAdd(&pixel_stats[3], 1u);
            }
        }
    }
    atomicAdd(&pixel_stats[2], u32(count));
    write_pixel(id.x, PixelHistory(vec4(mean, count), surface, face));
    let rate = max(u32(view.pixel_tune.z), 1u);
    if count < 4.0 || (id.x + view.history.z) % rate == 0u {
        let slot = atomicAdd(&pixel_queue[0], 1u);
        atomicStore(&pixel_queue[1u + slot], id.x);
    }
}

@compute @workgroup_size(1)
fn size_pixel_queue() {
    pixel_args = array(atomicLoad(&pixel_queue[0]) / 64u + 1u, 1u, 1u);
}

@compute @workgroup_size(64)
fn trace_pixel_queue(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= atomicLoad(&pixel_queue[0]) { return; }
    let index = atomicLoad(&pixel_queue[1u + id.x]);
    var cell = current_pixels[index];
    atomicAdd(&pixel_stats[1], 1u);
    let sample = trace_receiver(cell.surface, face_normal(cell.face), pcg(index * 9781u + 0x9e3779b9u), view.history.z);
    if sample.known {
        cell.mean.w = min(cell.mean.w + 1.0, view.pixel_tune.y);
        cell.mean = vec4(mix(cell.mean.rgb, sample.energy.value.rgb, 1.0 / cell.mean.w), cell.mean.w);
        write_pixel(index, cell);
    }
}
