// The far tier: per-pixel lighting for faces small on screen
// (`FarTierSettings`), in NAADF's manner. No face identity: each cell
// (`pixel_grid.z` pixels per side) samples one of its pixels per frame (a
// different one each frame), keeps a screen-space history reprojected by
// world position (up to `pixel_tune.y` samples), traces a path when that
// history is short or one frame in `pixel_tune.z`, and writes a packed cell
// the material filters. Cost follows the screen. Three dispatches:
// classify (reproject every cell, queue those to trace), size the queue's
// indirect dispatch, trace the queue densely.

struct PixelHistory {
    mean: vec4<f32>, // RGB, sample count
    surface: vec3<f32>,
    // Face (bits 0-2) and the frame that wrote it (bits 3-31, `pixel_stamp`).
    // Only far cells are written; a reader trusts a cell only with the
    // stamp it expects, so the near majority of cells costs no writes.
    face: u32,
    // This cell's next path index (its own attempt sequence; the history
    // count is capped and the trace schedule follows the frame, so neither
    // can index the quasi-random sampler). Follows the world point through
    // reprojection.
    sequence: u32,
}
struct PixelOutput {
    camera: vec4<f32>, // xyz
    axis: vec4<f32>, // view forward, pixels per unit at depth 1
    grid: vec4<u32>, // cells x, y, pixels per cell side, enabled
    origin: vec4<u32>, // viewport origin, threshold px (bits), this frame's stamp
    cells: array<vec4<u32>>, // pack2x16float(rg), pack2x16float(b, count), face and stamp, plane (bits)
}
@group(2) @binding(19) var<storage, read> previous_pixels: array<PixelHistory>;
@group(2) @binding(20) var<storage, read_write> current_pixels: array<PixelHistory>;
// Far-tier counters (Rust: runtime::pixel_words), read back with the
// timing receipt.
const PIXEL_OWNED: u32 = 0u;
const PIXEL_TRACED: u32 = 1u;
const PIXEL_HISTORY: u32 = 2u;
const PIXEL_REPROJECTED: u32 = 3u;
const PIXEL_SEEDED: u32 = 4u;
const PIXEL_RECEIVERS_SEEDED: u32 = 5u;
@group(2) @binding(21) var<storage, read_write> pixel_stats: array<atomic<u32>, 8>;
@group(2) @binding(22) var<storage, read_write> pixel_output: PixelOutput;
// Cells to trace this frame: a count, then cell indices (dense, so the
// trace dispatch runs full waves however scattered the owned cells are).
@group(2) @binding(23) var<storage, read_write> pixel_queue: array<atomic<u32>>;
@group(2) @binding(24) var<storage, read_write> pixel_args: array<u32, 3>;

/// The same test the material makes (`FarTierMeasure`, mode in grid.w bits
/// 1-2), by distance from the camera, not view depth, so turning does not
/// move the boundary: 0, a full-resolution voxel at this distance under the
/// threshold's pixels across; 1, its rough covered area (side squared times the cosine
/// to the view ray) under the threshold's pixels; 2, the same for the
/// face's LoD block.
fn pixel_owned(voxel: vec3<i32>, face: u32) -> bool {
    let grid = view.pixel_grid.w;
    if (grid & 1u) == 0u || face >= 6u { return false; }
    let mode = (grid >> 1u) & 3u;
    let s = select(1, voxel_stride(voxel), mode == 2u);
    let ray = vec3<f32>(voxel & vec3(-s)) + 0.5 * f32(s) - view.camera.xyz;
    if dot(ray, view.pixel_axis.xyz) <= 0.0 { return false; }
    let side = f32(s) * view.pixel_axis.w / length(ray);
    if mode == 0u { return side < view.pixel_tune.x; }
    return side * side * abs(dot(face_normal(face), normalize(ray))) < view.pixel_tune.x;
}

fn pixel_stamp(frame: u32) -> u32 { return frame & 0x1fffffffu; }

/// Custom direct RGB (standalone `VoxelGiEmitter` sources, which have no
/// Bevy light to draw it): a per-frame presentation term like receivers'
/// (`publish_view`), never part of the accumulated indirect history.
fn custom_direct(value: PixelHistory) -> vec3<f32> {
    if params.transport.y <= 0.0 { return vec3(0.0); }
    return direct_at(value.surface, face_normal(value.face & 7u)).custom_visible;
}

fn write_output(index: u32, value: PixelHistory) {
    let axis = (value.face & 7u) / 2u;
    let shown = value.mean.rgb + custom_direct(value);
    pixel_output.cells[index] = vec4(pack2x16float(shown.rg), pack2x16float(vec2(shown.b, value.mean.w)),
        value.face, bitcast<u32>(value.surface[axis]));
}

/// The cell at this world point last frame (bilinear over the previous
/// cells around its old screen position), keeping the same face orientation
/// within four blocks along its normal and two cells' world footprint across
/// (NAADF's depth/normal test: a jittered sample of sub-pixel terraces lands
/// on a neighbouring plane most frames). `mean.w` is the sample count, 0 if
/// none; `sequence` is the best-weighted cell's.
struct PreviousCell { mean: vec4<f32>, sequence: u32 }
fn previous_cell(surface: vec3<f32>, face: u32, voxel: vec3<i32>) -> PreviousCell {
    var found: PreviousCell;
    let previous = bitcast<u32>(view.pixel_tune.w);
    if previous == 0xffffffffu { return found; }
    let clip = view.previous_clip_from_world * vec4(surface, 1.0);
    if clip.w <= 0.0 { return found; }
    let grid = view.pixel_grid;
    let size = grid.z;
    let normal = face_normal(face);
    let uv = clip.xy / clip.w * vec2(0.5, -0.5) + 0.5;
    let p = uv * vec2<f32>(view.viewport.zw) / f32(size) - 0.5;
    let base = vec2<i32>(floor(p));
    let f = p - floor(p);
    let reach = 2.0 * f32(size) * length(surface - view.camera.xyz) / view.pixel_axis.w;
    let tolerance = 4.0 * f32(voxel_stride(voxel)) + 0.01;
    let expected = face | (pixel_stamp(previous) << 3u);
    var weight = 0.0;
    var sum = vec3(0.0);
    var samples = 0.0;
    var best = 0.0;
    for (var y = 0; y <= 1; y++) {
        for (var x = 0; x <= 1; x++) {
            let q = base + vec2(x, y);
            if any(q < vec2(0)) || any(q >= vec2<i32>(grid.xy)) { continue; }
            let old = previous_pixels[u32(q.y) * grid.x + u32(q.x)];
            if old.face != expected || old.mean.w == 0.0 { continue; }
            let delta = old.surface - surface;
            if abs(dot(delta, normal)) > tolerance || dot(delta, delta) > reach * reach { continue; }
            let w = select(1.0 - f.x, f.x, x == 1) * select(1.0 - f.y, f.y, y == 1) + 1e-4;
            sum += old.mean.rgb * w;
            samples += old.mean.w * w;
            weight += w;
            if w > best { best = w; found.sequence = old.sequence; }
        }
    }
    if weight > 0.0 { found.mean = vec4(sum / weight, samples / weight); }
    return found;
}

@compute @workgroup_size(64)
fn classify_pixels(@builtin(global_invocation_id) id: vec3<u32>) {
    let grid = view.pixel_grid;
    let stamp = pixel_stamp(view.history.z);
    if id.x == 0u {
        pixel_output.camera = vec4(view.camera.xyz, 0.0);
        pixel_output.axis = view.pixel_axis;
        pixel_output.grid = grid;
        pixel_output.origin = vec4(view.viewport.xy, bitcast<u32>(view.pixel_tune.x), stamp);
    }
    if id.x >= grid.x * grid.y { return; }
    let size = grid.z;
    let cell = vec2<u32>(id.x % grid.x, id.x / grid.x);
    // One pixel of the cell per frame, cycling through all of them: 5 is
    // coprime with every supported cell area (1, 4, 9, 16).
    let k = (view.history.z * 5u + (cell.x ^ cell.y)) % (size * size);
    let local = vec2<u32>(k % size, k / size);
    let pixel = vec2<i32>(view.viewport.xy + cell * size + local);
    if any(vec2<u32>(pixel) >= view.viewport.xy + view.viewport.zw) { return; }
    let z = textureLoad(depth, pixel, 0);
    if z <= 0.0 { return; }
    var surface = prepass_position(pixel, z);
    // The side test needs only distance: most near cells stop here, before
    // the normal load (a margin covers the voxel-centre rounding below).
    if ((grid.w >> 1u) & 3u) == 0u
        && view.pixel_axis.w > 1.05 * view.pixel_tune.x * length(surface - view.camera.xyz) { return; }
    let face = prepass_face(pixel);
    if face >= 6u { return; }
    let normal = face_normal(face);
    surface[face / 2u] = round(surface[face / 2u]);
    let voxel = vec3<i32>(floor(surface - normal * 0.01));
    if !pixel_owned(voxel, face) { return; }
    atomicAdd(&pixel_stats[PIXEL_OWNED], 1u);
    let found = previous_cell(surface, face, voxel);
    var history = found.mean;
    var sequence = found.sequence;
    if history.w > 0.0 {
        atomicAdd(&pixel_stats[PIXEL_REPROJECTED], 1u);
    } else {
        // Handoff: a face crossing into the far band starts from its view
        // receiver's history (same estimator: emitter plus continuation)
        // rather than from nothing. Its count is capped like any history.
        var key: ViewReceiver;
        key.face = face;
        key.voxel = voxel;
        key.surface = vec3<f32>(voxel) + 0.5 + normal * 0.5;
        let owner = previous_owner(block_receiver(key));
        if owner != 0xffffffffu && previous_estimates[owner].mean.w > 0.0 {
            history = previous_estimates[owner].mean;
            sequence = previous_estimates[owner].identity.x;
            atomicAdd(&pixel_stats[PIXEL_SEEDED], 1u);
        }
    }
    let count = min(history.w, view.pixel_tune.y);
    atomicAdd(&pixel_stats[PIXEL_HISTORY], u32(count));
    let value = PixelHistory(vec4(history.rgb, count), surface, face | (stamp << 3u), sequence);
    current_pixels[id.x] = value;
    let rate = max(u32(view.pixel_tune.z), 1u);
    if count < 4.0 || (id.x + view.history.z) % rate == 0u {
        // The trace writes this cell's output.
        let slot = atomicAdd(&pixel_queue[0], 1u);
        atomicStore(&pixel_queue[1u + slot], id.x);
    } else {
        write_output(id.x, value);
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
    atomicAdd(&pixel_stats[PIXEL_TRACED], 1u);
    let sample = trace_receiver(cell.surface, face_normal(cell.face & 7u), pcg(index * 9781u + 0x9e3779b9u), cell.sequence);
    // Unknown is not black, but it still consumes its index.
    cell.sequence += 1u;
    if sample.known {
        cell.mean.w = min(cell.mean.w + 1.0, view.pixel_tune.y);
        cell.mean = vec4(mix(cell.mean.rgb, sample.energy.value.rgb, 1.0 / cell.mean.w), cell.mean.w);
    }
    current_pixels[index] = cell;
    write_output(index, cell);
}
