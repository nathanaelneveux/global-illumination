// Voxel GI presentation for host materials: what turns the view tier's
// output into light at a fragment. Import it from a material shader
// (`#import voxel_gi::presentation::{indirect_light, sunlight, ...}`); its
// globals are namespaced by the import, so they cannot collide with the
// host's. It binds the GI resources at 104 and 106-111 of the material bind
// group: hosts add them with `VoxelGiBindings` (an `ExtendedMaterial`
// extension) and keep those numbers free.
#define_import_path voxel_gi::presentation

struct GpuTransportSettings {
    enabled: u32,
    reconstruction: u32,
    intensity: f32,
    debug_irradiance: u32,
    voxel_sky: u32,
    sun_count: u32,
    sun_direction: array<vec4<f32>, 4>,
    sun_color: array<vec4<f32>, 4>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(104)
var<uniform> transport_settings: GpuTransportSettings;

// The GI worker's transport geometry, for sun visibility of uncovered faces.
struct TraversalParams {
    counts: vec4<u32>, // sun visibility distance (0 before the first scene), unused, directory mask, unused
    minimum: vec4<i32>,
    maximum: vec4<i32>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(106)
var<storage, read> directory: array<vec4<i32>>;
@group(#{MATERIAL_BIND_GROUP}) @binding(107)
var<storage, read> regions: array<vec4<u32>>;
@group(#{MATERIAL_BIND_GROUP}) @binding(108)
var<storage, read> cells: array<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(109)
var<uniform> params: TraversalParams;

struct FarPixel {
    surface: vec3<f32>, face: u32,
    irradiance: vec3<f32>, evidence: f32,
    sky: vec4<u32>,
}
struct FarOutput {
    viewport: vec4<u32>, shape: vec4<u32>, optical: array<vec4<f32>, 1024>,
    keys: array<u32, 262144>,
    heads: array<u32, 65536>, links: array<u32, 131072>,
    pixels: array<FarPixel>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(110)
var<storage, read> far_view: FarOutput;

// The far tier (voxel_view_pixels.wgsl). Faces small on screen by its
// measure (`pixel_owned`) have no receiver; its cells hold their per-pixel
// indirect light.
struct FarCells {
    camera: vec4<f32>,
    axis: vec4<f32>, // view forward, pixels per unit at depth 1
    grid: vec4<u32>, // cells x, y, pixels per cell side, enabled (bit 0) and measure (bits 1-2)
    origin: vec4<u32>, // viewport origin, threshold px (bits), this frame's stamp
    cells: array<vec4<u32>>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(111)
var<storage, read> far_cells: FarCells;

// #include voxel_traversal.wgsl

// The per-pixel tier's ownership test, exactly as it and discovery compute
// it (`pixel_owned` in voxel_view_pixels.wgsl; mode in grid.w bits 1-2).
fn pixel_owned(voxel: vec3<i32>, face: u32) -> bool {
    let grid = far_cells.grid.w;
    if (grid & 1u) == 0u || face >= 6u { return false; }
    let mode = (grid >> 1u) & 3u;
    let s = select(1, voxel_stride(voxel), mode == 2u);
    let ray = vec3<f32>(voxel & vec3(-s)) + 0.5 * f32(s) - far_cells.camera.xyz;
    if dot(ray, far_cells.axis.xyz) <= 0.0 { return false; }
    let side = f32(s) * far_cells.axis.w / length(ray);
    let threshold = bitcast<f32>(far_cells.origin.z);
    if mode == 0u { return side < threshold; }
    return side * side * abs(dot(vec3<f32>(FACE_NORMALS[face]), normalize(ray))) < threshold;
}

// The cells around this pixel on this face: coplanar ones, else parallel
// planes up to four blocks away (a distant ledge's thin side), weighted by
// sample count and distance in cells.
fn pixel_irradiance(pixel: vec2<f32>, position: vec3<f32>, normal: vec3<f32>, s: i32) -> vec4<f32> {
    let grid = far_cells.grid;
    let c = (pixel - vec2<f32>(far_cells.origin.xy)) / f32(grid.z) - 0.5;
    let centre = vec2<i32>(round(c));
    let face = face_index_of(normal);
    let plane = round(position[face / 2u]);
    var sum = vec3(0.0);
    var weight = 0.0;
    var parallel_sum = vec3(0.0);
    var parallel_weight = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = centre + vec2(x, y);
            if any(q < vec2(0)) || any(q >= vec2<i32>(grid.xy)) { continue; }
            let cell = far_cells.cells[u32(q.y) * grid.x + u32(q.x)];
            // Only cells written this frame (face and stamp) are current.
            if cell.z != (face | (far_cells.origin.w << 3u)) { continue; }
            let blue_count = unpack2x16float(cell.y);
            if blue_count.y <= 0.0 { continue; }
            let rgb = vec3(unpack2x16float(cell.x), blue_count.x);
            let offset = abs(bitcast<f32>(cell.w) - plane);
            let d = vec2<f32>(q) - c;
            let w = blue_count.y * exp(-dot(d, d));
            if offset <= 0.01 {
                sum += rgb * w;
                weight += w;
            } else if offset <= 4.0 * f32(s) + 0.01 {
                parallel_sum += rgb * w;
                parallel_weight += w;
            }
        }
    }
    if weight == 0.0 && parallel_weight > 0.0 { return vec4(parallel_sum / parallel_weight, 1.0); }
    return vec4(sum / max(weight, 1e-8), select(0.0, 1.0, weight > 0.0));
}

// Geometry-aware upsampling only. No value is reused across an unrelated
// face orientation or depth plane, even at foreground/background edges.
// `voxel` is the receiver key (a coarse block's minimum voxel) and `surface`
// its face centre, which also tells unit and block faces apart.
fn far_face_index(voxel: vec3<i32>, face: u32, surface: vec3<f32>) -> u32 {
    var slot = (hash_voxel(voxel) ^ pcg(face + 1u)) & far_view.shape.w;
    for (var step = 0u; step < 64u; step++) {
        let entry = far_view.keys[slot];
        if entry == 0u { break; }
        let value = far_view.pixels[entry - 1u];
        if value.face == face && all(value.surface == surface) { return entry - 1u; }
        slot = (slot + 1u) & far_view.shape.w;
    }
    return 0xffffffffu;
}

fn far_irradiance(pixel: vec2<f32>, position: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    if far_view.shape.z == 0u { return vec4(0.0); }
    let uv = (pixel - vec2<f32>(far_view.viewport.xy)) / vec2<f32>(far_view.viewport.zw);
    if any(uv < vec2(0.0)) || any(uv >= vec2(1.0)) { return vec4(0.0); }
    // Exact keys need the face's axis, not the interpolated mesh normal.
    let face = face_index_of(normal);
    let s = voxel_stride(vec3<i32>(floor(position - normal * 0.01)));
    let voxel = vec3<i32>(floor(position - normal * 0.01)) & vec3(-s);
    let surface = vec3<f32>(voxel) + f32(s) * (0.5 + 0.5 * vec3<f32>(FACE_NORMALS[face]));
    let exact = far_face_index(voxel, face, surface);
    if exact != 0xffffffffu {
        let value = far_view.pixels[exact];
        if value.evidence > 0.0 { return vec4(value.irradiance, 1.0); }
    }
    // Physical slots have no screen location. Search the four closest spatial
    // bins, including the containing bin, with unchanged world geometry gates.
    let tiles = (far_view.shape.xy + 1u) / 2u;
    let base = vec2<i32>(floor(uv * vec2<f32>(far_view.shape.xy) * 0.5 - 0.5));
    // Coplanar donors first. A thin face with none (a distant ledge's
    // one-pixel side) would otherwise read black; its indirect light then
    // borrows from parallel faces up to four voxels in front of or behind it.
    var sum = vec3(0.0);
    var weight = 0.0;
    var parallel_sum = vec3(0.0);
    var parallel_weight = 0.0;
    for (var y = 0; y <= 1; y++) {
        for (var x = 0; x <= 1; x++) {
            let q = base + vec2(x, y);
            if any(q < vec2(0)) || any(q >= vec2<i32>(tiles)) { continue; }
            var entry = far_view.heads[u32(q.y) * tiles.x + u32(q.x)];
            for (var visited = 0u; entry != 0u && visited < far_view.shape.x * far_view.shape.y; visited++) {
                let index = entry - 1u;
                entry = far_view.links[index];
                let donor = far_view.pixels[index];
                if donor.face >= 6u { continue; }
                if donor.evidence == 0.0 { continue; }
                if dot(vec3<f32>(FACE_NORMALS[donor.face]), normal) < 0.99 { continue; }
                // Radii in blocks: a coarse page's faces are `s` apart.
                let delta = (donor.surface - surface) / f32(s);
                let d2 = dot(delta, delta);
                if d2 > 256.0 { continue; }
                let offset = abs(dot(delta, normal));
                let w = 1.0 / (0.25 + d2);
                if offset <= 0.05 / f32(s) {
                    sum += donor.irradiance * w;
                    weight += w;
                } else if offset <= 4.05 {
                    parallel_sum += donor.irradiance * w;
                    parallel_weight += w;
                }
            }
        }
    }
    if weight == 0.0 && parallel_weight > 0.0 {
        return vec4(parallel_sum / parallel_weight, 1.0);
    }
    return vec4(sum / max(weight, 1e-8), select(0.0, 1.0, weight > 0.0));
}


const FACE_NORMALS = array<vec3<i32>, 6>(
    vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
    vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1),
);

fn corner_lit(word: u32, bit: u32) -> f32 {
    return select(0.0, 1.0, (word & bit) != 0u);
}

// These are ray measurements AT the virtual corners, not an average of nearby
// face-center rays. The four values naturally retain contact shadows at walls.
// Corners store which suns they see; each sun's current direction and color
// light the face here, so a moving sun brightens or dims every frame while
// cached shadows change only when its quantized direction does. Unknown
// corners contribute no direct sunlight yet; available corners remain usable.
fn resolve_sky_corners(corners: vec4<u32>, axis: vec3<f32>, position: vec2<f32>) -> vec4<f32> {
    let ready = select(0.0, 1.0, all(corners != vec4(0u)));
    let uv = clamp(position, vec2(0.0), vec2(1.0));
    var sum = vec3(0.0);
    for (var light = 0u; light < min(transport_settings.sun_count, 4u); light += 1u) {
        let cosine = max(dot(axis, transport_settings.sun_direction[light].xyz), 0.0);
        if cosine <= 0.0 { continue; }
        let bit = 1u << light;
        let c00 = corner_lit(corners.x, bit);
        let c10 = corner_lit(corners.y, bit);
        let c01 = corner_lit(corners.z, bit);
        let c11 = corner_lit(corners.w, bit);
        var visible = (c00 + c10 + c01 + c11) * 0.25;
        if transport_settings.reconstruction != 0u {
            visible = mix(mix(c00, c10, uv.x), mix(c01, c11, uv.x), uv.y);
        }
        sum += transport_settings.sun_color[light].rgb * (cosine * visible);
    }
    return vec4(sum, ready);
}

// Sunlight from the same four corner visibility words as the near cache.
// Corner c (u = c & 1, v = c >> 1) is shared by the coplanar faces of the up
// to four voxels around it, which see the sun identically there, so a face
// the view does not own takes each corner from an owned face sharing it.
// A farther coplanar face can have different visibility and is not a valid
// corner donor. A corner no owned
// face shares is traced, per fragment, under NAADF's kind of budget: a
// bounded step count (its sun rays stop at 80-120) where running out counts as
// lit, and one ray from the face centre for faces about two pixels or less,
// whose corner interpolation is invisible anyway. No screen-space or history
// reuse.
const FALLBACK_SUN_STEPS: u32 = 128u;
fn far_sun(position: vec3<f32>, normal: vec3<f32>, footprint: f32) -> vec4<f32> {
    let reach = f32(params.counts.x);
    if reach == 0.0 { return vec4(0.0); }
    let dominant = max(abs(normal.x), max(abs(normal.y), abs(normal.z)));
    let axis = select(vec3(0.0), sign(normal), abs(normal) == vec3(dominant));
    let face = face_index_of(axis);
    // A coarse block's corners are its block corners, `s` voxels apart.
    let s = voxel_stride(vec3<i32>(floor(position - axis * 0.01)));
    let voxel = vec3<i32>(floor(position - axis * 0.01)) & vec3(-s);
    let surface = vec3<f32>(voxel) + f32(s) * (0.5 + axis * 0.5);
    var u = vec3<i32>(0, s, 0);
    var v = vec3<i32>(0, 0, s);
    if abs(axis.y) > 0.5 { u = vec3<i32>(s, 0, 0); }
    else if abs(axis.z) > 0.5 { u = vec3<i32>(s, 0, 0); v = vec3<i32>(0, s, 0); }
    // Far tier: one ray per light from this fragment, as in NAADF.
    if pixel_owned(vec3<i32>(floor(position - axis * 0.01)), face) {
        var sum = vec3(0.0);
        for (var light = 0u; light < min(transport_settings.sun_count, 4u); light += 1u) {
            let direction = transport_settings.sun_direction[light].xyz;
            let cosine = dot(axis, direction);
            if cosine <= 0.0 { continue; }
            let origin = position + axis * 0.002 + direction * 0.002;
            if trace_domain_limited(origin, direction, reach, true, FALLBACK_SUN_STEPS).status != 1u {
                sum += transport_settings.sun_color[light].rgb * cosine;
            }
        }
        return vec4(sum, 1.0);
    }
    var words = vec4<u32>(0u);
    if far_view.shape.z != 0u {
        // The face itself first, then its eight coplanar neighbours.
        let order = array(4u, 1u, 3u, 5u, 7u, 0u, 2u, 6u, 8u);
        for (var n = 0u; n < 9u && any(words == vec4(0u)); n++) {
            let du = i32(order[n] % 3u) - 1;
            let dv = i32(order[n] / 3u) - 1;
            let offset = u * du + v * dv;
            let index = far_face_index(voxel + offset, face, surface + vec3<f32>(offset));
            if index == 0xffffffffu { continue; }
            let donor = far_view.pixels[index].sky;
            for (var c = 0u; c < 4u; c++) {
                let cu = i32(c & 1u) - du;
                let cv = i32(c >> 1u) - dv;
                if words[c] == 0u && cu >= 0 && cu <= 1 && cv >= 0 && cv <= 1 {
                    words[c] = donor[u32(cu) + u32(cv) * 2u];
                }
            }
        }
    }
    if any(words == vec4(0u)) {
        let base = vec3<f32>(voxel) + max(axis, vec3(0.0)) * f32(s);
        let tiny = footprint >= 0.5 * f32(s);
        var center_word = 0u;
        for (var c = 0u; c < 4u; c++) {
            if words[c] != 0u { continue; }
            if tiny && center_word != 0u {
                words[c] = center_word;
                continue;
            }
            var word = 0x80000000u;
            for (var light = 0u; light < min(transport_settings.sun_count, 4u); light += 1u) {
                let direction = transport_settings.sun_direction[light].xyz;
                if dot(axis, direction) <= 0.0 { continue; }
                var corner = vec3<f32>(u) * f32(c & 1u) + vec3<f32>(v) * f32(c >> 1u);
                if tiny { corner = vec3<f32>(u + v) * 0.5; }
                let origin = base + corner + direction * 0.002;
                if trace_domain_limited(origin, direction, reach, true, FALLBACK_SUN_STEPS).status != 1u {
                    word |= 1u << light;
                }
            }
            words[c] = word;
            if tiny { center_word = word; }
        }
    }
    let local = (position - vec3<f32>(voxel)) / f32(s * s);
    return resolve_sky_corners(words, axis, vec2(dot(local, vec3<f32>(u)), dot(local, vec3<f32>(v))));
}

// Canonical reference for a particular world face, independent of the view
// receiver selection (validation's `sun-reference` component).
fn exact_corner_sun(position: vec3<f32>, axis: vec3<f32>) -> vec4<f32> {
    let reach = f32(params.counts.x);
    if reach == 0.0 { return vec4(0.0); }
    let voxel = floor(position - axis * 0.01);
    var u = vec3(0.0, 1.0, 0.0);
    var v = vec3(0.0, 0.0, 1.0);
    var uv = position.yz - voxel.yz;
    if abs(axis.y) > 0.5 { u = vec3(1.0, 0.0, 0.0); uv = position.xz - voxel.xz; }
    else if abs(axis.z) > 0.5 { u = vec3(1.0, 0.0, 0.0); v = vec3(0.0, 1.0, 0.0); uv = position.xy - voxel.xy; }
    let base = voxel + max(axis, vec3(0.0));
    var sum = vec3(0.0);
    for (var light = 0u; light < min(transport_settings.sun_count, 4u); light += 1u) {
        let direction = transport_settings.sun_direction[light].xyz;
        let cosine = max(dot(axis, direction), 0.0);
        if cosine <= 0.0 { continue; }
        var visible = vec4(0.0);
        for (var corner = 0u; corner < 4u; corner++) {
            let origin = base + u * f32(corner & 1u) + v * f32(corner >> 1u) + direction * 0.002;
            visible[corner] = select(0.0, 1.0, trace_domain(origin, direction, reach, true).status == 0u);
        }
        let amount = mix(mix(visible.x, visible.y, uv.x), mix(visible.z, visible.w, uv.x), uv.y);
        sum += transport_settings.sun_color[light].rgb * cosine * amount;
    }
    return vec4(sum, 1.0);
}

// Four coplanar faces meet at a virtual voxel corner. Only the connected
// component containing this face contributes. In particular, a diagonal face
// across two missing/occluded edge faces cannot leak light into this surface.
fn corner_value(center: vec4<f32>, edge_u: vec4<f32>, edge_v: vec4<f32>, diagonal: vec4<f32>) -> vec3<f32> {
    var sum = vec4(center.rgb * center.w, center.w)
        + vec4(edge_u.rgb * edge_u.w, edge_u.w)
        + vec4(edge_v.rgb * edge_v.w, edge_v.w);
    if edge_u.w > 0.0 || edge_v.w > 0.0 {
        sum += vec4(diagonal.rgb * diagonal.w, diagonal.w);
    }
    return sum.rgb / max(sum.w, 0.000001);
}

fn display_face(voxel: vec3<i32>, face: u32, s: i32) -> vec4<f32> {
    let surface = vec3<f32>(voxel) + f32(s) * (0.5 + 0.5 * vec3<f32>(FACE_NORMALS[face]));
    let index = far_face_index(voxel, face, surface);
    if index != 0xffffffffu {
        let value = far_view.pixels[index];
        return vec4(value.irradiance, min(value.evidence, 1.0));
    }
    return vec4(0.0);
}

// Virtual-corner interpolation over visible faces. The view index only
// contains visible faces; missing neighbours never become invented paths or
// borrowed physical history. All of this remains presentation-only. A coarse
// block's neighbours are the blocks beside it.
fn corner_irradiance(position: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    if transport_settings.enabled == 0u {
        return vec4(0.0);
    }
    let s = voxel_stride(vec3<i32>(floor(position - normal * 0.01)));
    let voxel = vec3<i32>(floor(position - normal * 0.01)) & vec3(-s);
    var face = 0u;
    var u = vec3<i32>(0, s, 0);
    var v = vec3<i32>(0, 0, s);
    if abs(normal.y) > 0.5 {
        face = 2u;
        u = vec3<i32>(s, 0, 0);
    } else if abs(normal.z) > 0.5 {
        face = 4u;
        u = vec3<i32>(s, 0, 0);
        v = vec3<i32>(0, s, 0);
    }
    let axis = face / 2u;
    face += select(0u, 1u, normal[axis] < 0.0);
    let center = display_face(voxel, face, s);
    if transport_settings.reconstruction == 0u {
        return center;
    }
    let left = display_face(voxel - u, face, s);
    let right = display_face(voxel + u, face, s);
    let bottom = display_face(voxel - v, face, s);
    let top = display_face(voxel + v, face, s);
    let c00 = corner_value(center, left, bottom, display_face(voxel - u - v, face, s));
    let c10 = corner_value(center, right, bottom, display_face(voxel + u - v, face, s));
    let c01 = corner_value(center, left, top, display_face(voxel - u + v, face, s));
    let c11 = corner_value(center, right, top, display_face(voxel + u + v, face, s));
    let local = (position - vec3<f32>(voxel)) / f32(s * s);
    let uv = clamp(vec2(dot(local, vec3<f32>(u)), dot(local, vec3<f32>(v))), vec2(0.0), vec2(1.0));
    return vec4(mix(mix(c00, c10, uv.x), mix(c01, c11, uv.x), uv.y), center.w);
}

fn use_voxel_sky() -> bool {
    return transport_settings.voxel_sky != 0u;
}

// ---- Host API ----

/// GI is presenting (`VoxelGiDisplay` and a view are active).
fn enabled() -> bool {
    return transport_settings.enabled != 0u;
}

/// Sunlight comes from GI's cached sun visibility (replace the host's
/// directional-light term with `sunlight`), not the host's shadow maps.
fn voxel_sky() -> bool {
    return use_voxel_sky();
}

/// The validation diagnostic mode (0 in normal presentation).
fn debug_mode() -> u32 {
    return transport_settings.debug_irradiance;
}

/// Indirect diffuse light arriving at a fragment, scaled by the display
/// intensity: multiply by the surface albedo. `found` is false where no
/// support exists yet (show no indirect light). `coverage` is the
/// validation colour: cyan far tier, green receiver, yellow hole fill, red
/// none.
struct IndirectLight {
    light: vec3<f32>,
    found: bool,
    coverage: vec3<f32>,
}

/// `pixel` is the fragment's framebuffer position (`@builtin(position).xy`),
/// `position` its world position on the voxel face, `normal` the face's
/// axis-aligned normal.
fn indirect_light(pixel: vec2<f32>, position: vec3<f32>, normal: vec3<f32>) -> IndirectLight {
    var out: IndirectLight;
    out.coverage = vec3(1.0, 0.0, 0.0);
    if transport_settings.enabled == 0u { return out; }
    let voxel = vec3<i32>(floor(position - normal * 0.01));
    var far = vec4(0.0);
    if pixel_owned(voxel, face_index_of(normal)) {
        // Far tier; receivers only fill its holes.
        far = pixel_irradiance(pixel, position, normal, voxel_stride(voxel));
        if far.w > 0.0 { out.coverage = vec3(0.0, 1.0, 1.0); }
        else if far_view.shape.z != 0u {
            far = far_irradiance(pixel, position, normal);
            if far.w > 0.0 { out.coverage = vec3(1.0, 1.0, 0.0); }
        }
    } else if far_view.shape.z != 0u {
        far = corner_irradiance(position, normal);
        if far.w > 0.0 { out.coverage = vec3(0.0, 1.0, 0.0); }
        // Thin/distant faces may miss the reduced prepass grid entirely.
        // Only those holes use geometric upsampling, not world history.
        if far.w == 0.0 {
            far = far_irradiance(pixel, position, normal);
            if far.w > 0.0 { out.coverage = vec3(1.0, 1.0, 0.0); }
        }
    }
    out.found = far.w > 0.0;
    if out.found { out.light = far.rgb * transport_settings.intensity; }
    return out;
}

/// Direct sunlight at a fragment from GI's cached sun visibility, in
/// transport units (sun illuminance / 10,000): for a Lambertian surface
/// multiply by albedo * 10,000 / pi. `footprint` is the fragment's world
/// size in voxels (the largest component of `fwidth(world_position)`).
fn sunlight(position: vec3<f32>, normal: vec3<f32>, footprint: f32) -> vec3<f32> {
    return far_sun(position, normal, footprint).rgb;
}

/// The validation reference: the exact corner sun of this face, traced.
fn sunlight_reference(position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return exact_corner_sun(position, normal).rgb;
}

/// The emitted light of the voxel under a face (zero for air or before the
/// first view output).
fn emission(position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    if far_view.shape.z == 0u { return vec3(0.0); }
    let material = voxel_material(vec3<i32>(floor(position - normal * 0.01)));
    if material >= 256u { return vec3(0.0); }
    return far_view.optical[material * 4u + 2u].rgb;
}

/// GI's suns (the Bevy `DirectionalLight`s it uses, at most four), for
/// hosts that light faces GI does not cover with their own visibility test.
/// `sun_direction` points toward the sun; `sun_color` is in transport units,
/// like `sunlight`.
fn sun_count() -> u32 {
    return min(transport_settings.sun_count, 4u);
}

fn sun_direction(sun: u32) -> vec3<f32> {
    return transport_settings.sun_direction[sun].xyz;
}

fn sun_color(sun: u32) -> vec3<f32> {
    return transport_settings.sun_color[sun].rgb;
}

/// Whether this face is an exposed face of GI's geometry. GI's light
/// (`indirect_light`, `sunlight`, `emission`) describes only such faces:
/// hosts light the others themselves (faces outside the declared pages, or
/// geometry GI has not received yet).
fn face_in_geometry(position: vec3<f32>, normal: vec3<f32>) -> bool {
    let voxel = vec3<i32>(floor(position - normal * 0.01));
    let face = face_index_of(normal);
    return voxel_material(voxel) != 256u && voxel_material(voxel + FACE_NORMALS[face]) == 256u;
}
