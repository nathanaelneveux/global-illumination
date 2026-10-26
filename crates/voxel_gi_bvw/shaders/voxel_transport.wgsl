#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_functions,
    view_transformations::position_world_to_clip
}
#import bevy_pbr::pbr_bindings
#import bevy_pbr::{lighting, shadows, mesh_view_bindings}
#import bevy_render::instance_index::get_instance_index

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var mat_array_texture: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var mat_array_texture_sampler: sampler;

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

struct Vertex {
    @builtin(instance_index) instance_index: u32,
#ifdef VERTEX_POSITIONS
    @location(0) position: vec3<f32>,
#endif
#ifdef VERTEX_NORMALS
    @location(1) normal: vec3<f32>,
#endif
#ifdef VERTEX_UVS
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_TANGENTS
    @location(4) tangent: vec4<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
    @location(8) tex_idx: vec3<u32>,
}

struct CustomVertexOutput {
    @builtin(position) position: vec4<f32>,
    // MSAA may shade a covered sample whose pixel centre is outside this
    // primitive. Face-keyed lighting must use a position on the actual face.
    @location(0) @interpolate(perspective, centroid) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
#ifdef VERTEX_UVS
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_TANGENTS
    @location(4) world_tangent: vec4<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(6) @interpolate(flat) instance_index: u32,
#endif
    @location(8) tex_idx: vec3<u32>,
}

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

// EXPERIMENT: the per-pixel tier (voxel_view_pixels.wgsl). Faces whose block
// projects under `origin.z` pixels have no receiver; their cells hold
// per-pixel indirect light.
struct FarCells {
    camera: vec4<f32>,
    axis: vec4<f32>, // view forward, pixels per unit at depth 1
    grid: vec4<u32>, // cells x, y, pixels per cell side, enabled
    origin: vec4<u32>, // viewport origin, threshold px (bits)
    cells: array<vec4<u32>>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(111)
var<storage, read> far_cells: FarCells;

// The per-pixel tier's ownership test, exactly as it and discovery compute
// it (`pixel_owned` in voxel_view_pixels.wgsl; mode in grid.w bits 1-2).
fn pixel_owned(voxel: vec3<i32>, face: u32) -> bool {
    let grid = far_cells.grid.w;
    if (grid & 1u) == 0u || face >= 6u { return false; }
    let mode = (grid >> 1u) & 3u;
    let s = select(1, voxel_stride(voxel), mode == 2u);
    let ray = vec3<f32>(voxel & vec3(-s)) + 0.5 * f32(s) - far_cells.camera.xyz;
    let depth = dot(ray, far_cells.axis.xyz);
    if depth <= 0.0 { return false; }
    let side = f32(s) * far_cells.axis.w / depth;
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
            if cell.z != face { continue; }
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

// #include voxel_traversal.wgsl

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
// Borrowing a coplanar face's corners from up to 16 voxels away (as before)
// gave a face another voxel's shadow at some camera poses. A corner no owned
// face shares is traced, per fragment, under NAADF's kind of budget: a
// bounded step count (its sun rays stop at 80-120) where running out counts as
// lit, and one ray from the face centre for faces about two pixels or less,
// whose corner interpolation is invisible anyway. No screen-space or history
// reuse.
const FALLBACK_SUN_STEPS: u32 = 128u;
// World units per pixel at this fragment; set by `fragment` before lighting.
var<private> fragment_footprint: f32 = 0.0;

fn far_sun(position: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
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
    // EXPERIMENT per-pixel tier: one ray per light from this fragment.
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
        let tiny = fragment_footprint >= 0.5 * f32(s);
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

// Naga-oil overrides are scoped to this material's shader. Other materials,
// point lights and the simulation's sources remain untouched. Sum the cached
// directional sources once, not once per Bevy directional-light loop entry.
override fn lighting::directional_light(
    light_id: u32,
    input: ptr<function, lighting::LightingInput>,
    enable_diffuse: bool,
) -> vec3<f32> {
    if !use_voxel_sky() {
        return lighting::directional_light(light_id, input, enable_diffuse);
    }
    let sky = far_sun((*input).P, (*input).layers[lighting::LAYER_BASE].N);
    if light_id != 0u || !enable_diffuse { return vec3(0.0); }
    // Transport imports sun illuminance / 10,000. Restore photometric scale
    // for a Lambertian response here; Bevy applies camera exposure afterwards.
    // This comparison intentionally has no view-dependent sun specular lobe.
    return (*input).diffuse_color * sky.rgb * (10000.0 / 3.14159265359);
}

override fn shadows::fetch_directional_shadow(
    light_id: u32,
    frag_position: vec4<f32>,
    surface_normal: vec3<f32>,
    view_z: f32,
    frag_coord_xy: vec2<f32>,
) -> f32 {
    if use_voxel_sky() { return 1.0; }
    return shadows::fetch_directional_shadow(light_id, frag_position, surface_normal, view_z, frag_coord_xy);
}

@vertex
fn vertex(vertex: Vertex) -> CustomVertexOutput {
    var out: CustomVertexOutput;
    let model = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(vertex.normal, vertex.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(vertex.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
#ifdef VERTEX_UVS
    out.uv = vertex.uv;
#endif
#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(model, vertex.tangent, vertex.instance_index);
#endif
    out.color = vertex.color;
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
    out.tex_idx = vertex.tex_idx;
    return out;
}

@fragment
fn fragment(in: CustomVertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var standard_in: VertexOutput;
    standard_in.position = in.position;
    standard_in.world_normal = in.world_normal;
    standard_in.world_position = in.world_position;
    standard_in.uv = in.uv;
    standard_in.color = in.color;
    standard_in.instance_index = in.instance_index;
    var pbr_input = pbr_input_from_standard_material(standard_in, is_front);
    // Derivatives before any divergent branch.
    let world_per_pixel = fwidth(in.world_position.xyz);
    fragment_footprint = max(world_per_pixel.x, max(world_per_pixel.y, world_per_pixel.z));

    var tex_face = 0;
    if in.world_normal.y == 0.0 {
        tex_face = 1;
    } else if in.world_normal.y < 0.0 {
        tex_face = 2;
    }

    let albedo = textureSample(mat_array_texture, mat_array_texture_sampler, in.uv, in.tex_idx[tex_face]);
    pbr_input.material.base_color = albedo * in.color;
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    return deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    var indirect = vec3(0.0);
    var view_coverage = vec3(1.0, 0.0, 0.0);
    let voxel = vec3<i32>(floor(in.world_position.xyz - in.world_normal * 0.01));
    if transport_settings.enabled != 0u && pixel_owned(voxel, face_index_of(normalize(in.world_normal))) {
        // EXPERIMENT: per-pixel tier; receivers only fill its holes.
        var far = pixel_irradiance(in.position.xy, in.world_position.xyz, normalize(in.world_normal), voxel_stride(voxel));
        if far.w > 0.0 { view_coverage = vec3(0.0, 1.0, 1.0); }
        else if far_view.shape.z != 0u {
            far = far_irradiance(in.position.xy, in.world_position.xyz, normalize(in.world_normal));
            if far.w > 0.0 { view_coverage = vec3(1.0, 1.0, 0.0); }
        }
        if far.w > 0.0 { indirect = far.rgb * transport_settings.intensity; }
    } else if far_view.shape.z != 0u && transport_settings.enabled != 0u {
        var far = corner_irradiance(in.world_position.xyz, normalize(in.world_normal));
        if far.w > 0.0 { view_coverage = vec3(0.0, 1.0, 0.0); }
        // Thin/distant faces may miss the reduced prepass grid entirely.
        // Only those holes use geometric upsampling, not world history.
        if far.w == 0.0 {
            far = far_irradiance(in.position.xy, in.world_position.xyz, normalize(in.world_normal));
            if far.w > 0.0 { view_coverage = vec3(1.0, 1.0, 0.0); }
        }
        if far.w > 0.0 { indirect = far.rgb * transport_settings.intensity; }
    }
    // Validation-only identity coverage, before tone mapping or fog. Selected
    // receiver counters cannot detect visible surfaces absent from the grid.
    if transport_settings.debug_irradiance == 6u {
        let face = face_index_of(normalize(in.world_normal));
        let valid = voxel_material(voxel) != 256u && voxel_material(voxel + FACE_NORMALS[face]) == 256u;
        // Independent channels survive MSAA resolve: red is missing indirect
        // support, green is a lookup outside exposed transport geometry.
        // A completely covered indoor view legitimately has a zero failure
        // mask. A tiny blue witness proves the diagnostic shader ran, without
        // relaxing black-frame checks for ordinary lighting captures. The
        // analyzer excludes blue (sky, mixed silhouettes and this witness).
        let witness = select(0.0, 0.5, all(in.position.xy < vec2(4.0)));
        out.color = vec4(1.0 - view_coverage.g, select(1.0, 0.0, valid), witness, 1.0);
        return out;
    }
    if transport_settings.debug_irradiance == 2u {
        out.color = vec4(view_coverage, 1.0);
        return out;
    }
    var emission = vec3(0.0);
    if far_view.shape.z != 0u {
        let material = voxel_material(voxel);
        if material < 256u {
            emission = far_view.optical[material * 4u + 2u].rgb;
        }
    }
    out.color = vec4<f32>(
        out.color.rgb + pbr_input.material.base_color.rgb * (1.0 - pbr_input.material.metallic) * indirect + emission,
        out.color.a,
    );
    if transport_settings.debug_irradiance != 0u {
        var diagnostic_light = indirect;
        // Component captures isolate motion-induced lighting changes from
        // textured appearance. Normal presentation never enters this branch.
        if transport_settings.debug_irradiance >= 4u { diagnostic_light = vec3(0.0); }
        if transport_settings.debug_irradiance != 3u && use_voxel_sky() {
            var sunlight = far_sun(in.world_position.xyz, normalize(in.world_normal)).rgb;
            if transport_settings.debug_irradiance == 5u {
                sunlight = exact_corner_sun(in.world_position.xyz, normalize(in.world_normal)).rgb;
            }
            diagnostic_light += sunlight
                * (10000.0 / 3.14159265359) * mesh_view_bindings::view.exposure;
        }
        out.color = vec4(diagnostic_light, 1.0);
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
#endif
}
