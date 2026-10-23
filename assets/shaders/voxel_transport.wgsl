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
    keys: array<u32, 131072>,
    heads: array<u32, 32768>, links: array<u32, 65536>,
    pixels: array<FarPixel>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(110)
var<storage, read> far_view: FarOutput;

// Geometry-aware upsampling only. No value is reused across an unrelated
// face orientation or depth plane, even at foreground/background edges.
fn far_face_index(voxel: vec3<i32>, face: u32) -> u32 {
    var slot = (hash_voxel(voxel) ^ pcg(face + 1u)) & far_view.shape.w;
    for (var step = 0u; step < 64u; step++) {
        let entry = far_view.keys[slot];
        if entry == 0u { break; }
        let value = far_view.pixels[entry - 1u];
        let donor = vec3<i32>(floor(value.surface - vec3<f32>(FACE_NORMALS[face]) * 0.01));
        if value.face == face && all(donor == voxel) { return entry - 1u; }
        slot = (slot + 1u) & far_view.shape.w;
    }
    return 0xffffffffu;
}

fn far_value(donor: FarPixel, position: vec3<f32>, normal: vec3<f32>, sky: bool) -> vec3<f32> {
    if !sky { return donor.irradiance; }
    let local = position - floor(donor.surface - normal * 0.01);
    var uv = local.yz;
    if donor.face / 2u == 1u { uv = local.xz; }
    else if donor.face / 2u == 2u { uv = local.xy; }
    return resolve_sky_corners(donor.sky, normal, uv).rgb;
}

fn far_irradiance(pixel: vec2<f32>, position: vec3<f32>, normal: vec3<f32>, sky: bool) -> vec4<f32> {
    if far_view.shape.z == 0u { return vec4(0.0); }
    let uv = (pixel - vec2<f32>(far_view.viewport.xy)) / vec2<f32>(far_view.viewport.zw);
    if any(uv < vec2(0.0)) || any(uv >= vec2(1.0)) { return vec4(0.0); }
    let voxel = vec3<i32>(floor(position - normal * 0.01));
    let surface = vec3<f32>(voxel) + 0.5 + normal * 0.5;
    let exact = far_face_index(voxel, face_index_of(normal));
    if exact != 0xffffffffu {
        let value = far_view.pixels[exact];
        if (!sky && value.evidence > 0.0) || (sky && all(value.sky != vec4(0u))) {
            return vec4(far_value(value, position, normal, sky), 1.0);
        }
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
                if !sky && donor.evidence == 0.0 { continue; }
                if sky && any(donor.sky == vec4(0u)) { continue; }
                if dot(vec3<f32>(FACE_NORMALS[donor.face]), normal) < 0.99 { continue; }
                let delta = donor.surface - surface;
                let d2 = dot(delta, delta);
                if d2 > 256.0 { continue; }
                let offset = abs(dot(delta, normal));
                let w = 1.0 / (0.25 + d2);
                if offset <= 0.05 {
                    sum += far_value(donor, position, normal, sky) * w;
                    weight += w;
                } else if !sky && offset <= 4.05 {
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

// View-owned corners use the same visibility and interpolation as near. A
// surface missed by reduced-resolution discovery evaluates its exact corners
// here; this is coverage-hole shading, not a switch to shadow maps or hard
// per-pixel shadows. It has no history or feedback into indirect transport.
fn far_sun(position: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    let reach = f32(params.counts.x);
    if reach == 0.0 { return vec4(0.0); }
    let dominant = max(abs(normal.x), max(abs(normal.y), abs(normal.z)));
    let axis = select(vec3(0.0), sign(normal), abs(normal) == vec3(dominant));
    let clip = position_world_to_clip(position);
    let screen = clip.xy / clip.w * vec2(0.5, -0.5) + 0.5;
    let pixel = vec2<f32>(far_view.viewport.xy) + screen * vec2<f32>(far_view.viewport.zw);
    let cached = far_irradiance(pixel, position, axis, true);
    if cached.w > 0.0 { return cached; }
    return exact_corner_sun(position, axis);
}

// Canonical reference for a particular world face, independent of the view
// receiver selection. Also serves the existing uncovered-face fallback.
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

fn display_face(voxel: vec3<i32>, face: u32) -> vec4<f32> {
    let index = far_face_index(voxel, face);
    if index != 0xffffffffu {
        let value = far_view.pixels[index];
        return vec4(value.irradiance, min(value.evidence, 1.0));
    }
    return vec4(0.0);
}

// Virtual-corner interpolation over visible faces. The view index only
// contains visible faces; missing neighbours never become invented paths or
// borrowed physical history. All of this remains presentation-only.
fn corner_irradiance(position: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    if transport_settings.enabled == 0u {
        return vec4(0.0);
    }
    let voxel = vec3<i32>(floor(position - normal * 0.01));
    var face = 0u;
    var u = vec3<i32>(0, 1, 0);
    var v = vec3<i32>(0, 0, 1);
    if abs(normal.y) > 0.5 {
        face = 2u;
        u = vec3<i32>(1, 0, 0);
    } else if abs(normal.z) > 0.5 {
        face = 4u;
        u = vec3<i32>(1, 0, 0);
        v = vec3<i32>(0, 1, 0);
    }
    let axis = face / 2u;
    face += select(0u, 1u, normal[axis] < 0.0);
    let center = display_face(voxel, face);
    if transport_settings.reconstruction == 0u {
        return center;
    }
    let left = display_face(voxel - u, face);
    let right = display_face(voxel + u, face);
    let bottom = display_face(voxel - v, face);
    let top = display_face(voxel + v, face);
    let c00 = corner_value(center, left, bottom, display_face(voxel - u - v, face));
    let c10 = corner_value(center, right, bottom, display_face(voxel + u - v, face));
    let c01 = corner_value(center, left, top, display_face(voxel - u + v, face));
    let c11 = corner_value(center, right, top, display_face(voxel + u + v, face));
    let local = position - vec3<f32>(voxel);
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
    if far_view.shape.z != 0u && transport_settings.enabled != 0u {
        var far = corner_irradiance(in.world_position.xyz, normalize(in.world_normal));
        if far.w > 0.0 { view_coverage = vec3(0.0, 1.0, 0.0); }
        // Thin/distant faces may miss the reduced prepass grid entirely.
        // Only those holes use geometric upsampling, not world history.
        if far.w == 0.0 {
            far = far_irradiance(in.position.xy, in.world_position.xyz, normalize(in.world_normal), false);
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
