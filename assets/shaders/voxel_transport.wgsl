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

struct GpuHashEntry {
    voxel: vec3<i32>,
    probe_index: u32,
}

struct GpuProbe {
    faces: array<vec4<f32>, 6>,
    emission: vec4<f32>,
    direct_sky: array<vec4<u32>, 6>, // four corner visibility words per face
    resident: array<vec2<u32>, 6>, // slot + allocation incarnation
}

struct GpuTransportSettings {
    hash_mask: u32,
    probe_capacity: u32,
    enabled: u32,
    reconstruction: u32,
    intensity: f32,
    debug_irradiance: u32,
    voxel_sky: u32,
    sun_count: u32,
    sun_direction: array<vec4<f32>, 4>,
    sun_color: array<vec4<f32>, 4>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var<storage, read> transport_hash: array<GpuHashEntry>;
@group(#{MATERIAL_BIND_GROUP}) @binding(103)
var<storage, read> transport_probes: array<GpuProbe>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104)
var<uniform> transport_settings: GpuTransportSettings;
@group(#{MATERIAL_BIND_GROUP}) @binding(105)
var<storage, read> resident_faces: array<vec4<u32>>;

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
    @location(0) world_position: vec4<f32>,
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

// The GI worker's transport geometry, for faces without a display probe.
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

// #include voxel_traversal.wgsl

fn lookup_probe(voxel: vec3<i32>) -> u32 {
    var slot = hash_voxel(voxel) & transport_settings.hash_mask;
    var probe_index = 0xffffffffu;
    for (var attempt = 0u; attempt < 32u; attempt += 1u) {
        let entry = transport_hash[slot];
        if entry.probe_index == 0xffffffffu {
            break;
        }
        if entry.probe_index != 0xfffffffeu && all(entry.voxel == voxel) {
            probe_index = entry.probe_index;
            break;
        }
        slot = (slot + 1u) & transport_settings.hash_mask;
    }
    return probe_index;
}

fn lookup_face(voxel: vec3<i32>, face: u32) -> vec4<f32> {
    let probe_index = lookup_probe(voxel);
    if probe_index >= transport_settings.probe_capacity {
        return vec4(0.0);
    }

    return resolved_face(probe_index, face);
}

fn resolved_face(index: u32, face: u32) -> vec4<f32> {
    let fallback = transport_probes[index].faces[face];
    let address = transport_probes[index].resident[face];
    let base = address.x * 9u;
    if fallback.w > 0.0 && address.y != 0u && base + 8u < arrayLength(&resident_faces) {
        if resident_faces[base + 7u].x == address.y {
            // Final presentation is separate from the unfiltered temporal
            // resolve at record six. Spatial reconstruction never feeds back.
            let display = bitcast<vec4<f32>>(vec4(resident_faces[base + 1u].w, resident_faces[base + 3u].yzw));
            let previous = bitcast<vec4<f32>>(resident_faces[base + 8u]);
            let duration = resident_faces[base + 7u].z;
            let age = mesh_view_bindings::globals.frame_count - resident_faces[base + 8u].w;
            let elapsed = select(age, 0u, age > 0x7fffffffu);
            let blend = select(1.0, min(f32(elapsed) / f32(max(duration, 1u)), 1.0), duration > 0u && duration <= 32u);
            let custom = bitcast<vec4<f32>>(resident_faces[base + 5u]).yzw;
            return vec4(mix(previous.rgb, display.rgb, blend) + custom, max(display.a, 0.03125));
        }
    }
    return fallback;
}

const FACE_NORMALS = array<vec3<i32>, 6>(
    vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
    vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1),
);

// Bridge a mesh/cache publication gap, for both digging and placement. Prefer
// old equally oriented surfaces (including the surface just covered by a new
// block). Otherwise borrow faces bordering the receiver's known-air cell, or
// the removed voxel's old faces. No arbitrary-radius search through walls.
// This display-only estimate is never stored or used as transport evidence.
fn unpublished_face_prior(voxel: vec3<i32>, receiver_face: u32) -> vec4<f32> {
    var sum = vec4(0.0);
    for (var face = 0u; face < 6u; face += 1u) {
        let value = lookup_face(voxel + FACE_NORMALS[face], receiver_face);
        sum += vec4(value.rgb * value.w, value.w);
    }
    if sum.w > 0.0 { return vec4(sum.rgb / sum.w, 0.125); }
    let exterior = voxel + FACE_NORMALS[receiver_face];
    let index = lookup_probe(exterior);
    for (var face = 0u; face < 6u; face += 1u) {
        let value = lookup_face(exterior - FACE_NORMALS[face], face);
        sum += vec4(value.rgb * value.w, value.w);
        if index < transport_settings.probe_capacity {
            let removed = resolved_face(index, face);
            sum += vec4(removed.rgb * removed.w, removed.w);
        }
    }
    return vec4(sum.rgb / max(sum.w, 0.000001), select(0.0, 0.125, sum.w > 0.0));
}

fn corner_lit(word: u32, bit: u32) -> f32 {
    return select(0.0, 1.0, (word & bit) != 0u);
}

// These are ray measurements AT the virtual corners, not an average of nearby
// face-center rays. The four values naturally retain contact shadows at walls.
// Corners store which suns they see; each sun's current direction and color
// light the face here, so a moving sun brightens or dims every frame while
// cached shadows change only when its quantized direction does. Unknown
// corners contribute no direct sunlight yet; available corners remain usable.
// Readiness must NEVER switch a face to another shadow algorithm.
fn sky_irradiance(position: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    let voxel = vec3<i32>(floor(position - normal * 0.01));
    let index = lookup_probe(voxel);
    if index >= transport_settings.probe_capacity { return far_sun(position, normal); }
    var face = 0u;
    var uv = position.yz - vec2<f32>(voxel.yz);
    if abs(normal.y) > 0.5 {
        face = 2u;
        uv = position.xz - vec2<f32>(voxel.xz);
    } else if abs(normal.z) > 0.5 {
        face = 4u;
        uv = position.xy - vec2<f32>(voxel.xy);
    }
    let negative = normal[face / 2u] < 0.0;
    face += select(0u, 1u, negative);
    var axis = vec3(0.0);
    axis[face / 2u] = select(1.0, -1.0, negative);
    let corners = transport_probes[index].direct_sky[face];
    let ready = select(0.0, 1.0, all(corners != vec4(0u)));
    uv = clamp(uv, vec2(0.0), vec2(1.0));
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

// Faces with no display probe (beyond the receiver radius, or not published
// yet) trace their own sun visibility per fragment through the GI geometry:
// the same suns, colors and visibility rule as the corner kernel
// (`trace_sky_corners`), as hard per-pixel shadows instead of reconstructed
// corners.
fn far_sun(position: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    let reach = f32(params.counts.x);
    if reach == 0.0 { return vec4(0.0); }
    let dominant = max(abs(normal.x), max(abs(normal.y), abs(normal.z)));
    let axis = select(vec3(0.0), sign(normal), abs(normal) == vec3(dominant));
    var sum = vec3(0.0);
    for (var light = 0u; light < min(transport_settings.sun_count, 4u); light += 1u) {
        let direction = transport_settings.sun_direction[light].xyz;
        let cosine = max(dot(axis, direction), 0.0);
        if cosine <= 0.0 { continue; }
        let origin = position + axis * 0.002 + direction * 0.002;
        if trace_domain(origin, direction, reach, true).status == 0u {
            sum += transport_settings.sun_color[light].rgb * cosine;
        }
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

fn probe_irradiance(position: vec3<f32>, normal: vec3<f32>, sky: bool) -> vec3<f32> {
    if (!sky && transport_settings.enabled == 0u) || transport_settings.probe_capacity == 0u {
        return vec3(0.0);
    }
    if sky { return sky_irradiance(position, normal).rgb; }
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
    let intensity = transport_settings.intensity;
    var center = lookup_face(voxel, face);
    if center.w == 0.0 {
        center = unpublished_face_prior(voxel, face);
    }
    if transport_settings.reconstruction == 0u {
        return center.rgb * intensity;
    }
    let left = lookup_face(voxel - u, face);
    let right = lookup_face(voxel + u, face);
    let bottom = lookup_face(voxel - v, face);
    let top = lookup_face(voxel + v, face);
    let c00 = corner_value(center, left, bottom, lookup_face(voxel - u - v, face));
    let c10 = corner_value(center, right, bottom, lookup_face(voxel + u - v, face));
    let c01 = corner_value(center, left, top, lookup_face(voxel - u + v, face));
    let c11 = corner_value(center, right, top, lookup_face(voxel + u + v, face));
    let local = position - vec3<f32>(voxel);
    let uv = clamp(vec2(dot(local, vec3<f32>(u)), dot(local, vec3<f32>(v))), vec2(0.0), vec2(1.0));
    return mix(mix(c00, c10, uv.x), mix(c01, c11, uv.x), uv.y) * intensity;
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
    let sky = sky_irradiance((*input).P, (*input).layers[lighting::LAYER_BASE].N);
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
    let indirect = probe_irradiance(in.world_position.xyz, normalize(in.world_normal), false);
    let voxel = vec3<i32>(floor(in.world_position.xyz - in.world_normal * 0.01));
    let index = lookup_probe(voxel);
    var emission = vec3(0.0);
    if index < transport_settings.probe_capacity { emission = transport_probes[index].emission.rgb; }
    out.color = vec4<f32>(
        out.color.rgb + pbr_input.material.base_color.rgb * (1.0 - pbr_input.material.metallic) * indirect + emission,
        out.color.a,
    );
    if transport_settings.debug_irradiance != 0u {
        var diagnostic_light = indirect;
        if use_voxel_sky() {
            diagnostic_light += probe_irradiance(in.world_position.xyz, normalize(in.world_normal), true)
                * (10000.0 / 3.14159265359) * mesh_view_bindings::view.exposure;
        }
        out.color = vec4(diagnostic_light, 1.0);
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
#endif
}
