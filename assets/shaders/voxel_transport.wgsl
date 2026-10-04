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
    direct_sky: array<vec4<f32>, 6>,
}

struct GpuTransportSettings {
    hash_mask: u32,
    probe_count: u32,
    enabled: u32,
    reconstruction: u32,
    intensity: f32,
    debug_irradiance: u32,
    voxel_sky: u32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var<storage, read> transport_hash: array<GpuHashEntry>;
@group(#{MATERIAL_BIND_GROUP}) @binding(103)
var<storage, read> transport_probes: array<GpuProbe>;
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

fn pcg(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_voxel(voxel: vec3<i32>) -> u32 {
    var hash = pcg(bitcast<u32>(voxel.x));
    hash = pcg(hash ^ bitcast<u32>(voxel.y));
    return pcg(hash ^ bitcast<u32>(voxel.z));
}

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

fn lookup_face(voxel: vec3<i32>, face: u32, sky: bool) -> vec4<f32> {
    let probe_index = lookup_probe(voxel);
    if probe_index >= transport_settings.probe_count {
        return vec4(0.0);
    }

    if sky { return transport_probes[probe_index].direct_sky[face]; }
    return transport_probes[probe_index].faces[face];
}

// Four coplanar faces meet at a virtual voxel corner. Only the connected
// component containing this face contributes. In particular, a diagonal face
// across two missing/occluded edge faces cannot leak light into this surface.
fn corner_value(center: vec4<f32>, edge_u: vec4<f32>, edge_v: vec4<f32>, diagonal: vec4<f32>) -> vec3<f32> {
    var sum = center + edge_u + edge_v;
    if edge_u.w > 0.0 || edge_v.w > 0.0 {
        sum += diagonal;
    }
    return sum.rgb / max(sum.w, 1.0);
}

fn probe_irradiance(position: vec3<f32>, normal: vec3<f32>, sky: bool) -> vec3<f32> {
    if (!sky && transport_settings.enabled == 0u) || transport_settings.probe_count == 0u {
        return vec3(0.0);
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
    let intensity = select(transport_settings.intensity, 1.0, sky);
    let center = lookup_face(voxel, face, sky);
    if transport_settings.reconstruction == 0u || center.w == 0.0 {
        return center.rgb * intensity;
    }
    let left = lookup_face(voxel - u, face, sky);
    let right = lookup_face(voxel + u, face, sky);
    let bottom = lookup_face(voxel - v, face, sky);
    let top = lookup_face(voxel + v, face, sky);
    let c00 = corner_value(center, left, bottom, lookup_face(voxel - u - v, face, sky));
    let c10 = corner_value(center, right, bottom, lookup_face(voxel + u - v, face, sky));
    let c01 = corner_value(center, left, top, lookup_face(voxel - u + v, face, sky));
    let c11 = corner_value(center, right, top, lookup_face(voxel + u + v, face, sky));
    let local = position - vec3<f32>(voxel);
    let uv = clamp(vec2(dot(local, vec3<f32>(u)), dot(local, vec3<f32>(v))), vec2(0.0), vec2(1.0));
    return mix(mix(c00, c10, uv.x), mix(c01, c11, uv.x), uv.y) * intensity;
}

fn use_voxel_sky() -> bool {
    return transport_settings.voxel_sky != 0u && transport_settings.probe_count != 0u;
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
    if light_id != 0u || !enable_diffuse { return vec3(0.0); }
    let sky = probe_irradiance((*input).P, (*input).layers[lighting::LAYER_BASE].N, true);
    // Transport imports sun illuminance / 10,000. Restore photometric scale
    // for a Lambertian response here; Bevy applies camera exposure afterwards.
    // This comparison intentionally has no view-dependent sun specular lobe.
    return (*input).diffuse_color * sky * (10000.0 / 3.14159265359);
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
    if index < transport_settings.probe_count { emission = transport_probes[index].emission.rgb; }
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
