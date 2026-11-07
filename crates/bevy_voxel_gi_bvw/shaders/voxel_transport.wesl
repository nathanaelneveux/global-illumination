// The `bevy_voxel_world` mesh material lit by voxel GI: BVW's texture array
// and vertex layout over `StandardMaterial`, with GI's indirect light and
// emission, and GI's cached voxel sun in place of Bevy's shadowed sun
// (`bevy_voxel_gi::pbr::apply_pbr_lighting`).
import bevy_pbr::render::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_functions,
    mesh_view_bindings,
    view_transformations::position_world_to_clip,
};
import bevy_voxel_gi::presentation::{
    indirect_light, sunlight, sunlight_reference, emission, face_in_geometry, voxel_sky, debug_mode,
};

@if(PREPASS_PIPELINE)
import bevy_pbr::{
    prepass::io::{VertexOutput, FragmentOutput},
    deferred::functions::deferred_output,
};
@else
import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::main_pass_post_lighting_processing,
};
@if(!PREPASS_PIPELINE)
import bevy_voxel_gi::pbr::{apply_pbr_lighting, pixel_footprint};

@group(constants::MATERIAL_BIND_GROUP) @binding(100)
var mat_array_texture: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(101)
var mat_array_texture_sampler: sampler;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @if(VERTEX_POSITIONS)
    @location(0) position: vec3<f32>,
    @if(VERTEX_NORMALS)
    @location(1) normal: vec3<f32>,
    @if(VERTEX_UVS)
    @location(2) uv: vec2<f32>,
    @if(VERTEX_TANGENTS)
    @location(4) tangent: vec4<f32>,
    @if(VERTEX_COLORS)
    @location(5) color: vec4<f32>,
    @location(8) tex_idx: vec3<u32>,
}

struct CustomVertexOutput {
    @builtin(position) position: vec4<f32>,
    // MSAA may shade a covered sample whose pixel centre is outside this
    // primitive. Face-keyed lighting must use a position on the actual face.
    @location(0) @interpolate(perspective, centroid) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @if(VERTEX_UVS)
    @location(2) uv: vec2<f32>,
    @if(VERTEX_TANGENTS)
    @location(4) world_tangent: vec4<f32>,
    @if(VERTEX_COLORS)
    @location(5) color: vec4<f32>,
    @if(VERTEX_OUTPUT_INSTANCE_INDEX)
    @location(6) @interpolate(flat) instance_index: u32,
    // Integer varyings cannot interpolate.
    @location(8) @interpolate(flat) tex_idx: vec3<u32>,
}

@vertex
fn vertex(vertex: Vertex) -> CustomVertexOutput {
    var out: CustomVertexOutput;
    let model = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_normal = mesh_functions::mesh_normal_local_to_world(vertex.normal, vertex.instance_index);
    out.world_position = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(vertex.position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
    @if(VERTEX_UVS)
    out.uv = vertex.uv;
    @if(VERTEX_TANGENTS)
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(model, vertex.tangent, vertex.instance_index);
    out.color = vertex.color;
    @if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex.instance_index;
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
    let footprint = pixel_footprint(in.world_position.xyz);

    var tex_face = 0;
    if in.world_normal.y == 0.0 {
        tex_face = 1;
    } else if in.world_normal.y < 0.0 {
        tex_face = 2;
    }

    let albedo = textureSample(mat_array_texture, mat_array_texture_sampler, in.uv, in.tex_idx[tex_face]);
    pbr_input.material.base_color = albedo * in.color;
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color,
    );

@if(PREPASS_PIPELINE)
    return deferred_output(standard_in, pbr_input);
@else {
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input, footprint);
    let normal = normalize(in.world_normal);
    let gi = indirect_light(in.position.xy, in.world_position.xyz, normal);
    // Validation-only identity coverage, before tone mapping or fog. Selected
    // receiver counters cannot detect visible surfaces absent from the grid.
    if debug_mode() == 6u {
        let valid = face_in_geometry(in.world_position.xyz, normal);
        // Independent channels survive MSAA resolve: red is missing indirect
        // support, green is a lookup outside exposed transport geometry.
        // A completely covered indoor view legitimately has a zero failure
        // mask. A tiny blue witness proves the diagnostic shader ran, without
        // relaxing black-frame checks for ordinary lighting captures. The
        // analyzer excludes blue (sky, mixed silhouettes and this witness).
        let witness = select(0.0, 0.5, all(in.position.xy < vec2(4.0)));
        out.color = vec4(1.0 - gi.coverage.g, select(1.0, 0.0, valid), witness, 1.0);
        return out;
    }
    if debug_mode() == 2u {
        out.color = vec4(gi.coverage, 1.0);
        return out;
    }
    out.color = vec4<f32>(
        out.color.rgb + pbr_input.material.base_color.rgb * (1.0 - pbr_input.material.metallic) * gi.light
            + emission(in.world_position.xyz, normal),
        out.color.a,
    );
    if debug_mode() != 0u {
        var diagnostic_light = gi.light;
        // Component captures isolate motion-induced lighting changes from
        // textured appearance. Normal presentation never enters this branch.
        if debug_mode() >= 4u { diagnostic_light = vec3(0.0); }
        if debug_mode() != 3u && voxel_sky() {
            var light = sunlight(in.world_position.xyz, normal, footprint);
            if debug_mode() == 5u {
                light = sunlight_reference(in.world_position.xyz, normal);
            }
            diagnostic_light += light * (10000.0 / 3.14159265359) * mesh_view_bindings::view.exposure;
        }
        out.color = vec4(diagnostic_light, 1.0);
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
}
