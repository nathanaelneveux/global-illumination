#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_functions,
    view_transformations::position_world_to_clip
}
#import bevy_pbr::pbr_bindings
#import bevy_pbr::{lighting, shadows, mesh_view_bindings}
#import bevy_render::instance_index::get_instance_index
#import bevy_voxel_gi::presentation::{
    indirect_light, sunlight, sunlight_reference, emission, face_in_geometry, voxel_sky, debug_mode,
}

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

// World units per pixel at this fragment, set by `fragment` before
// lighting: GI's sunlight uses it (the lighting override has no derivatives).
var<private> fragment_footprint: f32 = 0.0;

// Naga-oil overrides are scoped to this material's shader. Other materials,
// point lights remain untouched. Sum the cached
// directional sources once, not once per Bevy directional-light loop entry.
override fn lighting::directional_light(
    light_id: u32,
    input: ptr<function, lighting::LightingInput>,
    enable_diffuse: bool,
) -> vec3<f32> {
    if !voxel_sky() {
        return lighting::directional_light(light_id, input, enable_diffuse);
    }
    if light_id != 0u || !enable_diffuse { return vec3(0.0); }
    let sky = sunlight((*input).P, (*input).layers[lighting::LAYER_BASE].N, fragment_footprint);
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
    if voxel_sky() { return 1.0; }
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
            var light = sunlight(in.world_position.xyz, normal, fragment_footprint);
            if debug_mode() == 5u {
                light = sunlight_reference(in.world_position.xyz, normal);
            }
            diagnostic_light += light * (10000.0 / 3.14159265359) * mesh_view_bindings::view.exposure;
        }
        out.color = vec4(diagnostic_light, 1.0);
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
#endif
}
