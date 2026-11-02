// `StandardMaterial` lit by voxel GI (`VoxelGiStandardMaterial`): Bevy's PBR
// with GI's indirect light and emission added, and GI's cached voxel sun in
// place of Bevy's shadowed sun. A starting point for host materials; see
// the BVW connector's material for one with its own vertex layout.
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}
#import bevy_pbr::{lighting, shadows, mesh_view_bindings}
#import bevy_voxel_gi::presentation::{indirect_light, sunlight, emission, voxel_sky, debug_mode}

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

// World size of a pixel at this fragment, set by `fragment` before lighting:
// GI's sunlight uses it (the lighting override has no derivatives).
var<private> fragment_footprint: f32 = 0.0;

// GI's sun replaces Bevy's directional light and shadow once per material:
// the cached voxel visibility (sun illuminance / 10,000 in transport units).
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
    return (*input).diffuse_color * sky * (10000.0 / 3.14159265359);
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

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let world_per_pixel = fwidth(in.world_position.xyz);
    fragment_footprint = max(world_per_pixel.x, max(world_per_pixel.y, world_per_pixel.z));
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
#ifdef PREPASS_PIPELINE
    return deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    let normal = normalize(in.world_normal);
    let gi = indirect_light(in.position.xy, in.world_position.xyz, normal);
    if debug_mode() == 2u {
        out.color = vec4(gi.coverage, 1.0);
        return out;
    }
    out.color = vec4(
        out.color.rgb + pbr_input.material.base_color.rgb * (1.0 - pbr_input.material.metallic) * gi.light
            + emission(in.world_position.xyz, normal),
        out.color.a,
    );
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
#endif
}
