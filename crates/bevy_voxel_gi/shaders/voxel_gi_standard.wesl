// `StandardMaterial` lit by voxel GI (`VoxelGiStandardMaterial`): Bevy's PBR
// with GI's indirect light and emission added, and GI's cached voxel sun in
// place of Bevy's shadowed sun (`bevy_voxel_gi::pbr::apply_pbr_lighting`).
// A starting point for host materials; see the BVW connector's material for
// one with its own vertex layout.
import bevy_pbr::render::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
};
import bevy_voxel_gi::presentation::{indirect_light, emission, debug_mode};

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

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color,
    );
@if(PREPASS_PIPELINE)
    return deferred_output(in, pbr_input);
@else {
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input, pixel_footprint(in.world_position.xyz));
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
}
}
