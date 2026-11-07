// Bevy's `apply_pbr_lighting` with voxel GI's sun: `bevy_voxel_gi::pbr`.
//
// Bevy 0.20's WESL has no `override fn`, so this is a verbatim copy of
// `apply_pbr_lighting` from bevy_pbr 0.20.1 (src/render/pbr_functions.wesl,
// MIT OR Apache-2.0) with one change, between the `VOXEL GI` comments: while
// GI's sky is on, GI's cached voxel sun replaces Bevy's first directional
// light and its shadow map. Every other light, probe and material feature is
// Bevy's own. Re-copy it when Bevy changes the function (see
// docs/releasing.md); a test composes it against Bevy's modules.
//
// `footprint` is the world size of a pixel at the fragment (from `fwidth`),
// which GI's sun uses to pick its detail; `pixel_footprint` computes it.

import bevy_pbr::{
    render::{
        pbr_types,
        mesh_view_bindings as view_bindings,
        mesh_view_types,
        pbr_lighting as lighting,
        pbr_lighting::{LAYER_BASE, LAYER_CLEARCOAT},
        clustered_forward as clustering,
        shadows,
        pbr_ambient as ambient,
        pbr_functions::{calculate_diffuse_color, calculate_F0, calculate_F0_dielectric},
        mesh_types::{
            MESH_FLAGS_SHADOW_RECEIVER_BIT,
            MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT,
        },
    },
};
import bevy_voxel_gi::presentation::{sunlight, voxel_sky};

@if(STANDARD_MATERIAL_SPECULAR_TRANSMISSION) import bevy_pbr::transmission;
@if(IRRADIANCE_VOLUME) import bevy_pbr::light_probe::irradiance_volume;
@if(ENVIRONMENT_MAP) import bevy_pbr::light_probe::environment_map;
@if(STANDARD_MATERIAL_ANISOTROPY) import bevy_pbr::render::pbr_functions::bend_normal_for_anisotropy;
@if(CONTACT_SHADOWS && DEPTH_PREPASS) import bevy_pbr::render::pbr_functions::calculate_contact_shadow;

/// The world size of a pixel at `world_position`, for `apply_pbr_lighting`.
fn pixel_footprint(world_position: vec3<f32>) -> f32 {
    let world_per_pixel = fwidth(world_position);
    return max(world_per_pixel.x, max(world_per_pixel.y, world_per_pixel.z));
}

// GI's sun at a lighting input: cached voxel visibility (sun illuminance /
// 10,000 in transport units), diffuse only.
fn voxel_sun(input: ptr<function, lighting::LightingInput>, footprint: f32) -> vec3<f32> {
    let sky = sunlight((*input).P, (*input).layers[LAYER_BASE].N, footprint);
    return (*input).diffuse_color * sky * (10000.0 / 3.14159265359);
}

@if(!PREPASS_FRAGMENT)
fn apply_pbr_lighting(
    in: pbr_types::PbrInput,
    footprint: f32,
) -> vec4<f32> {
    var output_color: vec4<f32> = in.material.base_color;

    let emissive = in.material.emissive;

    // calculate non-linear roughness from linear perceptualRoughness
    let metallic = in.material.metallic;
    let perceptual_roughness = in.material.perceptual_roughness;
    let roughness = lighting::perceptualRoughnessToRoughness(perceptual_roughness);
    let ior = in.material.ior;
    let thickness = in.material.thickness;
    let reflectance = in.material.reflectance;
    let diffuse_transmission = in.material.diffuse_transmission;
    let specular_transmission = in.material.specular_transmission;

    let specular_transmissive_color = specular_transmission * in.material.base_color.rgb;

    let diffuse_occlusion = in.diffuse_occlusion;
    let specular_occlusion = in.specular_occlusion;

    // Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
    let NdotV = max(dot(in.N, in.V), 0.0001);
    let R = reflect(-in.V, in.N);

    // Do the above calculations again for the clearcoat layer. Remember that
    // the clearcoat can have its own roughness and its own normal.
    @if(STANDARD_MATERIAL_CLEARCOAT)
    let clearcoat = in.material.clearcoat;
    @if(STANDARD_MATERIAL_CLEARCOAT)
    let clearcoat_perceptual_roughness = in.material.clearcoat_perceptual_roughness;
    @if(STANDARD_MATERIAL_CLEARCOAT)
    let clearcoat_roughness = lighting::perceptualRoughnessToRoughness(clearcoat_perceptual_roughness);
    @if(STANDARD_MATERIAL_CLEARCOAT)
    let clearcoat_N = in.clearcoat_N;
    @if(STANDARD_MATERIAL_CLEARCOAT)
    let clearcoat_NdotV = max(dot(clearcoat_N, in.V), 0.0001);
    @if(STANDARD_MATERIAL_CLEARCOAT)
    let clearcoat_R = reflect(-in.V, clearcoat_N);

    let diffuse_color = calculate_diffuse_color(
        output_color.rgb,
        metallic,
        specular_transmission,
        diffuse_transmission
    );

    // Diffuse transmissive strength is inversely related to metallicity and specular transmission, but directly related to diffuse transmission
    let diffuse_transmissive_color = output_color.rgb * (1.0 - metallic) * (1.0 - specular_transmission) * diffuse_transmission;

    // Calculate the world position of the second Lambertian lobe used for diffuse transmission, by subtracting material thickness
    let diffuse_transmissive_lobe_world_position = in.world_position - vec4<f32>(in.world_normal, 0.0) * thickness;

    let F0 = calculate_F0(output_color.rgb, metallic, reflectance);
    let F_ab = lighting::F_AB(perceptual_roughness, NdotV);

    var direct_light: vec3<f32> = vec3<f32>(0.0);

    // Transmitted Light (Specular and Diffuse)
    var transmitted_light: vec3<f32> = vec3<f32>(0.0);

    // Pack all the values into a structure.
    var lighting_input: lighting::LightingInput;
    lighting_input.layers[LAYER_BASE].NdotV = NdotV;
    lighting_input.layers[LAYER_BASE].N = in.N;
    lighting_input.layers[LAYER_BASE].R = R;
    lighting_input.layers[LAYER_BASE].perceptual_roughness = perceptual_roughness;
    lighting_input.layers[LAYER_BASE].roughness = roughness;
    lighting_input.P = in.world_position.xyz;
    lighting_input.V = in.V;
    lighting_input.diffuse_color = diffuse_color;
    lighting_input.metallic = metallic;
    lighting_input.F0_dielectric = calculate_F0_dielectric(reflectance);
    lighting_input.F0_metallic = output_color.rgb;
    lighting_input.F_ab = F_ab;
@if(STANDARD_MATERIAL_CLEARCOAT) {
    lighting_input.layers[LAYER_CLEARCOAT].NdotV = clearcoat_NdotV;
    lighting_input.layers[LAYER_CLEARCOAT].N = clearcoat_N;
    lighting_input.layers[LAYER_CLEARCOAT].R = clearcoat_R;
    lighting_input.layers[LAYER_CLEARCOAT].perceptual_roughness = clearcoat_perceptual_roughness;
    lighting_input.layers[LAYER_CLEARCOAT].roughness = clearcoat_roughness;
    lighting_input.clearcoat_strength = clearcoat;
}
@if(STANDARD_MATERIAL_ANISOTROPY) {
    lighting_input.anisotropy = in.anisotropy_strength;
    lighting_input.Ta = in.anisotropy_T;
    lighting_input.Ba = in.anisotropy_B;
}

    // And do the same for transmissive if we need to.
    @if(STANDARD_MATERIAL_DIFFUSE_TRANSMISSION)
    var transmissive_lighting_input: lighting::LightingInput;
@if(STANDARD_MATERIAL_DIFFUSE_TRANSMISSION) {
    transmissive_lighting_input.layers[LAYER_BASE].NdotV = 1.0;
    transmissive_lighting_input.layers[LAYER_BASE].N = -in.N;
    transmissive_lighting_input.layers[LAYER_BASE].R = vec3(0.0);
    transmissive_lighting_input.layers[LAYER_BASE].perceptual_roughness = 1.0;
    transmissive_lighting_input.layers[LAYER_BASE].roughness = 1.0;
    transmissive_lighting_input.P = diffuse_transmissive_lobe_world_position.xyz;
    transmissive_lighting_input.V = -in.V;
    transmissive_lighting_input.diffuse_color = diffuse_transmissive_color;
    transmissive_lighting_input.metallic = 0.0;
    transmissive_lighting_input.F0_dielectric = vec3(0.0);
    transmissive_lighting_input.F0_metallic = vec3(0.0);
    transmissive_lighting_input.F_ab = vec2(0.1);
@if(STANDARD_MATERIAL_CLEARCOAT) {
    transmissive_lighting_input.layers[LAYER_CLEARCOAT].NdotV = 0.0;
    transmissive_lighting_input.layers[LAYER_CLEARCOAT].N = vec3(0.0);
    transmissive_lighting_input.layers[LAYER_CLEARCOAT].R = vec3(0.0);
    transmissive_lighting_input.layers[LAYER_CLEARCOAT].perceptual_roughness = 0.0;
    transmissive_lighting_input.layers[LAYER_CLEARCOAT].roughness = 0.0;
    transmissive_lighting_input.clearcoat_strength = 0.0;
}
@if(STANDARD_MATERIAL_ANISOTROPY) {
    transmissive_lighting_input.anisotropy = in.anisotropy_strength;
    transmissive_lighting_input.Ta = in.anisotropy_T;
    transmissive_lighting_input.Ba = in.anisotropy_B;
}
}

    let view_z = dot(vec4<f32>(
        view_bindings::view.view_from_world[0].z,
        view_bindings::view.view_from_world[1].z,
        view_bindings::view.view_from_world[2].z,
        view_bindings::view.view_from_world[3].z
    ), in.world_position);
    let cluster_index = clustering::view_fragment_cluster_index(in.frag_coord.xy, view_z, in.is_orthographic);
    var clusterable_object_index_ranges =
        clustering::unpack_clusterable_object_index_ranges(cluster_index);

    @if(CONTACT_SHADOWS)
    let contact_shadow_steps = view_bindings::contact_shadows_settings.linear_steps;
    @if(CONTACT_SHADOWS)
    let contact_shadow_enabled = contact_shadow_steps > 0u;

    // Point lights (direct)
    for (var i: u32 = clusterable_object_index_ranges.first_point_light_index_offset;
            i < clusterable_object_index_ranges.first_spot_light_index_offset;
            i = i + 1u) {
        let light_id = clustering::get_clusterable_object_id(i);

        // If we're lightmapped, disable diffuse contribution from the light if
        // requested, to avoid double-counting light.
        @if(LIGHTMAP)
        let enable_diffuse =
            (view_bindings::clustered_lights.data[light_id].flags &
                mesh_view_types::POINT_LIGHT_FLAGS_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE_BIT) != 0u;
        @else   // LIGHTMAP
        let enable_diffuse = true;

        var shadow: f32 = 1.0;
        if ((in.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u
                && (view_bindings::clustered_lights.data[light_id].flags & mesh_view_types::POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
            shadow = shadows::fetch_point_shadow(light_id, in.world_position, in.world_normal, in.frag_coord.xy);
        }

        @if(CONTACT_SHADOWS && DEPTH_PREPASS)
        if contact_shadow_enabled && (in.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u && shadow > 0.0 &&
                (view_bindings::clustered_lights.data[light_id].flags &
                    mesh_view_types::POINT_LIGHT_FLAGS_CONTACT_SHADOWS_ENABLED_BIT) != 0u {
            let L = normalize(view_bindings::clustered_lights.data[light_id].position_radius.xyz - in.world_position.xyz);
            shadow *= calculate_contact_shadow(in.world_position.xyz, in.frag_coord.xy, L, contact_shadow_steps);
        }

        let light_contrib = lighting::point_light(light_id, &lighting_input, enable_diffuse, true);
        direct_light += light_contrib * shadow;

@if(STANDARD_MATERIAL_DIFFUSE_TRANSMISSION) {
        // NOTE: We use the diffuse transmissive color, the second Lambertian lobe's calculated
        // world position, inverted normal and view vectors, and the following simplified
        // values for a fully diffuse transmitted light contribution approximation:
        //
        // roughness = 1.0;
        // NdotV = 1.0;
        // R = vec3<f32>(0.0) // doesn't really matter
        // F_ab = vec2<f32>(0.1)
        // F0 = vec3<f32>(0.0)
        var transmitted_shadow: f32 = 1.0;
        if ((in.flags & (MESH_FLAGS_SHADOW_RECEIVER_BIT | MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT)) == (MESH_FLAGS_SHADOW_RECEIVER_BIT | MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT)
                && (view_bindings::clustered_lights.data[light_id].flags & mesh_view_types::POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
            transmitted_shadow = shadows::fetch_point_shadow(light_id, diffuse_transmissive_lobe_world_position, -in.world_normal, in.frag_coord.xy);
        }

        let transmitted_light_contrib =
            lighting::point_light(light_id, &transmissive_lighting_input, enable_diffuse, true);
        transmitted_light += transmitted_light_contrib * transmitted_shadow;
}
    }

    // Spot lights (direct)
    for (var i: u32 = clusterable_object_index_ranges.first_spot_light_index_offset;
            i < clusterable_object_index_ranges.first_rect_light_index_offset;
            i = i + 1u) {
        let light_id = clustering::get_clusterable_object_id(i);

        // If we're lightmapped, disable diffuse contribution from the light if
        // requested, to avoid double-counting light.
        @if(LIGHTMAP)
        let enable_diffuse =
            (view_bindings::clustered_lights.data[light_id].flags &
                mesh_view_types::POINT_LIGHT_FLAGS_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE_BIT) != 0u;
        @else   // LIGHTMAP
        let enable_diffuse = true;

        var shadow: f32 = 1.0;
        if ((in.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u
                && (view_bindings::clustered_lights.data[light_id].flags &
                    mesh_view_types::POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
            shadow = shadows::fetch_spot_shadow(
                light_id,
                in.world_position,
                in.world_normal,
                view_bindings::clustered_lights.data[light_id].shadow_map_near_z,
                in.frag_coord.xy,
            );
        }

        @if(CONTACT_SHADOWS && DEPTH_PREPASS)
        if contact_shadow_enabled && (in.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u && shadow > 0.0 &&
                (view_bindings::clustered_lights.data[light_id].flags &
                    mesh_view_types::POINT_LIGHT_FLAGS_CONTACT_SHADOWS_ENABLED_BIT) != 0u {
            let L = normalize(view_bindings::clustered_lights.data[light_id].position_radius.xyz - in.world_position.xyz);
            shadow *= calculate_contact_shadow(in.world_position.xyz, in.frag_coord.xy, L, contact_shadow_steps);
        }

        let light_contrib = lighting::spot_light(light_id, &lighting_input, enable_diffuse);
        direct_light += light_contrib * shadow;

@if(STANDARD_MATERIAL_DIFFUSE_TRANSMISSION) {
        // NOTE: We use the diffuse transmissive color, the second Lambertian lobe's calculated
        // world position, inverted normal and view vectors, and the following simplified
        // values for a fully diffuse transmitted light contribution approximation:
        //
        // roughness = 1.0;
        // NdotV = 1.0;
        // R = vec3<f32>(0.0) // doesn't really matter
        // F_ab = vec2<f32>(0.1)
        // F0 = vec3<f32>(0.0)
        var transmitted_shadow: f32 = 1.0;
        if ((in.flags & (MESH_FLAGS_SHADOW_RECEIVER_BIT | MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT)) == (MESH_FLAGS_SHADOW_RECEIVER_BIT | MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT)
                && (view_bindings::clustered_lights.data[light_id].flags & mesh_view_types::POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
            transmitted_shadow = shadows::fetch_spot_shadow(
                light_id,
                diffuse_transmissive_lobe_world_position,
                -in.world_normal,
                view_bindings::clustered_lights.data[light_id].shadow_map_near_z,
                in.frag_coord.xy,
            );
        }

        let transmitted_light_contrib =
            lighting::spot_light(light_id, &transmissive_lighting_input, enable_diffuse);
        transmitted_light += transmitted_light_contrib * transmitted_shadow;
}
    }

    // directional lights (direct)
    let n_directional_lights = view_bindings::lights.n_directional_lights;
    for (var i: u32 = 0u; i < n_directional_lights; i = i + 1u) {
        // check if this light should be skipped, which occurs if this light does not intersect with the view
        // note point and spot lights aren't skippable, as the relevant lights are filtered in `assign_lights_to_clusters`
        let light = &view_bindings::lights.directional_lights[i];

        // If we're lightmapped, disable diffuse contribution from the light if
        // requested, to avoid double-counting light.
        @if(LIGHTMAP)
        let enable_diffuse =
            ((*light).flags &
                mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE_BIT) !=
                0u;
        @else   // LIGHTMAP
        let enable_diffuse = true;

        // VOXEL GI: GI's cached voxel sun replaces the first directional
        // light, unshadowed by Bevy; the others contribute nothing.
        if voxel_sky() {
            if i == 0u && enable_diffuse {
                direct_light += voxel_sun(&lighting_input, footprint);
                @if(STANDARD_MATERIAL_DIFFUSE_TRANSMISSION)
                transmitted_light += voxel_sun(&transmissive_lighting_input, footprint);
            }
            continue;
        }
        // VOXEL GI: end.

        var shadow: f32 = 1.0;
        if ((in.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u
                && (view_bindings::lights.directional_lights[i].flags & mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
            shadow = shadows::fetch_directional_shadow(i, in.world_position, in.world_normal, view_z, in.frag_coord.xy);
            shadow *= in.directional_shadow_factor;
        }

        @if(CONTACT_SHADOWS && DEPTH_PREPASS)
        if contact_shadow_enabled && (in.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u && shadow > 0.0 &&
                (view_bindings::lights.directional_lights[i].flags &
                    mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_CONTACT_SHADOWS_ENABLED_BIT) != 0u {
            let L = view_bindings::lights.directional_lights[i].direction_to_light;
            shadow *= calculate_contact_shadow(in.world_position.xyz, in.frag_coord.xy, L, contact_shadow_steps);
        }

        var light_contrib = lighting::directional_light(i, &lighting_input, enable_diffuse);

        @if(DIRECTIONAL_LIGHT_SHADOW_MAP_DEBUG_CASCADES)
        light_contrib = shadows::cascade_debug_visualization(light_contrib, i, view_z);
        direct_light += light_contrib * shadow;

@if(STANDARD_MATERIAL_DIFFUSE_TRANSMISSION) {
        // NOTE: We use the diffuse transmissive color, the second Lambertian lobe's calculated
        // world position, inverted normal and view vectors, and the following simplified
        // values for a fully diffuse transmitted light contribution approximation:
        //
        // roughness = 1.0;
        // NdotV = 1.0;
        // R = vec3<f32>(0.0) // doesn't really matter
        // F_ab = vec2<f32>(0.1)
        // F0 = vec3<f32>(0.0)
        var transmitted_shadow: f32 = 1.0;
        if ((in.flags & (MESH_FLAGS_SHADOW_RECEIVER_BIT | MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT)) == (MESH_FLAGS_SHADOW_RECEIVER_BIT | MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT)
                && (view_bindings::lights.directional_lights[i].flags & mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
            transmitted_shadow = shadows::fetch_directional_shadow(i, diffuse_transmissive_lobe_world_position, -in.world_normal, view_z, in.frag_coord.xy);
        }

        let transmitted_light_contrib =
            lighting::directional_light(i, &transmissive_lighting_input, enable_diffuse);
        transmitted_light += transmitted_light_contrib * transmitted_shadow;
}
    }

@if(AREA_LIGHT_LUTS) {
    // Rect lights
    // This combines a clustered path if AVAILABLE_STORAGE_BUFFER_BINDINGS >= 3
    // and a non-clustered fallback for WebGL2.
@if(AVAILABLE_STORAGE_BUFFER_BINDINGS__GE_3)
    let rect_light_start = clusterable_object_index_ranges.first_rect_light_index_offset;
@if(AVAILABLE_STORAGE_BUFFER_BINDINGS__GE_3)
    let rect_light_end = clusterable_object_index_ranges.first_reflection_probe_index_offset;
@if(!AVAILABLE_STORAGE_BUFFER_BINDINGS__GE_3)
    let rect_light_start = 0u;
@if(!AVAILABLE_STORAGE_BUFFER_BINDINGS__GE_3)
    let rect_light_end = view_bindings::lights.n_rect_lights;
    for (var i: u32 = rect_light_start; i < rect_light_end; i = i + 1u) {
@if(AVAILABLE_STORAGE_BUFFER_BINDINGS__GE_3)
        var light = lighting::unpack_clustered_rect_light(clustering::get_clusterable_object_id(i));
@if(!AVAILABLE_STORAGE_BUFFER_BINDINGS__GE_3)
        var light = view_bindings::lights.rect_lights[i];
        let enable_diffuse = true;
        let light_contrib = lighting::rect_light(&light, &lighting_input, enable_diffuse);
        direct_light += light_contrib;

    @if(STANDARD_MATERIAL_DIFFUSE_TRANSMISSION) {
        let transmitted_light_contrib =
            lighting::rect_light(&light, &transmissive_lighting_input, enable_diffuse);
        transmitted_light += transmitted_light_contrib;
    }
    }
}

    @if(STANDARD_MATERIAL_DIFFUSE_TRANSMISSION)
    // NOTE: We use the diffuse transmissive color, the second Lambertian lobe's calculated
    // world position, inverted normal and view vectors, and the following simplified
    // values for a fully diffuse transmitted light contribution approximation:
    //
    // perceptual_roughness = 1.0;
    // NdotV = 1.0;
    // F0 = vec3<f32>(0.0)
    // diffuse_occlusion = vec3<f32>(1.0)
    transmitted_light += ambient::ambient_light(diffuse_transmissive_lobe_world_position, -in.N, -in.V, 1.0, diffuse_transmissive_color, vec3<f32>(0.0), 1.0, vec3<f32>(1.0));

    // Diffuse indirect lighting can come from a variety of sources. The
    // priority goes like this:
    //
    // 1. Lightmap (highest)
    // 2. Irradiance volume
    // 3. Environment map (lowest)
    //
    // When we find a source of diffuse indirect lighting, we stop accumulating
    // any more diffuse indirect light. This avoids double-counting if, for
    // example, both lightmaps and irradiance volumes are present.

    var indirect_light = vec3(0.0f);
    var found_diffuse_indirect = false;

@if(LIGHTMAP) {
    indirect_light += in.lightmap_light * diffuse_color;
    found_diffuse_indirect = true;
}

    @if(IRRADIANCE_VOLUME)
    // Irradiance volume light (indirect)
    if (!found_diffuse_indirect) {
        let irradiance_volume_light = irradiance_volume::irradiance_volume_light(
            in.world_position.xyz,
            in.N,
            &clusterable_object_index_ranges,
        );
        indirect_light += irradiance_volume_light * diffuse_color * diffuse_occlusion;
        found_diffuse_indirect = true;
    }

    // Environment map light (indirect)
@if(ENVIRONMENT_MAP) {
    // If screen space reflections are going to be used for this material, only
    // accumulate the diffuse part of the environment map light. The SSR shader
    // will accumulate the specular part (including the environment map fallback
    // if SSR misses).
    @if(SCREEN_SPACE_REFLECTIONS)
    let use_ssr = perceptual_roughness <= view_bindings::ssr_settings.max_perceptual_roughness
        && perceptual_roughness >= view_bindings::ssr_settings.min_perceptual_roughness;
    @else   // SCREEN_SPACE_REFLECTIONS
    let use_ssr = false;

    @if(STANDARD_MATERIAL_ANISOTROPY)
    var bent_normal_lighting_input = lighting_input;
    @if(STANDARD_MATERIAL_ANISOTROPY)
    bend_normal_for_anisotropy(&bent_normal_lighting_input);
    @if(STANDARD_MATERIAL_ANISOTROPY)
    let environment_map_lighting_input = &bent_normal_lighting_input;
    @else   // STANDARD_MATERIAL_ANISOTROPY
    let environment_map_lighting_input = &lighting_input;

    let environment_light = environment_map::environment_map_light(
        environment_map_lighting_input,
        &clusterable_object_index_ranges,
        found_diffuse_indirect,
    );

    indirect_light += environment_light.diffuse * diffuse_occlusion;
    if (!use_ssr) {
        indirect_light += environment_light.specular * specular_occlusion;
    }
}

    // Ambient light (indirect)
    // If we are lightmapped, disable the ambient contribution if requested.
    // This is to avoid double-counting ambient light. (It might be part of the lightmap)
    @if(LIGHTMAP)
    let enable_ambient = (view_bindings::lights.ambient_light_flags & mesh_view_types::AMBIENT_LIGHT_FLAGS_AFFECTS_LIGHTMAPPED_MESHES_BIT) != 0u;
    @else   // LIGHTMAP
    let enable_ambient = true;
    if (enable_ambient) {
        indirect_light += ambient::ambient_light(in.world_position, in.N, in.V, NdotV, diffuse_color, F0, perceptual_roughness, diffuse_occlusion);
    }

    // we'll use the specular component of the transmitted environment
    // light in the call to `specular_transmissive_light()` below
    var specular_transmitted_environment_light = vec3<f32>(0.0);

@if(ENVIRONMENT_MAP) {

@if(STANDARD_MATERIAL_DIFFUSE_OR_SPECULAR_TRANSMISSION) {
    // NOTE: We use the diffuse transmissive color, inverted normal and view vectors,
    // and the following simplified values for the transmitted environment light contribution
    // approximation:
    //
    // diffuse_color = vec3<f32>(1.0) // later we use `diffuse_transmissive_color` and `specular_transmissive_color`
    // NdotV = 1.0;
    // R = T // see definition below
    // F0 = vec3<f32>(1.0) (using F0_dielectric = 1, F0_metallic = 0 and metallic = 0)
    // diffuse_occlusion = 1.0
    //
    // (This one is slightly different from the other light types above, because the environment
    // map light returns both diffuse and specular components separately, and we want to use both)

    let T = -normalize(
        in.V + // start with view vector at entry point
        refract(in.V, -in.N, 1.0 / ior) * thickness // add refracted vector scaled by thickness, towards exit point
    ); // normalize to find exit point view vector

    var transmissive_environment_light_input: lighting::LightingInput;
    transmissive_environment_light_input.diffuse_color = vec3(1.0);
    transmissive_environment_light_input.layers[LAYER_BASE].NdotV = 1.0;
    transmissive_environment_light_input.P = in.world_position.xyz;
    transmissive_environment_light_input.layers[LAYER_BASE].N = -in.N;
    transmissive_environment_light_input.V = in.V;
    transmissive_environment_light_input.layers[LAYER_BASE].R = T;
    transmissive_environment_light_input.layers[LAYER_BASE].perceptual_roughness = perceptual_roughness;
    transmissive_environment_light_input.layers[LAYER_BASE].roughness = roughness;
    transmissive_environment_light_input.metallic = 0.0;
    transmissive_environment_light_input.F0_dielectric = vec3<f32>(1.0);
    transmissive_environment_light_input.F0_metallic = vec3<f32>(0.0);
    transmissive_environment_light_input.F_ab = vec2(0.1);
@if(STANDARD_MATERIAL_CLEARCOAT) {
    // No clearcoat.
    transmissive_environment_light_input.clearcoat_strength = 0.0;
    transmissive_environment_light_input.layers[LAYER_CLEARCOAT].NdotV = 0.0;
    transmissive_environment_light_input.layers[LAYER_CLEARCOAT].N = in.N;
    transmissive_environment_light_input.layers[LAYER_CLEARCOAT].R = vec3(0.0);
    transmissive_environment_light_input.layers[LAYER_CLEARCOAT].perceptual_roughness = 0.0;
    transmissive_environment_light_input.layers[LAYER_CLEARCOAT].roughness = 0.0;
}

    let transmitted_environment_light = environment_map::environment_map_light(
        &transmissive_environment_light_input,
        &clusterable_object_index_ranges,
        false,
    );

    @if(STANDARD_MATERIAL_DIFFUSE_TRANSMISSION)
    transmitted_light += transmitted_environment_light.diffuse * diffuse_transmissive_color;
    @if(STANDARD_MATERIAL_SPECULAR_TRANSMISSION)
    specular_transmitted_environment_light = transmitted_environment_light.specular * specular_transmissive_color;

}

}

    var emissive_light = emissive.rgb * output_color.a;

    // "The clearcoat layer is on top of emission in the layering stack.
    // Consequently, the emission is darkened by the Fresnel term."
    //
    // <https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_clearcoat/README.md#emission>
    @if(STANDARD_MATERIAL_CLEARCOAT) {
    let clearcoat_fresnel = lighting::F_Schlick(0.04, 1.0, clearcoat_NdotV);
    emissive_light = emissive_light * (1.0 - clearcoat * clearcoat_fresnel);
    }

    emissive_light = emissive_light * mix(1.0, view_bindings::view.exposure, emissive.a);

@if(STANDARD_MATERIAL_SPECULAR_TRANSMISSION) {
    transmitted_light += transmission::specular_transmissive_light(in.world_position, in.frag_coord.xyz, view_z, in.N, in.V, F0, ior, thickness, perceptual_roughness, specular_transmissive_color, specular_transmitted_environment_light).rgb;

    if (in.material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_ATTENUATION_ENABLED_BIT) != 0u
        && in.material.attenuation_distance != 0.0 {
        // Compute light attenuation using Beer's law.
        let transmittance = pow(
            in.material.attenuation_color.rgb,
            vec3<f32>(thickness / in.material.attenuation_distance),
        );
        transmitted_light *= transmittance;
    }
}

    // Total light
    output_color = vec4<f32>(
        (view_bindings::view.exposure * (transmitted_light + direct_light + indirect_light)) + emissive_light,
        output_color.a
    );

    output_color = clustering::cluster_debug_visualization(
        output_color,
        view_z,
        in.is_orthographic,
        clusterable_object_index_ranges,
        cluster_index,
    );

    return output_color;
}
