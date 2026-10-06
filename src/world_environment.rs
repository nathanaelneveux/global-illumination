use bevy::light::{CascadeShadowConfigBuilder, SimulationLightSystems};
use bevy::prelude::*;
use bevy_voxel_world::rendering::VoxelWorldMaterialHandle;

use crate::terrain::TERRAIN_VIEW_DISTANCE;
use crate::voxel_gi::render::VoxelTransportMaterial;

#[derive(Component)]
struct DemoSun;

pub struct WorldEnvironmentPlugin;

impl Plugin for WorldEnvironmentPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(sky_color()))
            .add_systems(Startup, setup_lighting)
            .add_systems(
                PostUpdate,
                synchronize_sun_shadows
                    .before(SimulationLightSystems::UpdateDirectionalLightCascades),
            );
    }
}

pub fn sky_color() -> Color {
    Color::srgb(0.53, 0.74, 0.94)
}

fn setup_lighting(mut commands: Commands) {
    let cascade_shadow_config = CascadeShadowConfigBuilder {
        maximum_distance: TERRAIN_VIEW_DISTANCE,
        ..default()
    }
    .build();

    commands.spawn((
        Name::new("Sun"),
        DemoSun,
        DirectionalLight {
            color: Color::srgb(1.0, 0.96, 0.88),
            illuminance: 18_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::new(0.35, -0.8, 0.25), Vec3::Y),
        cascade_shadow_config,
    ));

    commands.insert_resource(GlobalAmbientLight::NONE);
}

/// Corner sunlight does not sample cascaded shadow maps. This is a demo policy,
/// not a GI-library side effect: only our sun is managed, and any non-voxel mesh
/// conservatively restores raster shadows. F7 takes effect before light culling
/// and render extraction in the same frame. Point-light shadows are untouched.
fn synchronize_sun_shadows(
    handle: Option<Res<VoxelWorldMaterialHandle<VoxelTransportMaterial>>>,
    materials: Res<Assets<VoxelTransportMaterial>>,
    other_meshes: Query<
        (),
        (
            With<Mesh3d>,
            Without<MeshMaterial3d<VoxelTransportMaterial>>,
        ),
    >,
    mut suns: Query<&mut DirectionalLight, With<DemoSun>>,
) {
    let corner_sun = handle
        .as_ref()
        .and_then(|handle| materials.get(&handle.handle))
        .is_some_and(|material| material.extension.settings.voxel_sky != 0);
    let required = !corner_sun || !other_meshes.is_empty();
    for mut sun in &mut suns {
        if sun.shadow_maps_enabled != required {
            sun.shadow_maps_enabled = required;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;

    #[test]
    fn shadows_follow_consumers_without_changing_other_lights() {
        let mut app = App::new();
        let mut materials = Assets::<VoxelTransportMaterial>::default();
        let material = materials.add(crate::voxel_gi::render::material_template(
            &crate::voxel_gi::render::VoxelTransportAssets {
                texture: default(),
                hash_entries: default(),
                probes: default(),
            },
        ));
        app.insert_resource(materials)
            .insert_resource(VoxelWorldMaterialHandle {
                handle: material.clone(),
            });
        let sun = app
            .world_mut()
            .spawn((
                DemoSun,
                DirectionalLight {
                    shadow_maps_enabled: true,
                    ..default()
                },
            ))
            .id();
        let unmanaged = app
            .world_mut()
            .spawn(DirectionalLight {
                shadow_maps_enabled: true,
                ..default()
            })
            .id();
        let update = |app: &mut App| {
            app.world_mut()
                .run_system_once(synchronize_sun_shadows)
                .unwrap();
            app.world()
                .get::<DirectionalLight>(sun)
                .unwrap()
                .shadow_maps_enabled
        };
        assert!(!update(&mut app));
        assert!(
            app.world()
                .get::<DirectionalLight>(unmanaged)
                .unwrap()
                .shadow_maps_enabled
        );
        app.world_mut()
            .resource_mut::<Assets<VoxelTransportMaterial>>()
            .get_mut(&material)
            .unwrap()
            .extension
            .settings
            .voxel_sky = 0;
        assert!(update(&mut app), "F7 must restore raster shadows");
        app.world_mut()
            .resource_mut::<Assets<VoxelTransportMaterial>>()
            .get_mut(&material)
            .unwrap()
            .extension
            .settings
            .voxel_sky = 1;
        let mesh = app.world_mut().spawn(Mesh3d::default()).id();
        assert!(update(&mut app), "other receivers still need shadow maps");
        app.world_mut().despawn(mesh);
        assert!(!update(&mut app));
        app.world_mut()
            .remove_resource::<VoxelWorldMaterialHandle<VoxelTransportMaterial>>();
        assert!(update(&mut app), "unknown material mode is conservative");
    }
}
