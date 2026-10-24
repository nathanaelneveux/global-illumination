use bevy::light::{CascadeShadowConfigBuilder, SimulationLightSystems};
use bevy::prelude::*;

use crate::terrain::TERRAIN_VIEW_DISTANCE;
use voxel_gi_bvw::VoxelTransportMaterial;

#[derive(Component)]
struct DemoSun;

/// The demo sun's rotation in degrees about the world Z axis from the scene's
/// sun direction (the day-cycle validation profile drives it). Host policy,
/// not GI: transport sees an ordinary moving directional light.
#[derive(Resource, Clone, Copy, PartialEq, Default)]
pub struct SunAngle(pub f32);

pub struct WorldEnvironmentPlugin;

impl Plugin for WorldEnvironmentPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(sky_color()))
            .add_systems(Startup, setup_lighting)
            .add_systems(Update, rotate_sun)
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
    let preset = crate::scene_presets::ScenePreset::current();
    let cave = preset == crate::scene_presets::ScenePreset::Cave;
    let canyon = preset == crate::scene_presets::ScenePreset::Canyon;
    if cave {
        commands.insert_resource(voxel_gi::VoxelGiEnvironment(
            crate::cave_scene::ENVIRONMENT,
        ));
    }
    if canyon {
        commands.insert_resource(voxel_gi::VoxelGiEnvironment(
            crate::canyon_scene::ENVIRONMENT,
        ));
    }
    let cascade_shadow_config = CascadeShadowConfigBuilder {
        maximum_distance: TERRAIN_VIEW_DISTANCE,
        ..default()
    }
    .build();

    let mut sun = commands.spawn((
        Name::new("Sun"),
        DemoSun,
        DirectionalLight {
            color: if cave || canyon {
                Color::WHITE
            } else {
                Color::srgb(1.0, 0.96, 0.88)
            },
            illuminance: if cave {
                40_000.0
            } else if canyon {
                20_000.0
            } else {
                18_000.0
            },
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::IDENTITY.looking_at(-scene_sun_direction(), Vec3::Y),
        cascade_shadow_config,
    ));
    if cave || canyon {
        sun.insert(voxel_gi::EnergyEmitter {
            spectrum: Vec3::ZERO,
            range: 0.0,
        });
    }

    commands.insert_resource(GlobalAmbientLight::NONE);
}

/// Direction toward the scene's sun.
fn scene_sun_direction() -> Vec3 {
    match crate::scene_presets::ScenePreset::current() {
        crate::scene_presets::ScenePreset::Cave => crate::cave_scene::sun_direction(),
        crate::scene_presets::ScenePreset::Canyon => crate::canyon_scene::sun_direction(),
        crate::scene_presets::ScenePreset::Terrain => -Vec3::new(0.35, -0.8, 0.25),
    }
}

fn rotate_sun(angle: Option<Res<SunAngle>>, mut suns: Query<&mut Transform, With<DemoSun>>) {
    let Some(angle) = angle.filter(|angle| angle.is_changed()) else {
        return;
    };
    let direction = Quat::from_rotation_z(angle.0.to_radians()) * scene_sun_direction();
    for mut transform in &mut suns {
        *transform = Transform::IDENTITY.looking_at(-direction, Vec3::Y);
    }
}

/// Corner sunlight does not sample cascaded shadow maps. This is a demo policy,
/// not a GI-library side effect: only our sun is managed, and any non-voxel mesh
/// conservatively restores raster shadows. F7 takes effect before light culling
/// and render extraction in the same frame. Point-light shadows are untouched.
fn synchronize_sun_shadows(
    display: Res<voxel_gi::VoxelGiDisplay>,
    other_meshes: Query<
        (),
        (
            With<Mesh3d>,
            Without<MeshMaterial3d<VoxelTransportMaterial>>,
        ),
    >,
    mut suns: Query<&mut DirectionalLight, With<DemoSun>>,
) {
    let corner_sun = display.voxel_sun;
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
        app.init_resource::<voxel_gi::VoxelGiDisplay>();
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
        app.world_mut().resource_mut::<voxel_gi::VoxelGiDisplay>().voxel_sun = false;
        assert!(update(&mut app), "F7 must restore raster shadows");
        app.world_mut().resource_mut::<voxel_gi::VoxelGiDisplay>().voxel_sun = true;
        let mesh = app.world_mut().spawn(Mesh3d::default()).id();
        assert!(update(&mut app), "other receivers still need shadow maps");
        app.world_mut().despawn(mesh);
        assert!(!update(&mut app));
    }
}
