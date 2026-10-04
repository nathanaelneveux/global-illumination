use bevy::light::CascadeShadowConfigBuilder;
use bevy::prelude::*;

use crate::terrain::TERRAIN_VIEW_DISTANCE;

pub struct WorldEnvironmentPlugin;

impl Plugin for WorldEnvironmentPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(sky_color()))
            .add_systems(Startup, setup_lighting);
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
