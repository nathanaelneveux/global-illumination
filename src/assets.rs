use bevy::asset::AssetEvent;
use bevy::prelude::*;
use bevy_asset_loader::prelude::{AssetCollection, AssetCollectionApp};
use bevy_common_assets::ron::RonAssetPlugin;

use crate::terrain::PrototypeWorld;

pub struct AssetSupportPlugin;

impl Plugin for AssetSupportPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RonAssetPlugin::<PrototypeConfig>::new(&["prototype.ron"]))
            .init_collection::<TemplateAssets>()
            .add_systems(
                Update,
                (
                    log_when_template_assets_are_ready,
                    refresh_voxel_texture_on_change,
                ),
            );
    }
}

#[derive(AssetCollection, Resource)]
pub struct TemplateAssets {
    #[asset(path = "example_voxel_texture.png")]
    pub voxel_texture: Handle<Image>,
    #[asset(path = "config/default.prototype.ron")]
    pub prototype_config: Handle<PrototypeConfig>,
}

#[derive(Asset, TypePath, Debug, Clone, PartialEq, serde::Deserialize)]
pub struct PrototypeConfig {
    pub world_seed: u32,
    pub terrain_period: f32,
    pub player_spawn_height_offset: f32,
    pub voxel_texture_layers: u32,
    pub ambient_occlusion: bool,
}

impl Default for PrototypeConfig {
    fn default() -> Self {
        Self {
            world_seed: 7,
            terrain_period: 256.0,
            player_spawn_height_offset: 8.0,
            voxel_texture_layers: 4,
            ambient_occlusion: false,
        }
    }
}

fn log_when_template_assets_are_ready(
    template_assets: Option<Res<TemplateAssets>>,
    prototype_configs: Res<Assets<PrototypeConfig>>,
    mut logged: Local<bool>,
) {
    if *logged {
        return;
    }

    let Some(template_assets) = template_assets else {
        return;
    };

    let _voxel_texture = template_assets.voxel_texture.id();
    let Some(config) = prototype_configs.get(&template_assets.prototype_config) else {
        return;
    };

    info!(
        world_seed = config.world_seed,
        terrain_period = config.terrain_period,
        player_spawn_height_offset = config.player_spawn_height_offset,
        voxel_texture_layers = config.voxel_texture_layers,
        ambient_occlusion = config.ambient_occlusion,
        "template asset support ready"
    );
    *logged = true;
}

fn refresh_voxel_texture_on_change(
    template_assets: Option<Res<TemplateAssets>>,
    world: Option<Res<PrototypeWorld>>,
    mut image_events: MessageReader<AssetEvent<Image>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(template_assets) = template_assets else {
        return;
    };
    let Some(world) = world else {
        return;
    };

    let texture_id = template_assets.voxel_texture.id();
    let mut should_refresh = world.is_changed();

    for event in image_events.read() {
        if event.is_loaded_with_dependencies(texture_id) || event.is_modified(texture_id) {
            should_refresh = true;
        }
    }

    if !should_refresh {
        return;
    }

    let Some(mut image) = images.get_mut(&template_assets.voxel_texture) else {
        return;
    };
    let layers = world.voxel_texture_layers().max(1);
    // Read through AssetMut first. Mutably dereferencing an already prepared
    // image would emit Modified again and reupload the texture every frame.
    if voxel_gi_bvw::texture_array_ready(&image, layers) {
        return;
    }
    if let Err(error) = voxel_gi_bvw::prepare_texture_array(&mut image, layers) {
        error!(%error, "failed to prepare terrain texture array");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::message::MessageCursor;
    use bevy::image::ImagePlugin;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

    fn stacked_image() -> Image {
        Image::new_fill(
            Extent3d {
                width: 2,
                height: 8,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[255; 4],
            TextureFormat::Rgba8UnormSrgb,
            default(),
        )
    }

    #[test]
    fn texture_preparation_does_not_create_a_modified_event_loop_and_handles_reload() {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin::default(),
            ImagePlugin::default(),
        ))
        .init_resource::<PrototypeWorld>()
        .add_systems(Update, refresh_voxel_texture_on_change);
        let texture = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(stacked_image());
        app.insert_resource(TemplateAssets {
            voxel_texture: texture.clone(),
            prototype_config: default(),
        });
        let mut events = MessageCursor::<AssetEvent<Image>>::default();
        let mut modified = 0;
        for _ in 0..8 {
            app.update();
            modified += events
                .read(app.world().resource::<Messages<AssetEvent<Image>>>())
                .filter(|event| event.is_modified(texture.id()))
                .count();
        }
        assert_eq!(
            modified, 1,
            "preparation must modify the image exactly once"
        );
        assert!(voxel_gi_bvw::texture_array_ready(
            app.world()
                .resource::<Assets<Image>>()
                .get(&texture)
                .unwrap(),
            4
        ));
        app.world_mut()
            .resource_mut::<Assets<Image>>()
            .insert(texture.id(), stacked_image())
            .unwrap();
        for _ in 0..8 {
            app.update();
            modified += events
                .read(app.world().resource::<Messages<AssetEvent<Image>>>())
                .filter(|event| event.is_modified(texture.id()))
                .count();
        }
        assert_eq!(modified, 3, "one reload and one preparation, then idle");
        *app.world_mut().resource_mut::<PrototypeWorld>() =
            PrototypeWorld::from_config(PrototypeConfig {
                voxel_texture_layers: 2,
                ..default()
            });
        for _ in 0..4 {
            app.update();
        }
        let image = app
            .world()
            .resource::<Assets<Image>>()
            .get(&texture)
            .unwrap();
        assert!(voxel_gi_bvw::texture_array_ready(image, 2));
        assert_eq!(image.texture_descriptor.size.height, 4);
        // Two 2x4 layers, each with its 2x4, 1x2 and 1x1 levels.
        assert_eq!(image.data.as_ref().unwrap().len(), 2 * (8 + 2 + 1) * 4);
    }
}
