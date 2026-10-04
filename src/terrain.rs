use std::collections::HashMap;
use std::sync::Arc;

use bevy::asset::AssetEvent;
use bevy::prelude::*;
use bevy_voxel_world::custom_meshing::{CHUNK_SIZE_F, CHUNK_SIZE_I};
use bevy_voxel_world::prelude::*;

use crate::assets::{PrototypeConfig, TemplateAssets};
use crate::terrain_meshing::build_chunk_mesh;
use crate::terrain_noise::{BEDROCK_FLOOR_Y, MAX_SURFACE_Y, TerrainColumn, TerrainNoise};
use crate::voxel_gi::{
    VoxelGiDirty,
    render::{VoxelTransportAssets, material_template},
};

pub type MaterialIndex = u8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TerrainMaterial {
    Grass = 0,
    Brick = 1,
    SnowBrick = 2,
    Snow = 3,
}

impl TerrainMaterial {
    pub const fn index(self) -> MaterialIndex {
        self as MaterialIndex
    }

    pub fn voxel(self) -> WorldVoxel<MaterialIndex> {
        WorldVoxel::Solid(self.index())
    }

    fn from_index(index: MaterialIndex) -> Option<Self> {
        match index {
            0 => Some(Self::Grass),
            1 => Some(Self::Brick),
            2 => Some(Self::SnowBrick),
            3 => Some(Self::Snow),
            _ => None,
        }
    }

    fn texture_indices(index: MaterialIndex) -> [u32; 3] {
        const SNOW_TEXTURE: u32 = 0;
        const SNOW_BRICK_TEXTURE: u32 = 1;
        const BRICK_TEXTURE: u32 = 2;
        const GRASS_TEXTURE: u32 = 3;

        match Self::from_index(index).unwrap_or(Self::Brick) {
            Self::Grass => [GRASS_TEXTURE, GRASS_TEXTURE, GRASS_TEXTURE],
            Self::Brick => [BRICK_TEXTURE, BRICK_TEXTURE, BRICK_TEXTURE],
            Self::SnowBrick => [SNOW_TEXTURE, SNOW_BRICK_TEXTURE, BRICK_TEXTURE],
            Self::Snow => [SNOW_TEXTURE, SNOW_TEXTURE, SNOW_TEXTURE],
        }
    }
}

const WATERLINE_Y: i32 = 2;
pub const SPAWNING_DISTANCE: u32 = 10;
pub const TERRAIN_VIEW_DISTANCE: f32 = (SPAWNING_DISTANCE - 1) as f32 * CHUNK_SIZE_F;
const MIN_DESPAWN_DISTANCE: u32 = 2;

pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        let transport_assets = app.world().resource::<VoxelTransportAssets>().clone();
        app.add_plugins(
            VoxelWorldPlugin::with_config(PrototypeWorld::default())
                .with_material(material_template(&transport_assets)),
        )
        .add_systems(Update, apply_live_prototype_config);
    }
}

#[derive(Resource, Clone)]
pub struct PrototypeWorld {
    config: PrototypeConfig,
    noise: Arc<TerrainNoise>,
}

impl Default for PrototypeWorld {
    fn default() -> Self {
        Self::from_config(PrototypeConfig::default())
    }
}

impl PrototypeWorld {
    pub fn from_config(config: PrototypeConfig) -> Self {
        Self {
            noise: Arc::new(TerrainNoise::from_config(&config)),
            config,
        }
    }

    pub fn player_spawn_position(&self) -> Vec3 {
        Vec3::new(
            0.0,
            self.noise.player_spawn_height(0, 0) + self.config.player_spawn_height_offset,
            0.0,
        )
    }

    pub fn voxel_texture_layers(&self) -> u32 {
        self.config.voxel_texture_layers
    }

    pub fn generated_voxel(&self, position: IVec3) -> WorldVoxel<MaterialIndex> {
        if position.y <= BEDROCK_FLOOR_Y {
            return TerrainMaterial::Brick.voxel();
        }

        let column = terrain_column(&self.noise, position.x, position.z);
        terrain_voxel(position, column)
    }

    fn matches_config(&self, config: &PrototypeConfig) -> bool {
        &self.config == config
    }
}

impl VoxelWorldConfig for PrototypeWorld {
    type MaterialIndex = MaterialIndex;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        SPAWNING_DISTANCE
    }

    fn min_despawn_distance(&self) -> u32 {
        MIN_DESPAWN_DISTANCE
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some((
            "example_voxel_texture.png".into(),
            self.config.voxel_texture_layers,
        ))
    }

    fn texture_index_mapper(&self) -> Arc<dyn Fn(Self::MaterialIndex) -> [u32; 3] + Send + Sync> {
        Arc::new(TerrainMaterial::texture_indices)
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let terrain_noise = Arc::clone(&self.noise);

        Box::new(move |chunk_pos, _lod, _previous| {
            let chunk_min_y = chunk_pos.y * CHUNK_SIZE_I;
            let chunk_max_y = chunk_min_y + CHUNK_SIZE_I - 1;

            if chunk_max_y < BEDROCK_FLOOR_Y {
                return Box::new(|_, _| TerrainMaterial::Brick.voxel());
            }

            if chunk_min_y > MAX_SURFACE_Y + 1 {
                return Box::new(|_, _| WorldVoxel::Air);
            }

            let mut column_cache = HashMap::<(i32, i32), TerrainColumn>::new();
            let noise = Arc::clone(&terrain_noise);

            Box::new(move |pos: IVec3, _previous| {
                if pos.y <= BEDROCK_FLOOR_Y {
                    return TerrainMaterial::Brick.voxel();
                }

                let column = *column_cache
                    .entry((pos.x, pos.z))
                    .or_insert_with(|| terrain_column(&noise, pos.x, pos.z));

                terrain_voxel(pos, column)
            })
        })
    }

    fn chunk_meshing_delegate(
        &self,
    ) -> ChunkMeshingDelegate<Self::MaterialIndex, Self::ChunkUserBundle> {
        let ambient_occlusion = self.config.ambient_occlusion;

        Some(Box::new(
            move |_pos, _lod, _data_shape, _mesh_shape, _previous| {
                Box::new(
                    move |voxels, data_shape, mesh_shape, texture_index_mapper| {
                        (
                            build_chunk_mesh(
                                voxels,
                                data_shape,
                                mesh_shape,
                                texture_index_mapper,
                                ambient_occlusion,
                            ),
                            None,
                        )
                    },
                )
            },
        ))
    }
}

fn terrain_column(noise: &TerrainNoise, x: i32, z: i32) -> TerrainColumn {
    if (-10..=10).contains(&x) && (-28..=-8).contains(&z) {
        TerrainColumn {
            surface_y: 10,
            soil_depth: 4,
        }
    } else {
        noise.sample_column(x, z)
    }
}

fn terrain_voxel(position: IVec3, column: TerrainColumn) -> WorldVoxel<MaterialIndex> {
    if let Some(voxel) = gi_showcase_voxel(position) {
        return voxel;
    }
    if position.y > column.surface_y {
        return WorldVoxel::Air;
    }

    let depth = column.surface_y - position.y;
    let material = if depth == 0 {
        if column.surface_y <= WATERLINE_Y {
            TerrainMaterial::Snow
        } else {
            TerrainMaterial::Grass
        }
    } else if depth <= column.soil_depth {
        if column.surface_y <= WATERLINE_Y {
            TerrainMaterial::Snow
        } else {
            TerrainMaterial::SnowBrick
        }
    } else {
        TerrainMaterial::Brick
    };

    material.voxel()
}

fn gi_showcase_voxel(position: IVec3) -> Option<WorldVoxel<MaterialIndex>> {
    if !(-6..=6).contains(&position.x)
        || !(-24..=-12).contains(&position.z)
        || !(11..=17).contains(&position.y)
    {
        return None;
    }

    let roof = position.y == 17;
    let west_wall = position.x == -6;
    let east_wall = position.x == 6;
    let north_wall = position.z == -24;
    let south_wall = position.z == -12;
    let doorway = south_wall && position.x.abs() <= 1 && position.y <= 14;
    let window = (north_wall || south_wall)
        && (-2..=2).contains(&position.x)
        && (13..=14).contains(&position.y);

    if roof {
        Some(TerrainMaterial::Snow.voxel())
    } else if west_wall && !doorway {
        Some(TerrainMaterial::Grass.voxel())
    } else if east_wall {
        Some(TerrainMaterial::Brick.voxel())
    } else if (north_wall || south_wall) && !doorway && !window {
        Some(TerrainMaterial::SnowBrick.voxel())
    } else {
        Some(WorldVoxel::Air)
    }
}

fn apply_live_prototype_config(
    template_assets: Option<Res<TemplateAssets>>,
    prototype_configs: Res<Assets<PrototypeConfig>>,
    mut prototype_config_events: MessageReader<AssetEvent<PrototypeConfig>>,
    mut prototype_world: ResMut<PrototypeWorld>,
    mut voxel_gi_dirty: ResMut<VoxelGiDirty>,
    mut commands: Commands,
    chunks: Query<Entity, (With<Chunk<PrototypeWorld>>, Without<NeedsDespawn>)>,
) {
    let Some(template_assets) = template_assets else {
        return;
    };

    let config_id = template_assets.prototype_config.id();
    let mut should_apply = false;

    for event in prototype_config_events.read() {
        if event.is_loaded_with_dependencies(config_id) || event.is_modified(config_id) {
            should_apply = true;
        }
    }

    if !should_apply {
        return;
    }

    let Some(config) = prototype_configs.get(&template_assets.prototype_config) else {
        return;
    };

    if prototype_world.matches_config(config) {
        return;
    }

    *prototype_world = PrototypeWorld::from_config(config.clone());
    voxel_gi_dirty.mark_all_loaded();

    for entity in &chunks {
        commands.entity(entity).try_insert(NeedsDespawn);
    }

    info!(
        world_seed = config.world_seed,
        terrain_period = config.terrain_period,
        player_spawn_height_offset = config.player_spawn_height_offset,
        voxel_texture_layers = config.voxel_texture_layers,
        ambient_occlusion = config.ambient_occlusion,
        "applied prototype config"
    );
}
