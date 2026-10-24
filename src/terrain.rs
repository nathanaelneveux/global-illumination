use std::collections::HashMap;
use std::sync::Arc;

use bevy::asset::AssetEvent;
use bevy::prelude::*;
use bevy_voxel_world::custom_meshing::{CHUNK_SIZE_F, CHUNK_SIZE_I};
use bevy_voxel_world::prelude::*;

use crate::assets::{PrototypeConfig, TemplateAssets};
use crate::scene_presets::ScenePreset;
use crate::structures::{Bounds, StructureLayer, VoxelOp, showcase};
use crate::terrain_meshing::build_chunk_mesh;
use crate::terrain_noise::{BEDROCK_FLOOR_Y, MAX_SURFACE_Y, TerrainColumn, TerrainNoise};
use crate::voxel_gi::{
    MaterialTransport, VoxelGiInput, VoxelMaterialTransportTable,
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
    Emissive = 4,
    CaveRock = 5,
    Lava = 6,
    SpireLight = 7,
    ThoriumCrystal = 8,
    CaveTorch = 9,
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
            4 => Some(Self::Emissive),
            5 => Some(Self::CaveRock),
            6 => Some(Self::Lava),
            7 => Some(Self::SpireLight),
            8 => Some(Self::ThoriumCrystal),
            9 => Some(Self::CaveTorch),
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
            Self::Snow
            | Self::Emissive
            | Self::CaveRock
            | Self::Lava
            | Self::SpireLight
            | Self::ThoriumCrystal
            | Self::CaveTorch => [SNOW_TEXTURE; 3],
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
        // Prototype palette policy belongs to the host, not the transport core.
        // Keep these responses independent of display texture/atlas indices.
        for (material, visible) in [
            (TerrainMaterial::Grass, Vec3::new(0.17, 0.42, 0.08)),
            (TerrainMaterial::Brick, Vec3::new(0.55, 0.20, 0.09)),
            (TerrainMaterial::SnowBrick, Vec3::new(0.46, 0.29, 0.20)),
            (TerrainMaterial::Snow, Vec3::new(0.82, 0.88, 0.94)),
        ] {
            let mut table = app
                .world_mut()
                .resource_mut::<VoxelMaterialTransportTable>();
            let mut optical = table.get(material.index());
            optical.diffuse = visible;
            table.set(material.index(), optical);
        }
        // Register once, not on every placement: editing a lamp uses ordinary
        // geometry invalidation instead of changing the global material table.
        app.world_mut()
            .resource_mut::<VoxelMaterialTransportTable>()
            .set(
                TerrainMaterial::Emissive.index(),
                emissive_block_transport(),
            );
        if let Some(materials) = ScenePreset::current().materials() {
            app.insert_resource(materials);
        } else {
            let cave = crate::cave_scene::materials();
            let mut table = app
                .world_mut()
                .resource_mut::<VoxelMaterialTransportTable>();
            for &(local, global) in &showcase::CAVE_PALETTE {
                table.set(global.index(), cave.get(local));
            }
        }
        let transport_assets = app.world().resource::<VoxelTransportAssets>().clone();
        app.add_plugins(
            VoxelWorldPlugin::with_config(PrototypeWorld::default())
                .with_material(material_template(&transport_assets)),
        )
        .add_systems(Update, apply_live_prototype_config);
    }
}

pub(crate) fn emissive_block_transport() -> MaterialTransport {
    MaterialTransport {
        diffuse: Vec3::splat(0.8),
        emission: Vec3::new(12.0, 5.0, 1.0),
    }
}

#[derive(Resource, Clone)]
pub struct PrototypeWorld {
    config: PrototypeConfig,
    noise: Arc<TerrainNoise>,
    scene: ScenePreset,
    structures: Option<Arc<StructureLayer>>,
}

impl Default for PrototypeWorld {
    fn default() -> Self {
        Self::from_config(PrototypeConfig::default())
    }
}

impl PrototypeWorld {
    pub fn from_config(config: PrototypeConfig) -> Self {
        let scene = ScenePreset::current();
        Self {
            noise: Arc::new(TerrainNoise::from_config(&config)),
            config,
            scene,
            structures: (scene == ScenePreset::Terrain).then(showcase::world_structures),
        }
    }

    pub fn player_spawn_position(&self) -> Vec3 {
        if let Some(spawn) = self.scene.spawn() {
            return spawn;
        }
        if std::env::var("VOXEL_GI_SPAWN").as_deref() == Ok("hallway") {
            return Vec3::new(-111., -10., -19.5);
        }
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
        if let Some(voxel) = self.scene.voxel(position) {
            return voxel;
        }
        if let Some(voxel) = self
            .structures
            .as_ref()
            .and_then(|layer| stamped_voxel(layer.sample(position)))
        {
            return voxel;
        }
        if position.y <= BEDROCK_FLOOR_Y {
            return TerrainMaterial::Brick.voxel();
        }

        let column = self.noise.sample_column(position.x, position.z);
        terrain_voxel(position, column)
    }

    fn matches_config(&self, config: &PrototypeConfig) -> bool {
        &self.config == config
    }
}

/// GI samples unmaterialized voxels from the same generator BVW uses.
impl crate::voxel_gi::GiFallbackVoxel for PrototypeWorld {
    fn fallback_voxel(&self, position: IVec3) -> WorldVoxel<MaterialIndex> {
        self.generated_voxel(position)
    }
}

impl VoxelWorldConfig for PrototypeWorld {
    type MaterialIndex = MaterialIndex;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        if self.scene == ScenePreset::Cave {
            4
        } else {
            SPAWNING_DISTANCE
        }
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
        let cave = self.scene == ScenePreset::Cave;
        Arc::new(move |id| {
            if cave && id == 2 {
                [0; 3]
            } else {
                TerrainMaterial::texture_indices(id)
            }
        })
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let scene = self.scene;
        if scene != ScenePreset::Terrain {
            return Box::new(move |_, _, _| Box::new(move |p, _| scene.voxel(p).unwrap()));
        }
        let terrain_noise = Arc::clone(&self.noise);
        let structures = self
            .structures
            .as_ref()
            .expect("normal world has structure plan")
            .clone();

        Box::new(move |chunk_pos, _lod, _previous| {
            let origin = chunk_pos * CHUNK_SIZE_I;
            let stamps = structures.for_bounds(Bounds {
                min: origin - IVec3::ONE,
                max: origin + IVec3::splat(CHUNK_SIZE_I + 1),
            });
            let chunk_min_y = chunk_pos.y * CHUNK_SIZE_I;
            let chunk_max_y = chunk_min_y + CHUNK_SIZE_I - 1;

            if stamps.is_empty() && chunk_max_y < BEDROCK_FLOOR_Y {
                return Box::new(|_, _| TerrainMaterial::Brick.voxel());
            }

            if stamps.is_empty() && chunk_min_y > MAX_SURFACE_Y + 1 {
                return Box::new(|_, _| WorldVoxel::Air);
            }

            let mut column_cache = HashMap::<(i32, i32), TerrainColumn>::new();
            let noise = Arc::clone(&terrain_noise);

            Box::new(move |pos: IVec3, _previous| {
                if let Some(voxel) = stamped_voxel(stamps.sample(pos)) {
                    return voxel;
                }
                if pos.y <= BEDROCK_FLOOR_Y {
                    return TerrainMaterial::Brick.voxel();
                }

                let column = *column_cache
                    .entry((pos.x, pos.z))
                    .or_insert_with(|| noise.sample_column(pos.x, pos.z));

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

fn stamped_voxel(op: VoxelOp) -> Option<WorldVoxel<u8>> {
    match op {
        VoxelOp::Keep => None,
        VoxelOp::Air => Some(WorldVoxel::Air),
        VoxelOp::Solid(id) => Some(WorldVoxel::Solid(id)),
    }
}

fn terrain_voxel(position: IVec3, column: TerrainColumn) -> WorldVoxel<MaterialIndex> {
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

fn apply_live_prototype_config(
    template_assets: Option<Res<TemplateAssets>>,
    prototype_configs: Res<Assets<PrototypeConfig>>,
    mut prototype_config_events: MessageReader<AssetEvent<PrototypeConfig>>,
    mut prototype_world: ResMut<PrototypeWorld>,
    mut voxel_gi_dirty: ResMut<VoxelGiInput>,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_world() -> PrototypeWorld {
        let config = PrototypeConfig::default();
        PrototypeWorld {
            noise: Arc::new(TerrainNoise::from_config(&config)),
            config,
            scene: ScenePreset::Terrain,
            structures: Some(showcase::world_structures()),
        }
    }

    #[test]
    fn structure_cave_has_a_clear_supported_route_and_open_skylight() {
        let world = normal_world();
        let mut route: Vec<_> = (0..=18).map(|x| IVec3::new(x, 10, -6)).collect();
        route.extend((-18..=-6).rev().map(|z| IVec3::new(18, 10, z)));
        route.extend((19..=20).map(|x| IVec3::new(x, 10, -18)));
        route.extend((21..=44).map(|x| IVec3::new(x, 30 - x, -18)));
        route.extend((45..=47).map(|x| IVec3::new(x, -14, -18)));
        route.extend(
            (-43..=-19)
                .rev()
                .map(|z| IVec3::new(47, if z >= -21 { -14 } else { z + 7 }, z)),
        );
        for &floor in &route {
            assert_eq!(
                world.generated_voxel(floor),
                TerrainMaterial::CaveRock.voxel(),
                "support at {floor:?}"
            );
            for h in 1..=3 {
                assert_eq!(
                    world.generated_voxel(floor + IVec3::Y * h),
                    WorldVoxel::Air,
                    "headroom at {floor:?}"
                );
            }
        }
        for step in route.windows(2) {
            let d = step[1] - step[0];
            assert!(
                d.x.abs() + d.z.abs() <= 1 && (-1..=0).contains(&d.y),
                "{step:?}"
            );
        }
        for y in 3..=MAX_SURFACE_Y + 1 {
            assert_eq!(
                world.generated_voxel(IVec3::new(93, y, -11)),
                WorldVoxel::Air
            );
        }
        assert_eq!(
            world.generated_voxel(IVec3::new(60, -36, -40)),
            TerrainMaterial::Lava.voxel()
        );
        assert_eq!(
            world.generated_voxel(IVec3::new(72, -20, -34)),
            TerrainMaterial::SpireLight.voxel()
        );
    }

    #[test]
    fn structure_lookup_matches_bvw_jobs_across_bedrock_and_chunk_halos() {
        let world = normal_world();
        let delegate = world.voxel_lookup_delegate();
        for chunk in [
            IVec3::new(0, 0, -1),
            IVec3::new(1, -1, -1),
            IVec3::new(1, -2, -2),
            IVec3::new(2, -1, -1),
            IVec3::new(3, 0, -1),
            IVec3::new(0, -3, 0),
            IVec3::new(0, 3, 0),
        ] {
            let mut job = delegate(chunk, 0, None);
            for z in -1..=32 {
                for y in -1..=32 {
                    for x in -1..=32 {
                        let p = chunk * 32 + IVec3::new(x, y, z);
                        assert_eq!(job(p, None), world.generated_voxel(p), "{p:?} in {chunk:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn structure_palette_never_repurposes_terrain_or_existing_lamps() {
        let mut ids = std::collections::HashSet::new();
        for &(_, material) in &showcase::CAVE_PALETTE {
            assert!(material.index() > TerrainMaterial::Emissive.index());
            assert!(ids.insert(material.index()));
            assert_eq!(TerrainMaterial::texture_indices(material.index()), [0; 3]);
        }
    }

    #[test]
    fn structures_preserve_original_house_and_foundation() {
        let config = PrototypeConfig::default();
        let world = PrototypeWorld {
            noise: Arc::new(TerrainNoise::from_config(&config)),
            config,
            scene: ScenePreset::Terrain,
            structures: Some(showcase::world_structures()),
        };
        for x in -10..=10 {
            for z in -28..=-9 {
                for y in -24..=35 {
                    let p = IVec3::new(x, y, z);
                    let expected = original_house_voxel(p).unwrap_or_else(|| {
                        if y <= BEDROCK_FLOOR_Y {
                            TerrainMaterial::Brick.voxel()
                        } else {
                            terrain_voxel(
                                p,
                                TerrainColumn {
                                    surface_y: 10,
                                    soil_depth: 4,
                                },
                            )
                        }
                    });
                    assert_eq!(world.generated_voxel(p), expected, "{p:?}");
                }
            }
        }
    }

    // Frozen 9930e2b authoring policy; only used to prove the structure migration.
    fn original_house_voxel(position: IVec3) -> Option<WorldVoxel<MaterialIndex>> {
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

    #[test]
    fn editing_lamp_uses_existing_texture_and_a_dedicated_emitting_material() {
        let material = TerrainMaterial::Emissive;
        assert_eq!(
            TerrainMaterial::from_index(material.index()),
            Some(material)
        );
        assert_eq!(material.voxel(), WorldVoxel::Solid(4));
        assert_eq!(TerrainMaterial::texture_indices(material.index()), [0; 3]);
        let mut table = VoxelMaterialTransportTable::default();
        table.set(material.index(), emissive_block_transport());
        let emission = table.get(material.index()).emission;
        assert!(emission.min_element() > 0.0);
        assert_eq!(
            table.get(TerrainMaterial::Grass.index()).emission,
            Vec3::ZERO
        );
    }
}
