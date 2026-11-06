//! Mapping a host's block types to GI's transport materials. GI needs only
//! optics (diffuse albedo, emission), so it has 256 transport materials
//! (`0..=255`) plus `AIR`. Hosts usually have many more block types than
//! distinct optics: `MaterialPalette` gives each host key the transport
//! material of its optics, sharing one ID between identical optics.
use super::world::{AIR, MaterialTransport, VoxelMaterialTransportTable};
use bevy::platform::collections::HashMap;
use std::hash::Hash;

/// More than 256 distinct optics were registered. Merge near-identical
/// albedos or emissions on the host side (GI's light transport is diffuse
/// RGB; small differences are rarely visible in indirect light).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaletteFull;

impl std::fmt::Display for PaletteFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "more than 256 distinct material optics")
    }
}

impl std::error::Error for PaletteFull {}

/// Host keys to transport material IDs, and the matching transport table.
///
/// ```
/// # use bevy_voxel_gi::{AIR, MaterialPalette, MaterialTransport};
/// # use bevy::math::Vec3;
/// let mut palette = MaterialPalette::new();
/// let stone = MaterialTransport { diffuse: Vec3::splat(0.5), emission: Vec3::ZERO };
/// let a = palette.insert("stone", stone).unwrap();
/// let b = palette.insert("stone_bricks", stone).unwrap();
/// assert_eq!(a, b, "identical optics share a transport material");
/// // Glass and water neither block nor receive GI light.
/// assert_eq!(palette.insert_transparent("glass"), AIR);
/// assert_eq!(palette.get(&"glass"), Some(AIR));
/// ```
/// Copy `table()` into the `VoxelMaterialTransportTable` resource, and map
/// voxels in `sample_page`. `get` is a hash lookup; with integer block IDs,
/// build a dense table once (GI samples every voxel of every page), giving
/// blocks you never registered a fallback:
///
/// ```
/// # use bevy_voxel_gi::{MaterialPalette, MaterialTransport};
/// # use bevy::math::Vec3;
/// let mut palette = MaterialPalette::<u16>::new();
/// palette.insert(7, MaterialTransport { diffuse: Vec3::splat(0.2), emission: Vec3::ZERO }).unwrap();
/// let fallback = palette.optics_id(MaterialTransport::default()).unwrap();
/// let dense: Vec<u16> = (0..=u16::MAX).map(|id| palette.get(&id).unwrap_or(fallback)).collect();
/// assert_eq!(dense[7], palette.get(&7).unwrap());
/// assert_eq!(dense[8], fallback);
/// ```
pub struct MaterialPalette<K> {
    ids: HashMap<K, u16>,
    optics: Vec<MaterialTransport>,
    table: VoxelMaterialTransportTable,
}

impl<K: Hash + Eq> Default for MaterialPalette<K> {
    fn default() -> Self {
        Self {
            ids: HashMap::default(),
            optics: Vec::new(),
            table: VoxelMaterialTransportTable::default(),
        }
    }
}

impl<K: Hash + Eq> MaterialPalette<K> {
    /// An empty palette.
    pub fn new() -> Self {
        Self::default()
    }

    /// The transport material for `key` with these optics: an existing one
    /// with identical optics, else the next free ID. Re-inserting a key
    /// remaps it.
    pub fn insert(&mut self, key: K, optics: MaterialTransport) -> Result<u16, PaletteFull> {
        let id = self.optics_id(optics)?;
        self.ids.insert(key, id);
        Ok(id)
    }

    /// The transport material for these optics without a key (a fallback
    /// for unregistered blocks, say): shared with keys of identical optics.
    pub fn optics_id(&mut self, optics: MaterialTransport) -> Result<u16, PaletteFull> {
        Ok(
            match self.optics.iter().position(|known| *known == optics) {
                Some(index) => index as u16,
                None => {
                    if self.optics.len() == 256 {
                        return Err(PaletteFull);
                    }
                    self.optics.push(optics);
                    let id = self.optics.len() as u16 - 1;
                    self.table.set(id as u8, optics);
                    id
                }
            },
        )
    }

    /// A key whose voxels neither block GI light nor receive it (glass,
    /// water, leaves cut out by the host): `AIR` to GI. Its surfaces then
    /// show the light of whatever opaque surface is behind them.
    pub fn insert_transparent(&mut self, key: K) -> u16 {
        self.ids.insert(key, AIR);
        AIR
    }

    /// The transport material of `key` (`AIR` for transparent keys).
    pub fn get(&self, key: &K) -> Option<u16> {
        self.ids.get(key).copied()
    }

    /// Distinct optics registered (at most 256).
    pub fn len(&self) -> usize {
        self.optics.len()
    }

    /// No optics registered yet.
    pub fn is_empty(&self) -> bool {
        self.optics.is_empty()
    }

    /// The transport table for the registered optics (unused IDs keep the
    /// default material).
    pub fn table(&self) -> &VoxelMaterialTransportTable {
        &self.table
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec3;

    #[test]
    fn distinct_optics_get_ids_until_the_table_is_full() {
        let mut palette = MaterialPalette::new();
        for i in 0..256u32 {
            let optics = MaterialTransport {
                diffuse: Vec3::splat(i as f32 / 256.0),
                emission: Vec3::ZERO,
            };
            assert_eq!(palette.insert(i, optics), Ok(i as u16));
            assert_eq!(palette.table().get(i as u8).diffuse, optics.diffuse);
        }
        let more = MaterialTransport {
            diffuse: Vec3::ONE,
            emission: Vec3::ONE,
        };
        assert_eq!(palette.insert(999, more), Err(PaletteFull));
        assert_eq!(
            palette.insert(
                1000,
                MaterialTransport {
                    diffuse: Vec3::splat(3.0 / 256.0),
                    emission: Vec3::ZERO
                }
            ),
            Ok(3),
            "known optics still map when full"
        );
        assert_eq!(palette.len(), 256);
    }

    #[test]
    fn keyless_optics_share_ids_with_keys() {
        let mut palette = MaterialPalette::new();
        let stone = MaterialTransport {
            diffuse: Vec3::splat(0.5),
            emission: Vec3::ZERO,
        };
        let fallback = palette.optics_id(MaterialTransport::default()).unwrap();
        assert_eq!(palette.insert("stone", stone), Ok(fallback + 1));
        assert_eq!(palette.optics_id(stone), Ok(fallback + 1));
        assert_eq!(
            palette.insert("unknown", MaterialTransport::default()),
            Ok(fallback)
        );
        assert_eq!(
            palette.get(&"never"),
            None,
            "keyless optics register no key"
        );
        assert_eq!(palette.len(), 2);
    }
}
