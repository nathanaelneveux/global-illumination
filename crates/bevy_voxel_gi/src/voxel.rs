//! What GI sees at a voxel: air, or one of 256 transport materials whose
//! optics (diffuse albedo, emission) live in `VoxelGiMaterials`. Hosts
//! usually have many more block types than distinct optics; blocks with
//! identical optics share a material.
use super::world::AIR;

/// One voxel as GI sees it: [`GiVoxel::AIR`], or a solid transport
/// material `0..=255` whose optics are in
/// [`VoxelGiMaterials`](crate::VoxelGiMaterials).
///
/// Get materials from [`VoxelGiMaterials::insert`](crate::VoxelGiMaterials::insert),
/// which shares one material between identical optics:
///
/// ```
/// # use bevy::prelude::*;
/// # use bevy_voxel_gi::{GiVoxel, MaterialTransport, VoxelGiMaterials};
/// let mut materials = VoxelGiMaterials::default();
/// let stone = MaterialTransport { diffuse: Vec3::splat(0.5), ..default() };
/// let a = materials.insert(stone).unwrap();
/// let b = materials.insert(stone).unwrap();
/// assert_eq!(a, b, "identical optics share a material");
/// assert!(!a.is_air());
/// // Glass and water neither block nor receive GI light: they are air.
/// let glass = GiVoxel::AIR;
/// assert_eq!(glass.material(), None);
/// ```
///
/// It is two bytes: a host can keep a dense `Vec<GiVoxel>` indexed by its
/// own block IDs and build pages from it without conversion. It cannot be
/// cast from arbitrary integers, which could hold no valid voxel:
///
/// ```compile_fail
/// # use bevy_voxel_gi::GiVoxel;
/// let voxels: &[GiVoxel] = bytemuck::cast_slice(&[257_u16]);
/// ```
// `NoUninit` only: GI's own zero-copy casts go from `GiVoxel` to `u16`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, bytemuck::NoUninit)]
#[repr(transparent)]
pub struct GiVoxel(u16);

impl GiVoxel {
    /// Empty space: neither blocks nor receives light.
    pub const AIR: Self = Self(AIR);

    /// Solid transport material `material`.
    pub const fn solid(material: u8) -> Self {
        Self(material as u16)
    }

    /// Whether this is [`Self::AIR`].
    pub const fn is_air(self) -> bool {
        self.0 == AIR
    }

    /// The solid material, or `None` for air.
    pub const fn material(self) -> Option<u8> {
        if self.is_air() {
            None
        } else {
            Some(self.0 as u8)
        }
    }

    /// GI's internal encoding (`0..=255` solid, 256 air).
    pub(crate) const fn raw(self) -> u16 {
        self.0
    }

    /// From GI's internal encoding.
    pub(crate) const fn from_raw(raw: u16) -> Self {
        debug_assert!(raw <= AIR);
        Self(raw)
    }
}

impl Default for GiVoxel {
    /// Air.
    fn default() -> Self {
        Self::AIR
    }
}

impl From<Option<GiVoxel>> for GiVoxel {
    /// `None` is air.
    fn from(voxel: Option<GiVoxel>) -> Self {
        voxel.unwrap_or(Self::AIR)
    }
}

impl std::fmt::Debug for GiVoxel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.material() {
            Some(material) => write!(f, "GiVoxel::solid({material})"),
            None => write!(f, "GiVoxel::AIR"),
        }
    }
}

/// All 256 transport materials have optics: merge near-identical albedos or
/// emissions on the host side (GI's light transport is diffuse RGB; small
/// differences are rarely visible in indirect light).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterialsFull;

impl std::fmt::Display for MaterialsFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "more than 256 distinct material optics")
    }
}

impl std::error::Error for MaterialsFull {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voxels_round_trip_and_air_is_not_material_zero() {
        assert_eq!(GiVoxel::solid(0).material(), Some(0));
        assert_eq!(GiVoxel::solid(255).material(), Some(255));
        assert!(GiVoxel::AIR.is_air() && !GiVoxel::solid(0).is_air());
        assert_eq!(GiVoxel::default(), GiVoxel::AIR);
        assert_eq!(GiVoxel::from(None), GiVoxel::AIR);
        assert_eq!(format!("{:?}", GiVoxel::solid(7)), "GiVoxel::solid(7)");
        assert_eq!(size_of::<GiVoxel>(), 2);
    }

    #[test]
    fn materials_share_identical_optics_and_skip_set_ones_until_full() {
        use crate::{MaterialTransport, VoxelGiMaterials};
        use bevy::math::Vec3;
        let optics = |i: u32| MaterialTransport {
            emission: Vec3::splat(i as f32),
            ..Default::default()
        };
        let mut materials = VoxelGiMaterials::default();
        // A host's own ID set first is never handed out by `insert`.
        materials.set(0, optics(1000));
        assert_eq!(materials.insert(optics(1)), Ok(GiVoxel::solid(1)));
        assert_eq!(materials.insert(optics(1)), Ok(GiVoxel::solid(1)));
        assert_eq!(materials.insert(optics(1000)), Ok(GiVoxel::solid(0)));
        // Out-of-range optics compare after clamping, as stored.
        let over = |diffuse: f32| MaterialTransport {
            diffuse: Vec3::splat(diffuse),
            ..Default::default()
        };
        let clamped = materials.insert(over(2.0)).unwrap();
        assert_eq!(clamped, GiVoxel::solid(2));
        assert_eq!(materials.insert(over(3.0)), Ok(clamped));
        for i in 3..256 {
            assert_eq!(materials.insert(optics(i)), Ok(GiVoxel::solid(i as u8)));
        }
        assert_eq!(materials.len(), 256);
        assert_eq!(materials.insert(optics(99_999)), Err(MaterialsFull));
        assert_eq!(materials.insert(optics(1)), Ok(GiVoxel::solid(1)));
    }
}
