//! Deterministic direct-sun visibility at virtual unit-face corners,
//! independent of greedy mesh vertices. A corner stores which directional
//! lights it sees, not their energy: the display applies each light's current
//! direction and color, so a moving or recolored sun changes only shadow
//! borders in this cache.
use bevy::prelude::*;

use super::world::FACE_OFFSETS;

/// A measured corner word. Zero is unknown/unmeasured; a known word has this
/// bit plus one lit bit per directional light (source order).
pub(crate) const KNOWN: u32 = 1 << 31;
/// Directional lights the corner cache and display carry.
pub(crate) const MAX_SKY_LIGHTS: usize = 4;

/// Is `light` visible from a known corner word?
pub(crate) fn lit(word: u32, light: usize) -> bool {
    word & KNOWN != 0 && word & (1 << light) != 0
}

pub(crate) fn face_axes(face: usize) -> (IVec3, IVec3) {
    match face / 2 {
        0 => (IVec3::Y, IVec3::Z),
        1 => (IVec3::X, IVec3::Z),
        _ => (IVec3::X, IVec3::Y),
    }
}

pub(crate) fn corner_position(voxel: IVec3, face: usize, corner: usize) -> IVec3 {
    let (u, v) = face_axes(face);
    voxel
        + FACE_OFFSETS[face].max(IVec3::ZERO)
        + u * (corner & 1) as i32
        + v * ((corner >> 1) & 1) as i32
}

#[cfg(test)]
pub(crate) fn contact_shadow_fixture() -> (
    std::sync::Arc<super::world::TransportScene>,
    super::sources::TransportSources,
    Vec<super::world::SurfaceProbe>,
) {
    use super::EnergySpectrum;
    use super::naadf::{linear_index, volume_len};
    use super::sources::{DirectionalTransportLight, TransportSources};
    use super::world::{AIR, PAGE_SIZE, SurfaceProbe, TransportScene};
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 7..12 {
        for x in 7..12 {
            page[linear_index(UVec3::new(x, 4, z), PAGE_SIZE)] = 2;
        }
        page[linear_index(UVec3::new(7, 5, z), PAGE_SIZE)] = 2;
    }
    let scene = std::sync::Arc::new(TransportScene::with_page_delta(
        &TransportScene::default(),
        vec![(IVec3::ZERO, page)],
        &[],
    ));
    let sources = TransportSources {
        directional: vec![DirectionalTransportLight {
            direction: Vec3::new(-0.35, 1.0, 0.0).normalize(),
            radiance: EnergySpectrum::ONE,
            custom_visible: Vec3::ZERO,
        }],
        ..default()
    };
    let probes = [IVec3::new(8, 4, 8), IVec3::new(8, 4, 9)]
        .map(|voxel| SurfaceProbe::new(voxel, 4))
        .to_vec();
    (scene, sources, probes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visibility_words_distinguish_unknown_shadowed_and_each_light() {
        assert!(!lit(0, 0));
        assert!(!lit(KNOWN, 0));
        let word = KNOWN | 0b101;
        assert!(lit(word, 0) && !lit(word, 1) && lit(word, 2));
        assert_ne!(KNOWN, 0);
    }

    #[test]
    fn shared_corner_positions_are_identical_for_all_face_directions_and_negative_coordinates() {
        let voxel = IVec3::new(-33, 15, -1);
        for face in 0..6 {
            let (u, v) = face_axes(face);
            for corner in 0..4 {
                let position = corner_position(voxel, face, corner);
                let du = if corner & 1 == 0 { -u } else { u };
                let dv = if corner & 2 == 0 { -v } else { v };
                assert_eq!(position, corner_position(voxel + du, face, corner ^ 1));
                assert_eq!(position, corner_position(voxel + dv, face, corner ^ 2));
            }
        }
    }
}
