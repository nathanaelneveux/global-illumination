//! Deterministic direct-sun samples at virtual unit-face corners, independent
//! of greedy mesh vertices. RGBE storage keeps the existing GPU probe stride.
use bevy::prelude::*;

use super::world::FACE_OFFSETS;

/// Zero means unknown/unmeasured, ONE means measured black. RGB mantissas
/// occupy the low 24 bits and the shared IEEE exponent occupies the high byte.
/// This is presentation-only quantization; simulation energy stays full f32.
pub(crate) fn pack(value: Vec3) -> u32 {
    let value = value.max(Vec3::ZERO);
    let maximum = value.max_element();
    if !maximum.is_finite() || maximum < 1e-30 {
        return 1;
    }
    let exponent = (maximum.to_bits() >> 23) & 255;
    let quantum = 2.0_f32.powi(exponent as i32 - 127 - 7);
    let mantissa = (value / quantum).round().min(Vec3::splat(255.0)).as_uvec3();
    mantissa.x | (mantissa.y << 8) | (mantissa.z << 16) | (exponent << 24)
}

/// Decode a packed corner value (zero and measured-black decode to zero).
pub(crate) fn unpack(value: u32) -> Vec3 {
    if value >> 24 == 0 {
        return Vec3::ZERO;
    }
    UVec3::new(value & 255, (value >> 8) & 255, (value >> 16) & 255).as_vec3()
        * 2.0_f32.powi((value >> 24) as i32 - 127 - 7)
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
    fn packed_sky_preserves_black_validity_and_hdr_without_growing_probe_storage() {
        assert_ne!(pack(Vec3::ZERO), 0);
        assert_eq!(unpack(pack(Vec3::ZERO)), Vec3::ZERO);
        for magnitude in [0.001, 0.1, 1.0, 1.99, 10.0, 1e6, 1e12] {
            let color = Vec3::new(1.0, 0.7, 0.2) * magnitude;
            assert!((unpack(pack(color)) - color).abs().max_element() <= magnitude / 128.0);
        }
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
