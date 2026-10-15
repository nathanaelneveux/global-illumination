//! Bounded, nonrecursive reconstruction on connected coplanar voxel surfaces.
//! Transport statistics and simulation remain untouched. The GPU implements
//! the same weights in voxel_accumulate.wgsl; this oracle qualifies captures.
use bevy::prelude::*;

pub(super) fn neighbors<T: Copy>(
    voxel: IVec3,
    face: usize,
    mut lookup: impl FnMut(IVec3, usize) -> Option<T>,
) -> [Option<T>; 20] {
    let (u, v) = match face / 2 {
        0 => (IVec3::Y, IVec3::Z),
        1 => (IVec3::X, IVec3::Z),
        _ => (IVec3::X, IVec3::Y),
    };
    let directions = [u, -u, v, -v];
    let mut result = [None; 20];
    for (axis, offset) in directions.into_iter().enumerate() {
        for distance in 1..=4 {
            let value = lookup(voxel + offset * distance, face);
            result[axis * 4 + distance as usize - 1] = value;
            if value.is_none() {
                break;
            }
        }
    }
    for (i, (a, b)) in [(0, 2), (0, 3), (1, 2), (1, 3)].into_iter().enumerate() {
        if result[a * 4].is_some() || result[b * 4].is_some() {
            result[16 + i] = lookup(voxel + directions[a] + directions[b], face);
        }
    }
    result
}

#[cfg(any(test, debug_assertions))]
#[derive(Clone, Copy)]
pub(super) struct Sample {
    pub mean: Vec3,
    pub second: Vec3,
    pub count: u32,
    pub display: Vec3,
}

#[cfg(any(test, debug_assertions))]
impl Sample {
    fn error(self) -> Vec3 {
        ((self.second - self.mean * self.mean).max(Vec3::ZERO) / self.count.max(1) as f32)
            .map(f32::sqrt)
    }
}

#[cfg(any(test, debug_assertions))]
pub(super) fn reconstruct(center: Sample, donors: [Option<Sample>; 20], strength: f32) -> Vec3 {
    let error = center.error();
    let noise = (error / (Vec3::splat(0.002) + center.mean.abs())).max_element();
    let blend = strength * noise * noise / (noise * noise + 0.15 * 0.15);
    if blend <= 0.001 {
        return center.display;
    }
    let mut sum = center.display;
    let mut weight = 1.0;
    for (i, donor) in donors.into_iter().enumerate() {
        let Some(donor) = donor.filter(|d| d.count > 0) else {
            continue;
        };
        let tolerance = Vec3::splat(0.002)
            + center.mean.abs().max(donor.mean.abs()) * 0.1
            + (error + donor.error()) * 2.0;
        let difference = ((center.mean - donor.mean).abs() / tolerance).max_element();
        let distance_squared = if i >= 16 {
            2.0
        } else {
            ((i % 4 + 1) as f32).powi(2)
        };
        let w = (-0.125 * distance_squared - 0.5 * difference * difference).exp()
            * donor.count as f32
            / (donor.count as f32 + 8.0);
        sum += donor.display * w;
        weight += w;
    }
    center.display.lerp(sum / weight, blend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighborhoods_stop_at_gaps_and_disconnected_diagonals() {
        let donor = IVec3::new(2, 0, 0);
        let diagonal = IVec3::new(1, 0, 1);
        let isolated = neighbors(IVec3::ZERO, 2, |p, _| {
            [donor, diagonal].contains(&p).then_some(p)
        });
        assert!(isolated.iter().all(Option::is_none));
        let connected = neighbors(IVec3::ZERO, 2, |p, _| {
            [IVec3::X, donor, diagonal].contains(&p).then_some(p)
        });
        assert_eq!(connected[0], Some(IVec3::X));
        assert_eq!(connected[1], Some(donor));
        assert_eq!(connected[16], Some(diagonal));
        assert_eq!(connected.iter().flatten().count(), 3);
    }

    #[test]
    fn constants_are_preserved_and_filter_fades_with_evidence() {
        let sample = Sample {
            mean: Vec3::ONE,
            second: Vec3::splat(10.0),
            count: 16,
            display: Vec3::ONE,
        };
        assert!((reconstruct(sample, [Some(sample); 20], 1.0) - Vec3::ONE).length() < 1e-6);
        let donors = [Some(Sample {
            mean: Vec3::splat(0.8),
            display: Vec3::splat(0.8),
            ..sample
        }); 20];
        let early = reconstruct(sample, donors, 1.0);
        let late = reconstruct(
            Sample {
                count: 65536,
                ..sample
            },
            donors,
            1.0,
        );
        assert!(early.x < 0.9);
        assert!(late.x > 0.99);
        assert_eq!(reconstruct(sample, donors, 0.0), sample.display);
        assert_eq!(reconstruct(sample, [None; 20], 1.0), sample.display);
    }
}
