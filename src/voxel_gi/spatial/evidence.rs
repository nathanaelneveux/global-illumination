//! Target-evidence reconstruction. Counts are a presentation heuristic, not new
//! physical samples. Gather unique raw/temporal donors, never filtered output.
use bevy::prelude::*;

const RADIUS: i32 = 16;
const EDGE: usize = (RADIUS * 2 + 1) as usize;
const CELLS: usize = EDGE * EDGE;
const MAX_FACES: usize = 1024;

/// Candidate search has a hard radius/work bound. Allow extra sites for uneven
/// weights and rejection; the GPU decides actual support using current history.
pub(crate) fn candidate_limit(target: u32, history: u32) -> usize {
    if history >= target {
        return 1;
    }
    (2 * target.div_ceil(history.max(1)) as usize).clamp(1, MAX_FACES)
}

/// Reused worker scratch: a breadth-first surface walk, not a per-face hash set
/// or a radius-squared allocation. The small bitset prevents duplicate donors
/// and repeated lookups at gaps. Same-normal paths may go around holes, never
/// across them or around a solid corner to a differently oriented surface.
pub(crate) struct Gather<T> {
    pub sites: Vec<(IVec2, T)>,
    visited: [u64; CELLS.div_ceil(64)],
}

impl<T> Default for Gather<T> {
    fn default() -> Self {
        Self {
            sites: Vec::new(),
            visited: [0; CELLS.div_ceil(64)],
        }
    }
}

impl<T: Copy> Gather<T> {
    pub fn collect(
        &mut self,
        voxel: IVec3,
        face: usize,
        center: T,
        limit: usize,
        mut lookup: impl FnMut(IVec3, usize) -> Option<T>,
    ) {
        let (u, v) = super::tangent_axes(face);
        self.collect_connected(center, limit, |_, _, q| {
            lookup(voxel + u * q.x + v * q.y, face)
        });
    }

    /// Same ordered walk using already-indexed surface edges. `direction` is
    /// +u, -u, +v, -v; the offset remains relative to the receiver. Keeping this
    /// traversal shared makes the position-lookup oracle an exact comparison.
    pub fn collect_connected(
        &mut self,
        center: T,
        limit: usize,
        mut neighbor: impl FnMut(T, usize, IVec2) -> Option<T>,
    ) {
        self.sites.clear();
        self.visited.fill(0);
        self.sites.push((IVec2::ZERO, center));
        self.mark(IVec2::ZERO);
        let mut cursor = 0;
        while cursor < self.sites.len() && self.sites.len() < limit.min(MAX_FACES) {
            let (p, value) = self.sites[cursor];
            cursor += 1;
            for (edge, direction) in [IVec2::X, IVec2::NEG_X, IVec2::Y, IVec2::NEG_Y]
                .into_iter()
                .enumerate()
            {
                let q = p + direction;
                if q.length_squared() > RADIUS * RADIUS || !self.mark(q) {
                    continue;
                }
                if let Some(value) = neighbor(value, edge, q) {
                    self.sites.push((q, value));
                    if self.sites.len() >= limit.min(MAX_FACES) {
                        break;
                    }
                }
            }
        }
    }

    fn mark(&mut self, p: IVec2) -> bool {
        let index = (p.y + RADIUS) as usize * EDGE + (p.x + RADIUS) as usize;
        let mask = 1 << (index % 64);
        let word = &mut self.visited[index / 64];
        let fresh = *word & mask == 0;
        *word |= mask;
        fresh
    }
}

#[cfg(any(test, debug_assertions))]
pub(crate) struct Resolve {
    pub rgb: Vec3,
    pub effective: f32,
    pub borrowed: usize,
}

/// Equal per-path variance/independence would give N_eff = W² / Q where
/// W=sum(w_i), Q=sum(w_i²/n_i). Here histories are capped and radiance-correlated,
/// so this only sizes the display footprint; it never certifies convergence.
#[cfg(any(test, debug_assertions))]
pub(crate) fn resolve(sites: &[(IVec2, super::Sample)], target: u32, strength: f32) -> Resolve {
    let center = sites[0].1;
    let n = center.count.max(1) as f32;
    let mut result = Resolve {
        rgb: center.display,
        effective: n,
        borrowed: 0,
    };
    if center.count >= target || strength <= 0.0 {
        return result;
    }
    let target = target as f32;
    let bandwidth2 = (target / n / std::f32::consts::PI).max(1.0);
    let mut sum = center.display * n;
    let mut weight = n;
    let mut squares = n;
    for &(offset, donor) in &sites[1..] {
        if donor.count == 0 {
            continue;
        }
        let tolerance = Vec3::splat(0.002)
            + center.mean.abs().max(donor.mean.abs()) * 0.1
            + (center.error() + donor.error()) * 2.0;
        let difference = ((center.mean - donor.mean).abs() / tolerance).max_element();
        let count = donor.count as f32;
        let mut w = (-0.5 * offset.length_squared() as f32 / bandwidth2
            - 0.5 * difference * difference)
            .exp()
            * count;
        if w <= 1e-8 {
            continue;
        }
        let next_effective = (weight + w).powi(2) / (squares + w * w / count);
        let reached = next_effective >= target;
        if reached {
            // Use only enough of the final donor to reach the target. This
            // makes borrowing vanish continuously as local history approaches
            // target, including edits where donor/receiver histories differ.
            let c = weight * weight - target * squares;
            let a = 1.0 - target / count;
            w = (-c / (weight + (weight * weight - a * c).max(0.0).sqrt())).clamp(0.0, w);
        }
        sum += donor.display * w;
        weight += w;
        squares += w * w / count;
        result.borrowed += 1;
        result.effective = weight * weight / squares;
        if reached {
            break;
        }
    }
    result.rgb = center.display.lerp(sum / weight, strength);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel_gi::spatial::Sample;

    fn sample(count: u32, value: f32) -> Sample {
        Sample {
            count,
            mean: Vec3::splat(value),
            second: Vec3::splat(10.0),
            display: Vec3::splat(value),
        }
    }

    #[test]
    fn footprint_reaches_support_then_shrinks_with_history() {
        let mut gather = Gather::default();
        let mut sizes = Vec::new();
        for count in [1, 4, 16, 128, 511, 512] {
            let value = sample(count, 1.0);
            gather.collect(
                IVec3::ZERO,
                2,
                value,
                candidate_limit(512, count),
                |_, _| Some(value),
            );
            let resolved = resolve(&gather.sites, 512, 1.0);
            assert!((resolved.rgb - Vec3::ONE).length() < 1e-5);
            assert!(
                (resolved.effective - 512.0).abs() < 0.1,
                "{count}: {}",
                resolved.effective
            );
            sizes.push(resolved.borrowed);
        }
        assert!(sizes.windows(2).all(|pair| pair[0] >= pair[1]));
        assert_eq!(*sizes.last().unwrap(), 0);
    }

    #[test]
    fn topology_bounds_and_unique_sites_do_not_force_support_across_gaps() {
        let mut gather = Gather::default();
        gather.collect(IVec3::ZERO, 2, 0, 1024, |p, _| (p.x != 1).then_some(0));
        assert!(
            gather
                .sites
                .iter()
                .all(|(p, _)| p.x <= 0 && p.length_squared() <= 256)
        );
        let unique: bevy::platform::collections::HashSet<_> =
            gather.sites.iter().map(|(p, _)| *p).collect();
        assert_eq!(unique.len(), gather.sites.len());
        let isolated = [(IVec2::ZERO, sample(16, 0.2))];
        let result = resolve(&isolated, 512, 1.0);
        assert_eq!(result.effective, 16.0);
        assert_eq!(result.rgb, Vec3::splat(0.2));
    }

    #[test]
    fn unequal_edit_histories_keep_local_ownership_and_continuous_transition() {
        for donor_history in [16, 256, 2048] {
            let early = [
                (IVec2::ZERO, sample(16, 0.2)),
                (IVec2::X, sample(donor_history, 0.4)),
            ];
            let result = resolve(&early, 512, 1.0);
            assert!(result.rgb.x > 0.2 && result.rgb.x <= 0.4);
            assert!(result.effective <= 512.1);
            let late = [(IVec2::ZERO, sample(511, 0.2)), early[1]];
            assert!((resolve(&late, 512, 1.0).rgb.x - 0.2).abs() < 0.001);
            assert_eq!(resolve(&early, 512, 0.0).rgb, early[0].1.display);
        }
    }
}
