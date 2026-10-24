//! World-cache batch sizing and the CPU's view of per-face stability. The GPU
//! is the authority for stability (voxel_accumulate.wgsl `record_stability`):
//! it reports each merged face's settled verdict, which the CPU keeps in
//! `ProbeProgress::settled` for the HUD and coverage counters.
use super::world::SurfaceProbe;

/// Faces whose latest GPU merge passed the adaptive stability checks in this
/// scene revision. Not an idle-work mask or a certified convergence estimate.
pub(crate) fn settled_face_mask(probe: &SurfaceProbe, revision: u64) -> u8 {
    if probe.transport_revision != revision {
        return 0;
    }
    probe.progress.settled & probe.face_mask & probe.fresh_mask
}

/// Candidate-search capacity is not the number of rays emitted. A partially
/// settled window must not collapse the all-world sweep to a handful of sites.
pub(crate) fn next_budget(
    current: usize,
    dispatched: usize,
    elapsed_ms: f32,
    target_ms: f32,
    limit: usize,
) -> usize {
    let ratio = (target_ms / elapsed_ms.max(0.1)).clamp(0.5, 1.25);
    let desired = (dispatched as f32 * ratio) as usize;
    let next = if elapsed_ms <= target_ms {
        current.max(desired)
    } else {
        desired
    };
    next.clamp(64, limit.max(64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::*;

    #[test]
    fn sparse_due_windows_cannot_collapse_search_capacity() {
        let mut budget = 8192;
        for _ in 0..600 {
            budget = next_budget(budget, 64, 0.1, 4.0, 8192);
        }
        assert_eq!(budget, 8192);
        assert_eq!(next_budget(8192, 8192, 8.0, 4.0, 8192), 4096);
        assert_eq!(next_budget(256, 256, 1.0, 4.0, 8192), 320);
    }

    #[test]
    fn settled_mask_counts_only_exposed_fresh_current_faces() {
        let mut probe = SurfaceProbe::new_slim(IVec3::ZERO, 5);
        probe.progress.settled = 63;
        assert_eq!(settled_face_mask(&probe, 0), 5);
        assert_eq!(settled_face_mask(&probe, 1), 0, "another scene revision");
        probe.fresh_mask = 1;
        assert_eq!(settled_face_mask(&probe, 0), 1);
        probe.progress.settled = 0;
        assert_eq!(settled_face_mask(&probe, 0), 0, "no GPU verdict");
    }
}
