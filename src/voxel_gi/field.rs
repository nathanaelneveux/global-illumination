use super::spectrum::EnergySpectrum;
use super::world::SurfaceProbe;
use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use std::collections::HashMap;
use std::time::Duration;

/// Keep the world-sized index small. Growing a flat HashMap of large ambient
/// cubes used to relocate hundreds of MB on the game thread during streaming.
#[derive(Default)]
pub(crate) struct ProbeStore(bevy::platform::collections::HashMap<IVec3, Box<SurfaceProbe>>);

impl ProbeStore {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn get(&self, key: &IVec3) -> Option<&SurfaceProbe> {
        self.0.get(key).map(Box::as_ref)
    }
    pub fn get_mut(&mut self, key: &IVec3) -> Option<&mut SurfaceProbe> {
        self.0.get_mut(key).map(Box::as_mut)
    }
    pub fn contains_key(&self, key: &IVec3) -> bool {
        self.0.contains_key(key)
    }
    pub fn keys(&self) -> impl Iterator<Item = &IVec3> {
        self.0.keys()
    }
    pub fn values(&self) -> impl Iterator<Item = &SurfaceProbe> {
        self.0.values().map(Box::as_ref)
    }
    #[cfg(test)]
    pub fn insert(&mut self, key: IVec3, probe: SurfaceProbe) {
        if let Some(existing) = self.0.get_mut(&key) {
            **existing = probe;
        } else {
            self.0.insert(key, Box::new(probe));
        }
    }
    /// Transfer ownership during page publication without freeing/reallocating
    /// every retained ambient cube. Integer keys are trusted world coordinates;
    /// this index uses the engine's fast hasher, as does the corner cache.
    pub fn remove(&mut self, key: &IVec3) -> Option<Box<SurfaceProbe>> {
        self.0.remove(key)
    }
    pub fn insert_box(&mut self, key: IVec3, probe: Box<SurfaceProbe>) {
        self.0.insert(key, probe);
    }
}

impl std::ops::Index<&IVec3> for ProbeStore {
    type Output = SurfaceProbe;
    fn index(&self, key: &IVec3) -> &Self::Output {
        self.get(key).expect("indexed live probe")
    }
}

pub(crate) trait ProbeLookup {
    fn lookup(&self, key: &IVec3) -> Option<&SurfaceProbe>;
}
impl ProbeLookup for ProbeStore {
    fn lookup(&self, key: &IVec3) -> Option<&SurfaceProbe> {
        self.get(key)
    }
}
#[cfg(test)]
impl ProbeLookup for HashMap<IVec3, SurfaceProbe> {
    fn lookup(&self, key: &IVec3) -> Option<&SurfaceProbe> {
        self.get(key)
    }
}

/// The authoritative world-space energy field. The infrared and ionizing
/// channels can be read by offscreen simulation systems.
#[derive(Resource, Default)]
pub struct VoxelEnergyField {
    pub(super) probes: ProbeStore,
    pub(super) order: Vec<IVec3>,
    pub(super) pages: HashMap<IVec3, Vec<IVec3>>,
    pub(super) source_revision: u64,
    pub(super) transport_revision: u64,
    pub(super) requested: HashSet<IVec3>,
    /// Geometry is ready but these receiver pages have not been published yet.
    pub(super) publishing_pages: HashSet<IVec3>,
}

/// An unfiltered surface measurement, never the spatially smoothed display
/// value. Bands currently use normalized transport units, not watts or dose.
#[derive(Clone, Copy, Debug)]
pub struct EnergyEstimate {
    pub incident: EnergySpectrum,
    pub variance: EnergySpectrum,
    pub effective_samples: u32,
    pub age: Duration,
    /// Analytic-source/material-table snapshot is current. Scene-bound emission
    /// changes are tracked by `transport_current`; consumers must check both.
    pub sources_current: bool,
    /// Includes scene geometry, voxel material and emission changes, even for
    /// distant receivers. Current means refreshed, not converged: bounded
    /// pre-change statistical weight can remain in a fresh estimate.
    pub transport_current: bool,
}

impl VoxelEnergyField {
    /// Ask for fresher measurements for a loaded voxel (all its exposed faces).
    /// Requests are deduplicated and serviced by the bounded urgent queue, not
    /// synchronously. They cannot starve the all-loaded-world sweep. A caller
    /// must still inspect `estimate` freshness; this does not promise completion.
    pub fn request_refresh(&mut self, voxel: IVec3) -> bool {
        if !self.probes.contains_key(&voxel) {
            return false;
        }
        self.requested.insert(voxel);
        true
    }

    /// Compatibility helper. Simulation code should use `estimate` to enforce
    /// its own age, confidence and source-revision requirements.
    pub fn sample(&self, voxel: IVec3, normal: Vec3) -> Option<EnergySpectrum> {
        self.estimate(voxel, normal).map(|sample| sample.incident)
    }

    pub fn estimate(&self, voxel: IVec3, normal: Vec3) -> Option<EnergyEstimate> {
        if self.publishing_pages.contains(&super::page_of(voxel)) {
            return None;
        }
        let probe = self.probes.get(&voxel)?;
        let normal = normal.normalize_or_zero();
        if normal == Vec3::ZERO {
            return None;
        }
        let weights = normal * normal;
        let mut incident = EnergySpectrum::ZERO;
        let mut variance = EnergySpectrum::ZERO;
        let mut samples = u32::MAX;
        let mut oldest = probe.updated_at?;
        let mut fresh = true;
        for axis in 0..3 {
            if weights[axis] < 1e-6 {
                continue;
            }
            let face = axis * 2 + usize::from(normal[axis] < 0.0);
            if probe.face_mask & (1 << face) == 0 || probe.samples[face] == 0 {
                return None;
            }
            incident += (probe.radiance[face] + probe.direct[face]) * weights[axis];
            variance += (probe.second_moment[face] - probe.radiance[face] * probe.radiance[face])
                .finite_or_zero()
                * (weights[axis] * weights[axis]);
            samples = samples.min(probe.samples[face]);
            oldest = oldest.min(probe.face_updated_at[face].unwrap_or(oldest));
            fresh &= probe.fresh_mask & (1 << face) != 0;
        }
        Some(EnergyEstimate {
            incident,
            variance,
            effective_samples: samples,
            age: oldest.elapsed(),
            sources_current: probe.source_revision == self.source_revision && fresh,
            transport_current: fresh && probe.transport_revision == self.transport_revision,
        })
    }

    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    pub(super) fn update_indices(
        &mut self,
        added: &[IVec3],
        removed: &HashMap<IVec3, Box<SurfaceProbe>>,
    ) {
        let mut indices: Vec<_> = removed.values().map(|probe| probe.order_index).collect();
        indices.sort_unstable_by(|a, b| b.cmp(a));
        for index in indices {
            self.order.swap_remove(index);
            if index < self.order.len() {
                self.probes.get_mut(&self.order[index]).unwrap().order_index = index;
            }
        }
        for voxel in added {
            self.probes.get_mut(voxel).unwrap().order_index = self.order.len();
            self.order.push(*voxel);
        }
    }
}
