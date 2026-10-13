use super::spectrum::EnergySpectrum;
use super::world::SurfaceProbe;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use std::time::Duration;

const PROBE_SHARDS: usize = 64;

/// Stable receiver allocations with independently growing, smaller index
/// partitions. Streaming no longer rehashes the entire loaded world's index in
/// one insertion. Shards are storage only: scheduling keeps its explicit order.
pub(crate) struct ProbeStore {
    shards: [HashMap<IVec3, Box<SurfaceProbe>>; PROBE_SHARDS],
    len: usize,
}

impl Default for ProbeStore {
    fn default() -> Self {
        Self {
            shards: std::array::from_fn(|_| HashMap::default()),
            len: 0,
        }
    }
}

fn probe_shard(key: &IVec3) -> usize {
    // All axes participate in the low bits so planar surfaces distribute too.
    // Wrapping arithmetic also handles negative/unbounded voxel coordinates.
    let mixed = (key.x as u32).wrapping_mul(0x9e3779b1)
        ^ (key.y as u32).wrapping_mul(0x85ebca77)
        ^ (key.z as u32).wrapping_mul(0xc2b2ae3d);
    super::sampling::pcg(mixed) as usize & (PROBE_SHARDS - 1)
}

#[cfg(test)]
mod store_tests {
    use super::*;
    use std::{hint::black_box, time::Instant};

    #[test]
    fn sharded_store_preserves_identity_count_and_negative_keys() {
        let keys: Vec<_> = (-1024..1024)
            .map(|i| IVec3::new(i * 64, i % 37, -i * 32))
            .chain([IVec3::MIN, IVec3::MAX])
            .collect();
        let mut store = ProbeStore::default();
        for &key in &keys {
            store.insert(key, SurfaceProbe::new(key, 4));
        }
        assert_eq!(store.len(), keys.len());
        assert_eq!(
            store.keys().copied().collect::<HashSet<_>>().len(),
            keys.len()
        );
        let key = keys[0];
        let address = store.get(&key).unwrap() as *const SurfaceProbe;
        let owned = store.remove(&key).unwrap();
        assert_eq!(store.len(), keys.len() - 1);
        assert!(store.remove(&key).is_none());
        store.insert_box(key, owned);
        assert_eq!(store.get(&key).unwrap() as *const _, address);
        store.insert(key, SurfaceProbe::new(key, 8));
        assert_eq!(store.get(&key).unwrap() as *const _, address);
        assert_eq!(store[&key].face_mask, 8);
        assert_eq!(store.len(), keys.len());
        for key in keys {
            assert!(store.contains_key(&key));
            assert_eq!(store.get(&key).unwrap().voxel, key);
            store.get_mut(&key).unwrap().samples[0] = 19;
            assert_eq!(store.remove(&key).unwrap().samples[0], 19);
        }
        assert_eq!(store.len(), 0);
        assert_eq!(store.values().count(), 0);
        assert_eq!(store.keys().count(), 0);
    }

    #[test]
    #[ignore = "CPU receiver-index growth/lookup study; no transport or GPU work"]
    fn receiver_index_growth_study() {
        // A usize has the same index payload width as Box<SurfaceProbe>. Exclude
        // receiver allocation to isolate index growth; do not call this a full
        // publication benchmark. Chunk-coherent coordinates, one million sites.
        const COUNT: usize = 1 << 20;
        const BATCH: usize = 512;
        let keys: Vec<_> = (0..COUNT)
            .map(|i| IVec3::new((i % 1024) as i32 - 512, (i / 1024) as i32 - 512, -64))
            .collect();
        let mut growth = [Vec::new(), Vec::new()];
        let mut lookup = [Vec::new(), Vec::new()];
        for round in 0..8 {
            for mode in [round % 2, 1 - round % 2] {
                let mut maps: Vec<HashMap<IVec3, usize>> =
                    (0..if mode == 0 { 1 } else { PROBE_SHARDS })
                        .map(|_| HashMap::default())
                        .collect();
                let mut worst: f64 = 0.0;
                for (chunk, keys) in keys.chunks(BATCH).enumerate() {
                    let start = Instant::now();
                    for (i, &key) in keys.iter().enumerate() {
                        let shard = if mode == 0 { 0 } else { probe_shard(&key) };
                        maps[shard].insert(key, chunk * BATCH + i);
                    }
                    worst = worst.max(start.elapsed().as_secs_f64() * 1000.0);
                }
                let start = Instant::now();
                let mut sum = 0usize;
                for i in 0..COUNT {
                    let key = &keys[(i * 65537) % COUNT];
                    let shard = if mode == 0 { 0 } else { probe_shard(key) };
                    sum += black_box(maps[shard][key]);
                }
                assert_eq!(sum, COUNT * (COUNT - 1) / 2);
                if round > 0 {
                    growth[mode].push(worst);
                    lookup[mode].push(start.elapsed().as_secs_f64() * 1000.0);
                }
            }
        }
        for mode in 0..2 {
            growth[mode].sort_by(f64::total_cmp);
            lookup[mode].sort_by(f64::total_cmp);
            eprintln!(
                "RECEIVER_INDEX mode={mode} growth_512_worst_median_ms={} lookup_1m_median_ms={}",
                growth[mode][3], lookup[mode][3]
            );
        }
    }
}

impl ProbeStore {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn get(&self, key: &IVec3) -> Option<&SurfaceProbe> {
        self.shards[probe_shard(key)].get(key).map(Box::as_ref)
    }
    pub fn get_mut(&mut self, key: &IVec3) -> Option<&mut SurfaceProbe> {
        self.shards[probe_shard(key)].get_mut(key).map(Box::as_mut)
    }
    pub fn contains_key(&self, key: &IVec3) -> bool {
        self.shards[probe_shard(key)].contains_key(key)
    }
    pub fn keys(&self) -> impl Iterator<Item = &IVec3> {
        self.shards.iter().flat_map(HashMap::keys)
    }
    #[cfg(test)]
    pub fn values(&self) -> impl Iterator<Item = &SurfaceProbe> {
        self.shards
            .iter()
            .flat_map(HashMap::values)
            .map(Box::as_ref)
    }
    #[cfg(test)]
    pub fn insert(&mut self, key: IVec3, probe: SurfaceProbe) {
        if let Some(existing) = self.shards[probe_shard(&key)].get_mut(&key) {
            **existing = probe;
        } else {
            self.insert_box(key, Box::new(probe));
        }
    }
    /// Transfer ownership during page publication without freeing/reallocating
    /// every retained ambient cube. Integer keys are trusted world coordinates;
    /// this index uses the engine's fast hasher, as does the corner cache.
    pub fn remove(&mut self, key: &IVec3) -> Option<Box<SurfaceProbe>> {
        let removed = self.shards[probe_shard(key)].remove(key);
        self.len -= usize::from(removed.is_some());
        removed
    }
    pub fn insert_box(&mut self, key: IVec3, probe: Box<SurfaceProbe>) {
        if self.shards[probe_shard(&key)].insert(key, probe).is_none() {
            self.len += 1;
        }
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
