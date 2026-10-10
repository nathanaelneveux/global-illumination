//! Main-world display publication. Replacement generations catch up privately
//! in bounded batches, then become visible atomically. The old generation keeps
//! receiving lighting updates throughout preparation.
use super::render::{
    GpuProbePatchQueue, GpuSnapshot, GpuTransportCache, PreparedGpuCache, VoxelTransportAssets,
    VoxelTransportMaterial, build_gpu_cache_for_residency, install_prepared_gpu_cache,
    patch_gpu_topology, patch_probe_values,
};
use super::telemetry::{DisplayStage, GiWorkTimings, MainStage};
use super::{VoxelEnergyField, VoxelGiRuntime, VoxelGiSettings};
use bevy::{
    platform::collections::HashSet,
    prelude::*,
    render::storage::ShaderBuffer,
    tasks::{AsyncComputeTaskPool, block_on, poll_once},
};
use bevy_voxel_world::rendering::VoxelWorldMaterialHandle;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

const CATCHUP_BATCH: usize = 512;
const CATCHUP_MAX_PROBES: usize = 8192;
const CATCHUP_BUDGET: Duration = Duration::from_micros(1500);

/// Coalesce changes without repeatedly scanning a mostly-empty hash table.
/// Popping makes a key eligible for re-enqueue if it changes again. FIFO keeps
/// recurring near updates from starving the rest of a replacement generation.
#[derive(Default)]
pub(super) struct DisplayDirtyQueue {
    order: VecDeque<IVec3>,
    queued: HashSet<IVec3>,
}

impl DisplayDirtyQueue {
    fn extend(&mut self, keys: impl IntoIterator<Item = IVec3>) {
        for key in keys {
            if self.queued.insert(key) {
                self.order.push_back(key);
            }
        }
    }
    fn clear(&mut self) {
        self.order.clear();
        self.queued.clear();
    }
    fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
    pub(super) fn len(&self) -> usize {
        self.order.len()
    }
    fn take_batch(&mut self, batch: &mut Vec<IVec3>) {
        batch.clear();
        while batch.len() < CATCHUP_BATCH {
            let Some(key) = self.order.pop_front() else {
                break;
            };
            self.queued.remove(&key);
            batch.push(key);
        }
    }
    #[cfg(test)]
    pub fn contains(&self, voxel: &IVec3) -> bool {
        self.queued.contains(voxel)
    }
}

fn preparing(runtime: &VoxelGiRuntime) -> bool {
    runtime.gpu_task.is_some() || runtime.gpu_snapshot.is_some() || runtime.gpu_prepared.is_some()
}

fn record_changes(runtime: &mut VoxelGiRuntime, live: bool) {
    // Feed the private ledger continuously during snapshot/build as well as
    // catch-up. Never accumulate all additions for one installation-frame burst.
    runtime
        .gpu_catchup_dirty
        .extend(runtime.gpu_added.drain(..));
    runtime
        .gpu_catchup_dirty
        .extend(runtime.gpu_removed.drain(..));
    runtime
        .gpu_catchup_dirty
        .extend(runtime.gpu_value_dirty.iter().copied());
    if !live {
        // Before the first publication there is no old generation to update.
        // Retaining this set would re-enqueue unchanged keys every frame forever.
        runtime.gpu_value_dirty.clear();
    }
}

pub(super) fn upload_transport_cache(
    mut runtime: ResMut<VoxelGiRuntime>,
    field: Res<VoxelEnergyField>,
    mut assets: ResMut<VoxelTransportAssets>,
    mut patch_queue: ResMut<GpuProbePatchQueue>,
    mut cache: ResMut<GpuTransportCache>,
    material_handle: Option<Res<VoxelWorldMaterialHandle<VoxelTransportMaterial>>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut materials: ResMut<Assets<VoxelTransportMaterial>>,
    settings: Res<VoxelGiSettings>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let mut timing = timings.main_scope(MainStage::DisplayPublication);
    let Some(material_handle) = material_handle else {
        return;
    };
    let build = runtime
        .gpu_task
        .as_mut()
        .and_then(|task| block_on(poll_once(task)));
    if let Some(build) = build {
        runtime.gpu_task = None;
        runtime.gpu_prepared = Some(PreparedGpuCache::new(build));
    }
    timing.display_checkpoint(DisplayStage::Poll);

    if preparing(&runtime) {
        record_changes(&mut runtime, cache.ready);
    }
    if let Some(mut prepared) = runtime.gpu_prepared.take() {
        let started = Instant::now();
        let mut batch = Vec::with_capacity(CATCHUP_BATCH);
        let mut processed = 0;
        let mut fits = true;
        while !runtime.gpu_catchup_dirty.is_empty() && processed < CATCHUP_MAX_PROBES {
            runtime.gpu_catchup_dirty.take_batch(&mut batch);
            processed += batch.len();
            if !prepared.synchronize(&batch, &field.probes) {
                fits = false;
                break;
            }
            if started.elapsed() >= CATCHUP_BUDGET {
                break;
            }
        }
        if !fits {
            // Actual capacity/hash exhaustion, not merely a newer scene revision.
            // Keep the live generation and snapshot current residency again.
            runtime.gpu_topology_dirty = true;
            runtime.gpu_rebuild_required = true;
        } else if runtime.gpu_catchup_dirty.is_empty() {
            install_prepared_gpu_cache(
                prepared,
                &mut cache,
                &mut patch_queue,
                &mut assets,
                &mut buffers,
                &mut materials,
                &material_handle.handle,
            );
            runtime.gpu_topology_dirty = false;
            runtime.gpu_rebuild_required = false;
            runtime.gpu_value_dirty.clear();
            runtime.sky_upload_pending = false;
            runtime.last_gpu_upload = Instant::now();
        } else {
            runtime.gpu_prepared = Some(prepared);
        }
    }
    timing.display_checkpoint(DisplayStage::Install);

    if runtime.gpu_topology_dirty
        && cache.ready
        && !preparing(&runtime)
        && !runtime.gpu_rebuild_required
    {
        let added = std::mem::take(&mut runtime.gpu_added);
        let removed = std::mem::take(&mut runtime.gpu_removed);
        if patch_gpu_topology(
            &mut cache,
            &patch_queue,
            &field.probes,
            &removed,
            &added,
            &runtime.gpu_value_dirty,
        ) {
            runtime.gpu_topology_dirty = false;
            runtime.gpu_value_dirty.clear();
        } else {
            runtime.gpu_rebuild_required = true;
        }
    }
    timing.display_checkpoint(DisplayStage::Topology);

    if runtime.gpu_topology_dirty
        && (!cache.ready || runtime.gpu_rebuild_required)
        && !preparing(&runtime)
    {
        runtime.gpu_snapshot = Some(GpuSnapshot::new(&field.order));
        runtime.gpu_added.clear();
        runtime.gpu_removed.clear();
        runtime.gpu_catchup_dirty.clear();
        record_changes(&mut runtime, cache.ready);
    }
    timing.display_checkpoint(DisplayStage::SnapshotStart);

    if let Some(snapshot) = runtime.gpu_snapshot.as_mut()
        && snapshot.advance(&field.probes)
    {
        let snapshot = runtime.gpu_snapshot.take().unwrap().finish();
        // Reserve for current residency, not only the beginning-of-snapshot
        // key list. Newly loaded probes are reconciled privately above.
        let resident_count = field.probe_count();
        runtime.gpu_task = Some(
            AsyncComputeTaskPool::get()
                .spawn(async move { build_gpu_cache_for_residency(snapshot, resident_count) }),
        );
    }
    timing.display_checkpoint(DisplayStage::SnapshotAdvance);

    if !cache.ready
        || (!runtime.sky_upload_pending
            && runtime.last_gpu_upload.elapsed() < settings.display_upload_interval)
    {
        return;
    }
    let dirty = std::mem::take(&mut runtime.gpu_value_dirty);
    patch_probe_values(&cache, &patch_queue, &field.probes, &dirty);
    timing.display_checkpoint(DisplayStage::Values);
    runtime.gpu_value_dirty = dirty;
    runtime.gpu_value_dirty.clear();
    runtime.last_gpu_upload = Instant::now();
    runtime.sky_upload_pending = false;
    timing.display_checkpoint(DisplayStage::Ledger);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catchup_queue_coalesces_but_requeues_later_changes_fairly() {
        let mut queue = DisplayDirtyQueue::default();
        queue.extend((0..2048).map(|x| IVec3::new(x, 0, 0)));
        queue.extend([IVec3::ZERO, IVec3::X]);
        let mut batch = Vec::new();
        queue.take_batch(&mut batch);
        assert_eq!(batch.len(), CATCHUP_BATCH);
        assert_eq!(batch[0], IVec3::ZERO);
        assert_eq!(queue.order.len(), 1536);
        queue.extend([IVec3::ZERO]);
        assert_eq!(queue.order.back(), Some(&IVec3::ZERO));
        for _ in 0..4 {
            queue.take_batch(&mut batch);
        }
        assert_eq!(batch, [IVec3::ZERO]);
        assert!(queue.is_empty());
    }
}
