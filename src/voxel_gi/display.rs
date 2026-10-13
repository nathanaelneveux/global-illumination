//! Main-world display publication. Replacement generations catch up privately
//! in bounded batches, then become visible atomically. The old generation keeps
//! receiving lighting updates throughout preparation.
use super::render::{
    GpuProbe, GpuProbePatchQueue, GpuSnapshot, GpuTransportCache, PreparedGpuCache,
    VoxelTransportAssets, VoxelTransportMaterial, build_gpu_cache_for_residency, gpu_probe,
    install_prepared_gpu_cache, patch_gpu_topology, patch_probe_values,
};
use super::telemetry::{DisplayStage, GiWorkTimings, MainStage};
use super::{VoxelEnergyField, VoxelGiRuntime, VoxelGiSettings, field::ProbeLookup};
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
const CATCHUP_BUDGET: Duration = Duration::from_micros(1500);
// A small tail can finish atomically on the game thread. Larger backlogs ship
// compact records to a worker; neither route increases the capture time slice.
const CATCHUP_WORKER_THRESHOLD: usize = 8192;

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
        self.queued.is_empty()
    }
    pub(super) fn len(&self) -> usize {
        self.queued.len()
    }
    fn captured(&mut self, key: &IVec3) {
        self.queued.remove(key);
    }
    fn take_batch(&mut self, batch: &mut Vec<IVec3>) -> usize {
        batch.clear();
        let mut scanned = 0;
        while scanned < CATCHUP_BATCH {
            let Some(key) = self.order.pop_front() else {
                break;
            };
            scanned += 1;
            if self.queued.remove(&key) {
                batch.push(key);
            }
        }
        scanned
    }
    #[cfg(test)]
    pub fn contains(&self, voxel: &IVec3) -> bool {
        self.queued.contains(voxel)
    }
}

impl FromIterator<IVec3> for DisplayDirtyQueue {
    fn from_iter<T: IntoIterator<Item = IVec3>>(keys: T) -> Self {
        let mut queue = Self::default();
        queue.extend(keys);
        queue
    }
}

impl VoxelGiRuntime {
    /// One actual presentation change feeds both consumers. Pending live
    /// uploads are not new changes and must not restart private catch-up.
    pub(super) fn mark_display_dirty(&mut self, voxel: IVec3) {
        self.gpu_value_dirty.insert(voxel);
        if preparing(self) {
            self.gpu_catchup_dirty.extend([voxel]);
        }
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
    if !live {
        // Before the first publication there is no old generation to update.
        // Retaining this set would re-enqueue unchanged keys every frame forever.
        runtime.gpu_value_dirty.clear();
    }
}

/// Reconcile a private generation within a time slice. False means capacity
/// exhaustion; a nonempty journal on success means continuation next frame.
pub(super) fn catch_up(
    prepared: &mut PreparedGpuCache,
    dirty: &mut DisplayDirtyQueue,
    probes: &impl ProbeLookup,
    budget: Duration,
) -> bool {
    let started = Instant::now();
    let mut batch = Vec::with_capacity(CATCHUP_BATCH);
    // Include obsolete queue entries in the deadline. A second item cap would
    // delay publication even when this budget is idle. Check after each bounded
    // batch so even a zero-duration slice makes progress.
    while !dirty.is_empty() {
        dirty.take_batch(&mut batch);
        if !prepared.synchronize(&batch, probes) {
            return false;
        }
        if started.elapsed() >= budget {
            break;
        }
    }
    true
}

fn capture_catchup(
    dirty: &mut DisplayDirtyQueue,
    probes: &impl ProbeLookup,
    budget: Duration,
) -> Vec<(IVec3, Option<GpuProbe>)> {
    let started = Instant::now();
    let mut delta = Vec::new();
    let mut keys = Vec::with_capacity(CATCHUP_BATCH);
    while !dirty.is_empty() {
        dirty.take_batch(&mut keys);
        delta.extend(
            keys.iter()
                .map(|&voxel| (voxel, probes.lookup(&voxel).map(gpu_probe))),
        );
        if started.elapsed() >= budget {
            break;
        }
    }
    delta
}

/// Reserve for receiver pages already built and awaiting bounded publication,
/// not just those copied into the field so far. Do not credit pending removals:
/// additions may publish first, so the intermediate occupancy can be larger
/// than the final count. This scans page headers, never their receiver contents.
fn publication_capacity(
    field: &VoxelEnergyField,
    pages: impl IntoIterator<Item = (IVec3, usize)>,
) -> usize {
    pages
        .into_iter()
        .fold(field.probe_count(), |count, (page, next)| {
            let current = field.pages.get(&page).map_or(0, Vec::len);
            count.saturating_add(next.saturating_sub(current))
        })
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
    device: Option<Res<super::gpu::GpuTransportDevice>>,
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
        runtime.gpu_prepared = Some(build);
    }
    timing.display_checkpoint(DisplayStage::Poll);

    if preparing(&runtime) {
        record_changes(&mut runtime, cache.ready);
    }
    if let Some(mut prepared) = runtime.gpu_prepared.take() {
        if prepared.is_valid() && runtime.gpu_catchup_dirty.len() > CATCHUP_WORKER_THRESHOLD {
            let delta = capture_catchup(
                &mut runtime.gpu_catchup_dirty,
                &field.probes,
                CATCHUP_BUDGET,
            );
            runtime.gpu_task = Some(AsyncComputeTaskPool::get().spawn(async move {
                prepared.synchronize_delta(&delta);
                prepared
            }));
        } else {
            let fits = prepared.is_valid()
                && catch_up(
                    &mut prepared,
                    &mut runtime.gpu_catchup_dirty,
                    &field.probes,
                    CATCHUP_BUDGET,
                );
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
        let runtime = runtime.as_mut();
        runtime
            .gpu_catchup_dirty
            .extend(runtime.gpu_value_dirty.iter().copied());
        record_changes(runtime, cache.ready);
    }
    timing.display_checkpoint(DisplayStage::SnapshotStart);

    let snapshot_done = {
        let runtime = runtime.as_mut();
        runtime.gpu_snapshot.as_mut().is_some_and(|snapshot| {
            snapshot.advance(&field.probes, |voxel| {
                runtime.gpu_catchup_dirty.captured(&voxel)
            })
        })
    };
    if snapshot_done {
        let snapshot = runtime.gpu_snapshot.take().unwrap().finish();
        // The scene worker may have finished a large streaming batch while
        // only a few pages have reached the field. Repeatedly allocating for
        // that partial count makes private generations fail as the known pages
        // arrive. Reserve their positive growth now; reconcile their values
        // normally as publication proceeds.
        let resident_count = publication_capacity(
            &field,
            runtime.scene_publication.iter().flat_map(|result| {
                result
                    .pages
                    .iter()
                    .map(|(page, probes)| (*page, probes.len()))
            }),
        );
        let device = device.map(|context| context.device.clone());
        runtime.gpu_task = Some(AsyncComputeTaskPool::get().spawn(async move {
            let build = build_gpu_cache_for_residency(snapshot, resident_count);
            let build = if let Some(device) = device {
                build.prepare_buffers(&device)
            } else {
                build
            };
            // CPU-only fallback payloads also belong on the worker, not in
            // the main-thread completion poll.
            PreparedGpuCache::new(build)
        }));
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
    fn worker_capture_is_bounded_immutable_and_requeues_later_changes() {
        use super::super::world::SurfaceProbe;
        let mut probes = bevy::platform::collections::HashMap::new();
        probes.insert(IVec3::ZERO, SurfaceProbe::new(IVec3::ZERO, 4));
        let mut dirty: DisplayDirtyQueue = (0..2048).map(|x| IVec3::new(x, 0, 0)).collect();
        let captured = capture_catchup(&mut dirty, &probes, Duration::ZERO);
        assert_eq!(captured.len(), CATCHUP_BATCH);
        assert!(captured[0].1.is_some());
        assert!(captured[1..].iter().all(|(_, value)| value.is_none()));
        assert_eq!(dirty.len(), 2048 - CATCHUP_BATCH);
        // A removal after capture does not mutate the worker's earlier record.
        // The ledger carries the new state until a later reconciliation.
        probes.remove(&IVec3::ZERO);
        dirty.extend([IVec3::ZERO]);
        assert!(captured[0].1.is_some());
        let next = capture_catchup(&mut dirty, &probes, Duration::from_secs(1));
        assert_eq!(next.last().unwrap().0, IVec3::ZERO);
        assert!(next.last().unwrap().1.is_none());
        assert!(dirty.is_empty());
    }

    #[test]
    fn capacity_includes_committed_growth_without_borrowing_future_removals() {
        use super::super::world::SurfaceProbe;
        let mut field = VoxelEnergyField::default();
        for x in 0..8 {
            let voxel = IVec3::new(x, 0, 0);
            field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
            field.pages.entry(IVec3::ZERO).or_default().push(voxel);
        }
        assert_eq!(publication_capacity(&field, []), 8);
        assert_eq!(publication_capacity(&field, [(IVec3::ZERO, 8)]), 8);
        assert_eq!(publication_capacity(&field, [(IVec3::ZERO, 3)]), 8);
        assert_eq!(publication_capacity(&field, [(IVec3::ZERO, 12)]), 12);
        assert_eq!(
            publication_capacity(&field, [(IVec3::ZERO, 0), (IVec3::NEG_X, 20)]),
            28,
            "a pending deletion cannot fund an earlier addition"
        );
        assert_eq!(
            publication_capacity(&field, [(IVec3::NEG_X, usize::MAX)]),
            usize::MAX
        );
    }

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

    #[test]
    fn snapshot_capture_cancels_obsolete_work_without_unbounded_tombstone_scan() {
        let mut queue: DisplayDirtyQueue = (0..2048).map(|x| IVec3::new(x, 0, 0)).collect();
        for x in 0..2048 {
            queue.captured(&IVec3::new(x, 0, 0));
        }
        assert!(queue.is_empty());
        queue.extend([IVec3::ZERO, IVec3::new(2048, 0, 0)]);
        let mut batch = Vec::new();
        assert_eq!(queue.take_batch(&mut batch), CATCHUP_BATCH);
        assert_eq!(batch, [IVec3::ZERO]);
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.take_batch(&mut batch), CATCHUP_BATCH);
        assert!(
            batch.is_empty(),
            "dead queue entries still count toward the frame limit"
        );
        while !queue.is_empty() {
            queue.take_batch(&mut batch);
        }
        assert_eq!(batch, [IVec3::new(2048, 0, 0)]);
    }

    #[test]
    fn unchanged_live_upload_debt_is_not_new_private_work() {
        let mut runtime = VoxelGiRuntime {
            gpu_snapshot: Some(GpuSnapshot::new(&[])),
            ..default()
        };
        runtime.mark_display_dirty(IVec3::ZERO);
        let mut batch = Vec::new();
        runtime.gpu_catchup_dirty.take_batch(&mut batch);
        record_changes(&mut runtime, true);
        assert!(runtime.gpu_value_dirty.contains(&IVec3::ZERO));
        assert!(runtime.gpu_catchup_dirty.is_empty());
        runtime.mark_display_dirty(IVec3::ZERO);
        assert!(runtime.gpu_catchup_dirty.contains(&IVec3::ZERO));
    }
}
