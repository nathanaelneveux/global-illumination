//! Main-world display publication orchestration. Transport and presentation
//! histories remain authoritative while render cache snapshots are built.
use super::render::{
    GpuProbePatchQueue, GpuSnapshot, GpuTransportCache, VoxelTransportAssets,
    VoxelTransportMaterial, build_gpu_cache, install_gpu_cache, patch_gpu_topology,
    patch_probe_values,
};
use super::{VoxelEnergyField, VoxelGiRuntime, VoxelGiSettings, telemetry::GiWorkTimings};
use bevy::{
    prelude::*,
    render::storage::ShaderBuffer,
    tasks::{AsyncComputeTaskPool, block_on, poll_once},
};
use bevy_voxel_world::rendering::VoxelWorldMaterialHandle;
use std::time::Instant;

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
    let _timing = timings.main_scope();
    let Some(material_handle) = material_handle else {
        return;
    };

    let mut installed = false;
    if let Some(task) = runtime.gpu_task.as_mut()
        && let Some(build) = block_on(poll_once(task))
    {
        runtime.gpu_task = None;
        let catchup = std::mem::take(&mut runtime.gpu_catchup_dirty);
        runtime.gpu_value_dirty.extend(catchup);
        // Install the snapshot even if streaming continued while it encoded;
        // the delta ledger below brings it current. Repeatedly discarding it
        // would prevent a fast-moving player from ever getting a cache.
        install_gpu_cache(
            build,
            &mut cache,
            &mut patch_queue,
            &mut assets,
            &mut buffers,
            &mut materials,
            &material_handle.handle,
        );
        runtime.gpu_rebuild_required = false;
        installed = true;
        runtime.gpu_topology_dirty = runtime.gpu_task_version != runtime.gpu_topology_version;
        runtime.last_gpu_upload = Instant::now();
        info!(
            probes = field.probe_count(),
            "world-space voxel GPU cache installed"
        );
    }

    if runtime.gpu_topology_dirty
        && cache.ready
        && runtime.gpu_task.is_none()
        && runtime.gpu_snapshot.is_none()
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
            // The transaction included these values. No rebuild is in flight
            // on this path, so there is no replacement catch-up ledger to keep.
            runtime.gpu_value_dirty.clear();
        } else {
            // Keep rendering the old cache while the larger allocation builds.
            runtime.gpu_rebuild_required = true;
        }
    }

    if runtime.gpu_topology_dirty
        && (!cache.ready || runtime.gpu_rebuild_required)
        && runtime.gpu_task.is_none()
        && runtime.gpu_snapshot.is_none()
    {
        runtime.gpu_snapshot = Some(GpuSnapshot::new(&field.order));
        runtime.gpu_added.clear();
        runtime.gpu_removed.clear();
        runtime.gpu_task_version = runtime.gpu_topology_version;
    }

    if let Some(snapshot) = runtime.gpu_snapshot.as_mut()
        && snapshot.advance(&field.probes)
    {
        let snapshot = runtime.gpu_snapshot.take().unwrap().finish();
        runtime.gpu_task =
            Some(AsyncComputeTaskPool::get().spawn(async move { build_gpu_cache(snapshot) }));
    }

    // Never expose a worker's stale snapshot for the normal upload interval.
    // Merge every value changed since that snapshot in its installation frame.
    if !cache.ready
        || (!installed
            && !runtime.sky_upload_pending
            && runtime.last_gpu_upload.elapsed() < settings.display_upload_interval)
    {
        return;
    }
    let dirty = std::mem::take(&mut runtime.gpu_value_dirty);
    patch_probe_values(&cache, &patch_queue, &field.probes, &dirty);
    if runtime.gpu_task.is_some() || runtime.gpu_snapshot.is_some() {
        // Catch-up and live publication are separate ledgers. Re-uploading the
        // entire growing catch-up set every 50ms was itself a streaming spike.
        runtime.gpu_catchup_dirty.extend(dirty.iter().copied());
    }
    // Reuse the high-water allocation instead of rebuilding a growing hash set
    // every display upload. Catch-up owns only keys, never this scratch storage.
    runtime.gpu_value_dirty = dirty;
    runtime.gpu_value_dirty.clear();
    runtime.last_gpu_upload = Instant::now();
    runtime.sky_upload_pending = false;
}
