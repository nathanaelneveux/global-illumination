//! Geometry is shared by worker compute and the renderer. Queue writes to its
//! individual buffers are not a transaction: another thread can submit between
//! them. Stage sparse edits privately, then apply all copies in one command
//! buffer. Readers see a coherent scene before or after that submission.
//!
//! Buffer growth, directory-mask changes, or changed traversal bounds start a
//! new allocation generation. All three buffers detach together; extracted
//! materials may keep the previous buffers and uniforms for another frame.
//! Ordinary edits reuse allocations and transfer only their sparse deltas.
use super::*;

impl GpuTracer {
    pub(crate) fn needs_geometry(&self, scene: &Arc<TransportScene>) -> bool {
        self.scene
            .as_ref()
            .is_none_or(|old| !Arc::ptr_eq(old, scene))
    }

    pub(super) fn synchronize_geometry(
        &mut self,
        scene: &Arc<TransportScene>,
    ) -> Result<(), String> {
        if !self.needs_geometry(scene) {
            return Ok(());
        }
        let previous_mask = self.packed.directory.len();
        let delta = self.packed.update(scene);
        #[cfg(test)]
        let delta = if self.full_geometry_uploads_for_benchmark {
            self.packed.rebuild_directory();
            super::super::gpu_scene::SceneDelta {
                directory_rebuilt: true,
                headers: std::iter::once(0..self.packed.headers.len()).collect(),
                ..delta
            }
        } else {
            delta
        };
        let lengths = [
            self.packed.directory.len() * 16,
            self.packed.headers.len() * 16,
            self.packed.cells.len() * 4,
        ];
        let bounds_changed = self.scene.as_ref().is_some_and(|old| {
            old.top_level.minimum != scene.top_level.minimum
                || old.top_level.maximum != scene.top_level.maximum
        });
        let detach = bounds_changed
            || previous_mask != self.packed.directory.len()
            || [&self.directory, &self.regions, &self.cells]
                .into_iter()
                .zip(lengths)
                .any(|(buffer, bytes)| bytes as u64 > buffer.capacity);
        let device = &self.context.device;
        let queue = &self.context.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GI coherent geometry transaction"),
        });
        if detach {
            for (buffer, bytes) in [&mut self.directory, &mut self.regions, &mut self.cells]
                .into_iter()
                .zip(lengths)
            {
                let mut replacement = UploadBuffer::new(device, buffer.label, buffer.usage);
                replacement.ensure(device, bytes.max(buffer.capacity as usize))?;
                // Unchanged contents stay GPU-resident even when the generation
                // changes. Only changed ranges cross the CPU/GPU boundary.
                encoder.copy_buffer_to_buffer(
                    &buffer.buffer,
                    0,
                    &replacement.buffer,
                    0,
                    buffer.capacity,
                );
                *buffer = replacement;
            }
        }
        let mut bytes = Vec::new();
        let mut patches = Vec::new();
        let mut stage = |buffer: usize, offset: usize, data: Vec<u8>| {
            if !data.is_empty() {
                patches.push((buffer, offset as u64, bytes.len() as u64, data.len() as u64));
                bytes.extend_from_slice(&data);
            }
        };
        if delta.directory_rebuilt {
            stage(0, 0, crate::gpu::std430(&self.packed.directory.clone()));
        } else {
            for range in delta.directory {
                stage(
                    0,
                    range.start * 16,
                    crate::gpu::std430(&self.packed.directory[range].to_vec()),
                );
            }
        }
        for range in delta.headers {
            stage(
                1,
                range.start * 16,
                crate::gpu::std430(&self.packed.headers[range].to_vec()),
            );
        }
        for range in delta.cells {
            stage(2, range.start * 4, encode_words(&self.packed.cells[range]));
        }
        self.geometry_staging.upload(device, queue, &bytes)?;
        let buffers = [&self.directory, &self.regions, &self.cells];
        for (target, offset, source, size) in patches {
            encoder.copy_buffer_to_buffer(
                &self.geometry_staging.buffer,
                source,
                &buffers[target].buffer,
                offset,
                size,
            );
        }
        if detach || !bytes.is_empty() {
            queue.submit([encoder.finish()]);
        }
        self.uploaded_geometry_bytes += bytes.len() as u64;
        self.scene = Some(Arc::clone(scene));
        Ok(())
    }
}
