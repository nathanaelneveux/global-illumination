//! Owned asynchronous snapshots. A mapping, its staging allocation, and the
//! submitted job travel together; a later dispatch cannot overwrite them.
use super::*;

pub(crate) struct Readback {
    pub(super) buffer: UploadBuffer,
    pub(super) submission: wgpu::SubmissionIndex,
    // Mutex supplies Sync for Bevy resource ownership, but the worker owns the
    // snapshot exclusively and accesses the receiver with get_mut, never locks.
    pub(super) completion: std::sync::Mutex<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    pub(super) submitted: Instant,
    pub(super) face_bytes: usize,
    pub(super) bytes: usize,
    pub(super) timestamp_bytes: usize,
    pub(super) samples: u32,
    pub(super) sky_pending: Vec<IVec3>,
    pub(super) sky_ticket: super::super::sky_cache::SkyTicket,
    pub(super) prepare_ms: f32,
    pub(super) ready: bool,
}

impl Readback {
    pub(super) fn poll(&mut self, device: &RenderDevice) -> Result<bool, String> {
        if self.ready {
            return Ok(true);
        }
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| error.to_string())?;
        match self
            .completion
            .get_mut()
            .map_err(|error| error.to_string())?
            .try_recv()
        {
            Ok(result) => {
                result.map_err(|error| error.to_string())?;
                self.ready = true;
                Ok(true)
            }
            Err(mpsc::TryRecvError::Empty) if self.submitted.elapsed() < Duration::from_secs(5) => {
                Ok(false)
            }
            Err(error) => Err(format!("GI snapshot mapping unavailable: {error}")),
        }
    }

    /// The independent reference/tests use the same submission and decoder,
    /// with an explicit worker-only wait. Runtime residency only calls poll.
    pub(super) fn wait(&mut self, device: &RenderDevice) -> Result<(), String> {
        if self.poll(device)? {
            return Ok(());
        }
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(self.submission.clone()),
                timeout: Some(Duration::from_secs(5)),
            })
            .map_err(|error| error.to_string())?;
        self.completion
            .get_mut()
            .map_err(|error| error.to_string())?
            .recv_timeout(Duration::from_secs(1))
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?;
        self.ready = true;
        Ok(())
    }
}
