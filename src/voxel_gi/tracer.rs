//! Transport limits shared by the GPU tracer, world cache and view tier.

pub(crate) const MAX_PATH_DISTANCE: f32 = 768.0;
/// Coverage milestone, not a convergence or quality guarantee. Sampling never
/// stops here; the scheduler continually improves every loaded surface.
pub const BOOTSTRAP_SAMPLES: u32 = 8;
/// Accumulate evidence while the integrand is unchanged. A short rolling
/// history imposes a permanent noise floor on static refinement. Scene/source
/// policies cap retained weight; global model replacements reset it to zero.
/// Change response depends on that cap, not how long the world was static.
pub const HISTORY_LIMIT: u32 = 65_536;
