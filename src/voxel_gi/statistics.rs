//! Shared uncertainty measures for transport scheduling and presentation.
//! These are heuristics over correlated, bounded sample histories, not certified
//! confidence intervals. Callers choose tolerances; this module has no display,
//! scene, source, or scheduler state.
use super::spectrum::EnergySpectrum;

/// A per-band heuristic for adaptive work allocation. This is NOT a confidence
/// bound: QMC samples and bounded temporal histories are correlated. Normalize
/// each band independently so bright RGB cannot hide uncertain infrared.
pub(crate) fn relative_error(mean: EnergySpectrum, second: EnergySpectrum, samples: u32) -> f32 {
    relative_error_with_floor(mean, second, samples, 0.02)
}

/// The presentation filter and the refinement scheduler have different error
/// tolerances. A display-stabilizing floor must not make dark transport appear
/// well measured. Callers supply a positive floor in normalized energy units.
pub(crate) fn relative_error_with_floor(
    mean: EnergySpectrum,
    second: EnergySpectrum,
    samples: u32,
    floor: f32,
) -> f32 {
    if samples < 2 {
        return 1.0;
    }
    let variance = (second - mean * mean).finite_or_zero();
    let means = [
        mean.visible.x,
        mean.visible.y,
        mean.visible.z,
        mean.infrared,
        mean.ionizing,
    ];
    let variances = [
        variance.visible.x,
        variance.visible.y,
        variance.visible.z,
        variance.infrared,
        variance.ionizing,
    ];
    means
        .into_iter()
        .zip(variances)
        .map(|(mean, variance)| (variance / samples as f32).sqrt() / (floor + mean.abs()))
        .fold(0.0, f32::max)
        .min(8.0)
}
