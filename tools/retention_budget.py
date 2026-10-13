"""Sampling-calibrated history selection for offline policy experiments.

Spatial evidence, assumed change falloff, quality calibration and the temporal
error budget are separate inputs. Nothing here reads future paths or reference
values of the transition being classified. This is not a production API yet.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class QualityGoal:
    p99: float = 0.05
    maximum: float = 0.10

    def __post_init__(self):
        if not 0 < self.p99 <= self.maximum < 1:
            raise ValueError("quality thresholds must satisfy 0 < p99 <= max < 1")


@dataclass(frozen=True)
class SamplingCalibration:
    """Measured work to meet the same raw-face RGB quality gate.

    Import a new qualified sample sweep after changing the sampler. A better
    estimator may reduce variance through correlations without reducing its
    single-path second moment, so that moment alone is not a calibration.
    RGB calibration does not certify an IR/ionizing simulation tolerance.
    """
    paths_for_quality: float
    sampler_id: str
    goal: QualityGoal = QualityGoal()

    def __post_init__(self):
        if not math.isfinite(self.paths_for_quality) or self.paths_for_quality <= 0 or not self.sampler_id:
            raise ValueError("positive finite calibrated work and sampler identity required")

    @classmethod
    def from_sweep(cls, report, sampler_id, goal=QualityGoal()):
        """Require all four complete windows and sustained larger-prefix passes.

        The caller supplies a qualified reference-study report and records its
        provenance. Recompute the threshold, rather than trusting its summary.
        This calibration scene must be separate from transition qualification.
        """
        rows = report["measurements"]
        windows = sorted({r["window"] for r in rows})
        counts = sorted({r["paths"] for r in rows})
        if len(windows) != 4 or not counts or any(n <= 0 for n in counts):
            raise ValueError("require four windows and positive path checkpoints")
        expected = {(w,n) for w in windows for n in counts}
        if len(rows) != len(expected) or {(r["window"],r["paths"]) for r in rows} != expected:
            raise ValueError("incomplete or duplicated calibration windows")
        for r in rows:
            values = (r["centers"]["p99"],r["centers"]["max"])
            if any(not math.isfinite(v) or v < 0 for v in values) or values[0] > values[1]:
                raise ValueError("invalid raw-face error measurement")
        for n in counts:
            if all(r["centers"]["p99"] <= goal.p99 and r["centers"]["max"] <= goal.maximum
                   for r in rows if r["paths"] >= n):
                return cls(float(n),sampler_id,goal)
        raise ValueError("sampler never sustains the requested quality gate")


@dataclass(frozen=True)
class ChangePrior:
    """Heuristic normalized change envelope, NOT a physical influence bound.

    Actual changes in sunlight and geometry share this same distance law.
    Extent/source-power/multibounce qualification is still required before a
    production policy can treat it as an upper bound. There is no 384 cutoff.
    """
    full_change_distance: float = 48.0
    power: float = 2.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (self.full_change_distance,self.power)):
            raise ValueError("positive finite change-prior parameters required")

    def magnitude(self, distance):
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("a finite known distance is required, not missing evidence")
        return (self.full_change_distance / max(distance,self.full_change_distance))**self.power


@dataclass(frozen=True)
class ResponseBudget:
    """Stale-light allowance after a fraction of calibrated settling work.

    Defaults: 3% stale contribution after one sixth of the work needed for the
    fixed quality goal. Sampling noise is evaluated separately, not silently
    included in this bias allowance. These are experimental knobs, not paper
    constants or an assertion that the spatial prior is a proven bound.
    """
    stale_error: float = 0.03
    work_fraction: float = 1/6

    def __post_init__(self):
        if not 0 < self.stale_error < 1 or not math.isfinite(self.work_fraction) or self.work_fraction <= 0:
            raise ValueError("invalid response budget")

    def retained(self, available, change, calibration):
        if any(not math.isfinite(v) or v < 0 for v in (available,change)):
            raise ValueError("finite nonnegative evidence required")
        if change <= self.stale_error:
            return available
        fresh = calibration.paths_for_quality * self.work_fraction
        return min(available,fresh*self.stale_error/(change-self.stale_error))


def select_limit(geometry_distance, sunlight_distance, calibration, budget=ResponseBudget(), prior=ChangePrior()):
    """Known changed sunlight origins supplement, never remove, geometry evidence.

    None means no known changed sunlight origins in this complete fixture
    query, NOT unknown/unmeasured sun. A future runtime adapter must explicitly
    handle missing evidence and revision gaps before calling this function.
    """
    distance = geometry_distance
    if sunlight_distance is not None:
        if not math.isfinite(sunlight_distance) or sunlight_distance < 0:
            raise ValueError("invalid sunlight change distance")
        distance = min(distance,sunlight_distance)
    return budget.retained(65536,prior.magnitude(distance),calibration)
