//! Opt-in, read-only attribution of accepted history reductions. Raw corner
//! evidence is a diagnostic sidecar: it never changes production bounds, event
//! ordering, caps or sampling. Frozen scene oracles run after the timed workload.
use super::super::{
    VoxelGiRuntime, VoxelGiSettings, VoxelMaterialTransportTable,
    emission::EmissionDistribution,
    gpu::{GpuTracer, GpuTransportDevice},
    sources::TransportSources,
    work::{ProbeWork, TraceWork},
    world::{AIR, FACE_OFFSETS, SurfaceProbe, TransportScene},
};
use super::{Profile, reference::ReferencePatch, reference_oracle::channels};
use bevy::{
    platform::collections::HashMap,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task},
};
use serde::Serialize;
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::Instant,
};

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("VOXEL_GI_VALIDATE_RETENTION_AUDIT").is_ok_and(|v| v == "1"))
}

pub fn site(voxel: IVec3) -> Option<(usize, usize)> {
    if !enabled() {
        return None;
    }
    static SITES: OnceLock<HashMap<IVec3, (usize, usize)>> = OnceLock::new();
    SITES
        .get_or_init(|| {
            super::temporal::sites(Profile::Cave)
                .into_iter()
                .enumerate()
                .map(|(i, (v, f))| (v, (i, f)))
                .collect()
        })
        .get(&voxel)
        .copied()
}

pub fn watches_page(page: IVec3) -> bool {
    if !enabled() {
        return false;
    }
    static PAGES: OnceLock<bevy::platform::collections::HashSet<IVec3>> = OnceLock::new();
    PAGES
        .get_or_init(|| {
            super::temporal::sites(Profile::Cave)
                .into_iter()
                .map(|(v, _)| super::super::page_of(v))
                .collect()
        })
        .contains(&page)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SunKind {
    KnownChange,
    FirstMeasurement,
    Unresolved,
    Recovered,
}

impl SunKind {
    /// Called only for a production change event. Unresolved takes precedence
    /// over first measurement, so an exhausted first trace cannot look known.
    pub fn classify(observed: bool, before: [u32; 6], after: [u32; 6]) -> Self {
        if after.contains(&0) {
            Self::Unresolved
        } else if !observed {
            Self::FirstMeasurement
        } else if before.contains(&0) {
            Self::Recovered
        } else {
            Self::KnownChange
        }
    }
}

pub struct SunEvidence {
    pub position: IVec3,
    pub kind: SunKind,
}

#[derive(Clone, Serialize)]
pub struct Decision {
    pub since: u64,
    pub revision: u64,
    pub cap: u32,
    /// First four are exhaustive production mechanisms. The last four are
    /// counterfactual raw-point caps; overlapping causes must not be summed.
    pub caps: [u32; 8],
    pub distance_squared: [i64; 6],
    /// Read-only long-journal counterfactual, never used by transport. None
    /// means the diagnostic journal itself cannot reconstruct this interval.
    pub unfolded: Option<Box<Decision>>,
}

impl Decision {
    pub fn new(since: u64, revision: u64) -> Self {
        Self {
            since,
            revision,
            cap: u32::MAX,
            caps: [u32::MAX; 8],
            distance_squared: [i64::MAX; 6],
            unfolded: None,
        }
    }
}

#[derive(Serialize)]
struct Publication {
    #[serde(skip)]
    at: Option<Instant>,
    at_s: f64,
    site: usize,
    before: u32,
    retained: u32,
    sequence: u32,
    selected: bool,
    source_changed: bool,
    model_restart: bool,
    sky_current: bool,
    decision: Decision,
}

#[derive(Default)]
pub struct Recorder {
    records: Vec<Publication>,
    overflow: bool,
}

impl Recorder {
    pub fn record(
        &mut self,
        job: &ProbeWork,
        before: &SurfaceProbe,
        at: Instant,
        sky_current: bool,
    ) {
        let Some(decision) = &job.retention_audit else {
            return;
        };
        let Some((site, face)) = site(job.voxel) else {
            return;
        };
        if self.records.len() >= 200_000 {
            self.overflow = true;
            return;
        }
        self.records.push(Publication {
            at: Some(at),
            at_s: 0.0,
            site,
            before: before.samples[face],
            retained: before.samples[face].min(decision.cap),
            sequence: before.sequence[face],
            selected: job.trace_mask & (1 << face) != 0,
            source_changed: before.source_revision != job.source_revision,
            model_restart: job.restart,
            sky_current,
            decision: (**decision).clone(),
        });
    }

    pub fn save(
        &self,
        output: &Path,
        started: Instant,
        summary: Option<super::super::retention::SummaryMode>,
    ) -> std::io::Result<()> {
        if !enabled() {
            return Ok(());
        }
        if self.overflow {
            return Err(std::io::Error::other("retention audit overflow"));
        }
        let events: Vec<_> = self
            .records
            .iter()
            .filter(|r| r.at.unwrap() >= started)
            .map(|r| {
                let mut value = serde_json::to_value(r).unwrap();
                value["at_s"] = json!(r.at.unwrap().duration_since(started).as_secs_f64());
                value
            })
            .collect();
        std::fs::write(
            output.join("retention-audit.json"),
            serde_json::to_vec(&json!({
            "schema_version":4,
                "spatial_retention":summary.map(|_| super::super::retention::POLICY_NAME),
                "retention_summary":summary.map(|s| s.name()),
                "sites":super::temporal::sites(Profile::Cave).iter().map(|(v,f)| json!({"voxel":v.to_array(),"face":f})).collect::<Vec<_>>(),
                "cap_labels":["geometry","sunlight_coalesced","folded_geometry","folded_sunlight","sun_known_change","sun_first_measurement","sun_unresolved","sun_recovered"],
                "distance_labels":["geometry","sunlight_coalesced","sun_known_change","sun_first_measurement","sun_unresolved","sun_recovered"],
                "events":events,
                "note":"Accepted jobs only, before fresh paths are merged. Retained=min(authoritative weight, spatial cap); source/model changes are flagged separately. First four causes are exhaustive; raw-corner counterfactual causes can overlap. History weights are not literal stored rays. Instrumented headless cadence is not gameplay FPS."
            }))?,
        )
    }
}

struct SceneSample {
    at_s: f32,
    generation: u64,
    source_revision: u64,
    scene: Arc<TransportScene>,
    sources: TransportSources,
    materials: Arc<VoxelMaterialTransportTable>,
}

#[derive(Default)]
pub struct SceneSnapshots {
    samples: Vec<SceneSample>,
}

impl SceneSnapshots {
    pub fn record(&mut self, at_s: f32, runtime: &VoxelGiRuntime) {
        if !enabled() || self.samples.len() == 6 || at_s < 30.0 * (self.samples.len() + 1) as f32 {
            return;
        }
        self.samples.push(SceneSample {
            at_s,
            generation: runtime.generation,
            source_revision: runtime.source_revision,
            scene: runtime.scene.clone(),
            sources: runtime.sources.clone(),
            materials: runtime.materials.clone(),
        });
    }
}

pub fn start_reference(
    snapshots: SceneSnapshots,
    device: Option<&GpuTransportDevice>,
    settings: &VoxelGiSettings,
    output: PathBuf,
) -> Result<Task<Result<(), String>>, String> {
    if !crate::headless_validation() || std::env::var_os("VOXEL_GI_VALIDATE_EXIT").is_none() {
        return Err("Retention oracle requires headless validation with EXIT".into());
    }
    if snapshots.samples.len() != 6 {
        return Err("Retention audit requires all six long-cave checkpoints".into());
    }
    let device = device
        .ok_or("Retention audit requires the native GPU")?
        .clone();
    let transport = settings.transport;
    Ok(AsyncComputeTaskPool::get().spawn(async move {
        let sites = super::temporal::sites(Profile::Cave);
        let patch = ReferencePatch::new(&sites, |voxel, face| {
            snapshots.samples.iter().all(|s| exposed(&s.scene, voxel, face))
        });
        for sample in &snapshots.samples {
            if sample.source_revision != snapshots.samples[0].source_revision {
                return Err("Source/model changes invalidate this geometry-only oracle comparison".into());
            }
            for &(voxel, face) in &patch.sites {
                if !exposed(&sample.scene, voxel, face) { return Err(format!("Missing audit receiver {voxel:?}/{face}")); }
            }
            // Do not silently prune changing corner topology into a common blur.
            let full = ReferencePatch::new(&sites, |v,f| exposed(&sample.scene,v,f));
            if full.sites != patch.sites || full.corners != patch.corners {
                return Err("Corner topology changed across oracle checkpoints".into());
            }
        }
        let mut gpu = GpuTracer::new(device)?;
        let mut records = Vec::new();
        for sample in snapshots.samples {
            let emitters = Arc::new(EmissionDistribution::build(&sample.scene, &sample.materials));
            let mut replicas = Vec::new();
            for replicate in 0..2 {
                let mut work = TraceWork { transport, ..default() };
                for &(voxel, face) in &patch.sites {
                    let mut probe = SurfaceProbe::new(voxel, 1 << face);
                    probe.sequence[face] = 4_194_304 + replicate * 1_048_576;
                    work.push(&probe, 1 << face, 0, 0, false, u32::MAX, u32::MAX);
                    work.probes.last_mut().unwrap().sky_requested = false;
                }
                let mut mean = vec![[0.0_f64; 5]; patch.sites.len()];
                for _ in 0..2 {
                    for face in &mut work.faces { face.samples=0; face.radiance=default(); face.second_moment=default(); }
                    for _ in 0..32 {
                        gpu.trace_compact(sample.scene.clone(), &sample.sources, &sample.materials,
                            &mut work, 1024, &emitters, false)?;
                    }
                    for (sum, face) in mean.iter_mut().zip(&work.faces) {
                        for (sum, value) in sum.iter_mut().zip(channels(face.radiance)) { *sum += f64::from(value) / 2.0; }
                    }
                }
                replicas.push(mean);
            }
            let corners: Vec<_> = replicas.iter().map(|r| patch.reconstruct(r)).collect();
            records.push(json!({"at_s":sample.at_s,"generation":sample.generation,"regions":sample.scene.regions.len(),
                "source_revision":sample.source_revision,"patch_references":replicas,"corner_references":corners}));
            info!(at_s=sample.at_s, "Retention frozen-scene oracle complete");
        }
        std::fs::write(output.join("retention-reference.json"), serde_json::to_vec(&json!({
            "schema_version":1,"paths_per_replicate":65536,"sequence_start":4194304,"replicate_stride":1048576,
            "transport":transport,"sites":sites.iter().map(|(v,f)|json!({"voxel":v.to_array(),"face":f})).collect::<Vec<_>>(),
            "patch_sites":patch.sites.iter().map(|(v,f)|json!({"voxel":v.to_array(),"face":f})).collect::<Vec<_>>(),
            "corner_members":patch.corners,"checkpoints":records,
            "note":"Six exact immutable runtime scenes; production finite-depth estimator without history/scheduling. Same two disjoint sequence windows at every checkpoint reduce difference noise, not formal confidence intervals. Checkpoints do not resolve every intervening event. No whole-image or nonvisible-band accuracy certificate."
        })).unwrap()).map_err(|e|e.to_string())
    }))
}

fn exposed(scene: &TransportScene, voxel: IVec3, face: usize) -> bool {
    scene.material_at(voxel).is_some_and(|m| m != AIR)
        && scene.material_at(voxel + FACE_OFFSETS[face]) == Some(AIR)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn evidence_distinguishes_measured_change_from_missing_knowledge() {
        assert_eq!(
            SunKind::classify(false, [0; 6], [1; 6]),
            SunKind::FirstMeasurement
        );
        assert_eq!(
            SunKind::classify(false, [0; 6], [0; 6]),
            SunKind::Unresolved
        );
        assert_eq!(SunKind::classify(true, [1; 6], [0; 6]), SunKind::Unresolved);
        assert_eq!(SunKind::classify(true, [0; 6], [1; 6]), SunKind::Recovered);
        assert_eq!(
            SunKind::classify(true, [1; 6], [2; 6]),
            SunKind::KnownChange
        );
        assert_eq!(
            SunKind::classify(true, [2; 6], [1; 6]),
            SunKind::KnownChange
        );
    }
}
