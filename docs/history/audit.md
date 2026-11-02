# Architecture audit — October 8, 2026

Historical checkpoint. The [October 10 scene-history audit](scene-history-audit.md)
supersedes the emitter-reset rationale below with controlled removal/streaming
evidence. The global voxel-emitter reset and its material-delta bookkeeping have
been removed; analytic-source and model-replacement policies remain explicit.
The [October 14 external-review follow-up](#october-14-external-review-follow-up)
records later display-capacity/publication fixes and remaining review findings.
The [focused follow-up review](#october-14-focused-follow-up-review) records the
current readiness assessment and the next measured work-removal checkpoints.

Baseline: `831931d`. Production cleanup and corner census: `8546ef3`. Final
offscreen-harness correction: `ad6ba8e`. All commits are local. The estimator,
four-path dispatch, temporal filter and preferred corner appearance are unchanged.
This is a completed code/design audit, not a claim that every future world size,
integration or lighting distribution is production-proven.

## Review coverage

| System | Conclusion / action |
| --- | --- |
| Geometry storage, NAADF, CPU/GPU traversal, packing | Keep independent CPU reference and incremental compressed GPU scene. Consolidate scene-delta scans. Preserve occupied-frontier semantics. |
| Sources, materials, emission sampling, spectral transport | Source snapshots have their own module. Canonical ordering and complete bounded source-change history fix unnecessary/missed relights. Preserve multi-band MIS and physical/display separation. |
| Revision state, scheduling, workers, publication | Shared journal mechanism; explicit publication/dispatch modules. Compact work packets retain their current ownership and stale-result rules. |
| Temporal display, corner reconstruction, buffer lifetime | Keep tested filter and connected-coplanar reconstruction. Statistics no longer belong to the presentation module. Retain generation-owned buffers and transactional patches. |
| Host terrain, meshing, noise, AO, collision, input and environment | Keep prototype responsibilities outside core GI. Host owns texture handle, array preparation, palette and reload. Correct repeated Modified-event loop. |
| Diagnostics, telemetry and validation | UI remains optional. Replace obsolete test-only merge path with the real publication contract. Make offscreen shadow-camera ownership explicit. |
| Dead paths and dependencies | Remove duplicate preparation/scans and obsolete topology-preservation test helper. Retain cfg(test) oracles and opt-in native references deliberately. Strict all-target Clippy is clean. |
| Research and future estimator | Compare primary sources and distinguish implemented techniques from proposals. Count actual oriented corner states before promoting a new estimator. |
| Documentation | Current architecture, research decisions and verification are separate from archived historical experiments. Source/API documentation builds with warnings denied. |

The review includes runtime Rust modules, both GI shaders and the prototype host,
plus relevant regression fixtures. It does not claim every optional Bevy feature
or transitive dependency is necessary: dependency-feature trimming needs its own
startup/render compatibility check. No dependency or existing public query API
was removed merely because the current demo has one consumer.

## Coherent checkpoints

| Commit | Change and rationale |
| --- | --- |
| `d0a030f` | Move common relative-error heuristics to `statistics.rs`; scheduling must not depend on temporal presentation. |
| `6042727` | Give analytic source values and Bevy adaptation their own owner, outside the CPU tracer. Pure movement first. |
| `8c114fc` | Reuse snapshot scratch storage; canonicalize coincident lights; journal every intervening source delta and share bounded journal machinery with geometry. Expose host-configurable environment energy. |
| `5476aae` | Split root orchestration into `publication`, `dispatch`, `display` and integration tests. Exercise actual `TraceWork` publication after topology compaction, removing the obsolete helper. |
| `2170db3` | One `SceneChanges` traversal determines identity changes, material IDs and geometry bounds instead of overlapping comparisons. |
| `5279f4e` | Repair a stale streaming-gap fixture: all-air frontiers escape; a missing region before occupied loaded geometry is unknown. Traversal implementation was not changed. |
| `3f44dc4` | Host supplies the texture handle. Check readiness before mutably touching an image; remove duplicate preparation and the Modified-event feedback loop. Preserve reload and layer-change support. |
| `8546ef3` | Count face/corner states in flat, hard-edge, disconnected and actual-house fixtures. No production estimator change. |
| `ad6ba8e` | Explicit offscreen capture-camera shadow LOD origin; remove Bevy's fallback warning without changing windowed setup. |

The source-history fix matters beyond the current room tests. Previously a face
that missed multiple moving-light changes could consult only the newest influence
region. Now it considers the complete retained chain; gaps/evictions use a
conservative policy. Snapshot ordering also no longer creates a light change when
coincident entities are merely enumerated differently.

The texture regression proves the host becomes idle after preparation, processes
a reload once, and can change array-layer count without altering pixel count.
That is removed work, not an inferred frame-rate percentage.

## Boundaries deliberately retained

- **Mechanism is shared; policies may differ.** Geometry edits and streaming use
  one comparison/journal path. Analytic-source deltas use the same bounded journal
  but different influence/history limits. Retaining obsolete emitted energy and
  retaining visibility evidence are not identical problems.
- **Direct sun and indirect GI share infrastructure, not an estimator.** One
  known-direction visibility result differs from a noisy hemisphere integral.
  Both use the same geometry and asynchronous submission infrastructure.
- **Measurements, freshness, urgency and display are separate state.** Display
  seeds never masquerade as measured energy; simulation queries expose staleness.
  History can remain visible while selected faces acquire fresh measurements.
- **No gameplay fixture controls the solver.** Houses, tunnels, doors and emitter
  test IDs belong in terrain/validation. The production rules use geometry,
  optical/source deltas, connectivity, evidence and timestamps.
- **No new debounce or raw-history extension.** Preserve the stable visual
  baseline so later speed work has an interpretable comparison.

Primary indirect corners are not promoted. In the actual house selection,
adapting the current oriented estimator requires 1,324 connected corner states
versus 1,019 face states. This does not rule out a different directional corner
representation; it rules out assuming a free sample-count reduction. See the
[research decisions](../research.md#indirect-corners).

## Verification

Focused verification is recorded in each code commit. At `8546ef3`:

- `cargo test --locked --offline --all-targets`: **130 passed, 13 opt-in ignored**.
  The first integrated run exposed the stale all-air-frontier expectation; the
  fixture was corrected in its own commit and the final suite passed.
- Native `gpu_matches_cpu`: passed on Apple M4 / Metal, reported RMS 0.000000 for
  both the standard and enclosed-emissive fixtures. Not a whole-game benchmark.
- Native headless material/cache-generation regression passed at `3f44dc4`;
  subsequent code before the capture-camera correction only adds a CPU census
  test, so that generation-lifetime evidence is reused, not claimed as rerun.
- Strict all-target Clippy, development build and warning-denied private-item
  rustdoc passed. Clippy/build were repeated after `ad6ba8e`; the regular transport
  suite was not rerun for that capture-camera-only change.

### End-to-end evidence

Machine: Apple M4, Metal, Rust 1.95.0. Development profile with the shared-cache
configuration in [validation](../validation.md#shared-cargo-cache). No simultaneous
compiler/GPU benchmark. Captures use Bevy offscreen 2560 × 1440, 60 Hz schedule
runner, no OS window/swapchain. Textures on, AO off, shared-corner sun and
corner-reconstructed indirect; seed 7, terrain period 256. Defaults: 4 ms trace
budget, 8,192 max receivers, 2 ms nominal page publication, 75 ms display response,
50 ms display upload, 64 minimum stable samples and 2,000 ms stable audit.

House run at `8546ef3`, with documentation-only dirty files:

`target/gi-validation/house-edits-1791482775108682000-60796/`

- 180 seconds, 9/9 valid nonblank captures, assets verified, no warnings/errors.
- Confirmed edits in `house.json`: one roof voxel and three voxels in each wall;
  the final turn grew loaded terrain from 662 to 959 pages. The generic report's
  `edits` counter does not include this separate house fixture's events.
- 106,338,240 traced paths; final coverage 100%. These are progress measures, not
  proof of convergence. Reviewed early/late skylight and late turned-corner images
  show textured dark interiors with smooth corner lighting, not black captures.
- Skylight / west-door / east-door-turn offscreen cadence p99: 26.38 / 28.73 /
  30.70 ms. Corresponding sampled near-field age p95: 1.94 / 1.99 / 1.94 seconds.
  The first-corner startup maximum was 850.76 ms; the final turn maximum was
  160.37 ms. These tails are retained, not hidden by the percentile summary.

These are integration observations, **not windowed FPS or a measured speedup**.
No fresh baseline/candidate timing pair was run for the structural cleanup. Sparse
screenshots cannot prove every intervening frame is flicker-free, and no new
high-sample image reference was rendered in this audit.

The first source run at `8546ef3`
(`sources-1791482978957165000-63891`) completed with five valid captures but emitted
the shadow-LOD fallback warning. That finding produced `ad6ba8e`; it is retained
as pre-fix evidence, not reported as a clean-render pass.

Corrected source run at `ad6ba8e`, with documentation-only dirty files:

`target/gi-validation/sources-1791483133493020000-66284/`

- 72 seconds, 5/5 valid nonblank captures, assets verified, no warnings/errors;
  active-source image reviewed with the emitter and illuminated interior visible.
- 90,334,880 traced paths; final coverage 100%; maximum 409 loaded pages. The
  workload moves/removes analytic light, changes emissive material and streams.
- Active-source / streaming offscreen cadence p99: 33.87 / 44.43 ms; sampled
  near-field age p95: 140.83 / 171.20 ms. These describe one run, not a comparison
  with the pre-fix run: changed scheduling/streaming timing is not held constant.

Documentation links and whitespace were reviewed. Large captures/logs stay under
ignored `target/`; this compact record survives their eventual cleanup.

## Remaining work is explicit, not hidden in the API

The current system is a library-style prototype, not yet a published generic
engine. BVW input/vertex layout, axis-aligned unit coordinates, Y-up environment,
f32 positions, bounded diffuse paths, heuristic history limits and monolithic
display-buffer capacity remain concrete integration limits. Infrared/ionizing
transport is normalized, opaque and diffuse, not a calibrated physics solver.

Next, profile useful work and test better directional sampling at a fixed ray
budget. Consider GPU-resident statistical state only as a separately measured
storage change. Keep current smoothing fixed during these experiments and compare
brightening, darkening, freshness and frame tails together. The research guide
records the relevant prior work, what is unimplemented and the acceptance criteria.

## October 14 external-review follow-up

Reconciled [the independent source review](CODE_REVIEW_2026-10-13.txt), checked
at `a68a1d8`, against current baseline `885deeb`. These are focused robustness/work
removal fixes, not a new estimator or another retention calibration.

- `8a94f1c` addresses the zero-source part of **F8**: the adapter removes analytic
  sources with no transported or custom visible energy. Invisible bands remain
  active. Four source tests passed, including activation and component removal.
- `63f786d` addresses **F3**: topology has its own required-value ledger and uses
  a nonblocking mirror lock. The 8,192-receiver regression leaves 8,190 unrelated
  values queued while publishing the two required records atomically. Render
  tests: 21 passed; display: 5; publication: 2; native headless generation: 1.
  A preliminary `publication::tests` filter matched zero tests; the corrected
  `publication::` invocation produced the two passes recorded here.
- `c212e30` addresses **F2**: checked capacity planning, constrained optional
  headroom, collision-growth bounds, actual-device revalidation and recoverable
  replacement rejection. This includes CPU transport's GPU display bindings.
  Render/capacity tests: 24 passed, then a separate adversarial hash-growth test
  passed; display: 6; restricted native device and headless generation: 1 each.
  The ordinary dev build passed. Large studies and the full toggle profile were
  not run. These runs overlap earlier tests; their counts are not additive
  coverage claims.

Native checks used Apple M4 / Metal, Rust 1.95, default dev features and the shared
Cargo target. There is no gameplay viewport or frame-time comparison in these
tests. The workload bound and zero-source omission are proven by state/byte tests;
no FPS improvement, faster visual settling or all-residency support is claimed.
The existing whole-range policy, transport samples, presentation and BVW are
unchanged. Capacity checks prevent unsupported allocations; they are not general
physical-memory exhaustion recovery. See the current display contract in
[architecture](../architecture.md#presentation).

Subsequent input/delta work is recorded below. This table records disposition
at that checkpoint; the [focused follow-up](#october-14-focused-follow-up-review)
subsequently addresses F8 and parts of F7/C1. Other findings are not called fixed:

| Findings | Disposition |
| --- | --- |
| F4–F5: live per-voxel ingestion and repeated delta discovery | Implemented in `ce17a4c`, `254234d`, `f4f1847`: tracked immutable input and a bounded shared region-change journal. Conservative fallback remains for unknown input or expired lineage. No BVW modification. See qualification below. |
| F1, F7, C3: input/configuration/runtime ownership | Valid architecture work, separate from sampling changes. Introduce typed host configuration and a GI-only harness incrementally; do not extract a generic engine or rewrite all state in this patch. |
| F6: light visibility | Real adapter mismatch; decide explicit GI/simulation enable semantics before implementation. Camera culling must never disable persistent world lighting, and hiding a visible object need not disable its heat/radiation. |
| F8: rebuilding static analytic-source lists | Still present. Change-driven synchronization must include component removal/despawn and later visibility semantics. Zero-source filtering does not solve this cost. |
| C1–C2: unused release helpers, diagnostic mirrors, duplicate slot maps | Valid cleanup candidates; not evidence of costly dead solver code. No blanket dead-code allowances or unmeasured memory claims. |
| C4: Python dependency/test entry point | Keep as harness reproducibility work before onboarding another analysis environment. This round used Rust/native checks, not Python analysis. |
| C5: shader lookup/guard cleanup | Profile separately; retain corner reconstruction and native parity oracles. |

The external review's independent CPU/GPU references, generation-owned buffers,
immutable scene/revision contracts and display/simulation separation remain
intentional. None should be removed merely to reduce file count or duplication.

### October 14 immutable-input and shared-delta qualification

Baseline: clean `5364955`. Three local, buildable code checkpoints:

- **`ce17a4c`**: exact edit gateway, copy-on-write page overlays and immutable
  chunk input. Tests cover last-write-wins, negative/boundary coordinates,
  in-flight snapshots, unload/reload and unknown-input fallback.
- **`254234d`**: a later unknown notification supersedes an earlier queued exact
  value in the same frame. Three adapter tests passed; shared-delta work was
  present but uncommitted during this targeted test run, and is not claimed to
  have been tested by those adapter checks.
- **`f4f1847`**: share bounded changed-region lineage across geometry, sunlight,
  emitter dependencies and GPU packing. Tests compare skipped/expired/forked
  versions with full discovery, prove retired geometry is not retained by the
  journal, compare exact packed bytes/upload ranges, and preserve distant sun
  blocker invalidation. World tests: 9 passed; packing: 6; sunlight: 9; emission:
  7; geometry: 2. Opt-in packing cost study and native GPU/CPU parity also passed.
  Standard and enclosed-emissive parity fixtures reported RMS 0.000000. Dev
  builds and strict binary Clippy passed; the full suite and optional toggle
  stress were not rerun. Clippy was run after the code commits, with only
  documentation changes in progress.

The [current input contract](../architecture.md#host-input-contract) requires all
tracked-mode writers to use the gateway or report unknown input. Sparse edits
currently persist across unload/reload and consume memory per distinct edited
position. The eight-transition delta journal is key-only and bounded in versions,
not a new lighting history policy. Full fallback remains necessary; top-level
map cloning, topology-dependent header refresh and directional sun dependency
scans are not claimed removed.

Isolated work reductions and native background-streaming/edit evidence are in
[the performance record](streaming-performance.md#october-14-immutable-input-and-shared-delta).
Transport, whole-range retention, display smoothing, preferred corner modes,
existing coalescing windows and BVW are unchanged. This is not a claim of a
whole-game speedup, faster convergence or a new 60-FPS guarantee.

## October 14 focused follow-up review

Baseline **`aa664d7`**, branch `codex/gi-whole-range-runtime`. This is an independent
focused pass after the external-review fixes, not acceptance of every prior
suggestion. Reviewed production ownership and hot paths in input, scene/delta
construction, sources/materials, retention, scheduling, worker transfer/merge,
packing, sunlight, display publication, diagnostics and the host integration.
Checked those contracts against the primary sources recorded in
[research](research-design-notes-2026-10.md#october-14-focused-source-and-production-comparison).
No BVW, shader, estimator, history-curve, smoothing or debounce change.

### Implemented findings

- **`22bc667`: avoid rebuilding static analytic sources.** A change gate covers
  environment, source components, transforms and removals before normalization
  and canonical sorting. Reclassification and reactivation remain correct.
  This addresses the rebuild part of F8, not visibility semantics or analytic
  ray cost. ECS change-tick scanning remains; voxel torches use a different path.
- **`22199bf`: bound local history-plan work.** Select local page enumeration
  when estimated cheaper, share occupancy checks across overlapping changes,
  and retain the full resident scan as a fallback. Exact bounds, ordering and
  minimum distances match the prior implementation. This accelerates geometry
  and measured-sunlight evidence without changing invalidation policy.
- **`e5b5ef8`: typed per-application construction.** `VoxelGiStartup` owns backend
  and retention choices. The demo parses launch flags; the plugin no longer
  reads those variables or panics on their invalid values. Two independent App
  instances test independent policy. This partially addresses F7, not every
  debug-harness global or BVW/host assumption.
- **`dbae7a0`: remove duplicate diagnostic state.** F4 and the panel share the
  material's enable state; an external material edit can no longer leave the
  keyboard mirror stale. Compact jobs own production selection masks; persistent
  probe masks exist only for test oracles. Detailed timing is enabled by the
  validation adapter, and diagnostic-only helpers are excluded from normal
  release-gated code. Six existing test-only Clippy findings were corrected
  without blanket allowances. No unmeasured memory saving is claimed.

### Verification and evidence

The final code checkpoint passes **251 regular tests** (60 opt-in native/large
studies deliberately ignored), strict all-target Clippy, the dev build and
private-item rustdoc with warnings denied. Application compilation with debug
assertions disabled and warnings denied also passes; this uses dev dependencies
and is not a release-performance build. Native CPU/GPU geometry-edit parity
passes with reported RMS 0.000000 in standard and enclosed-emissive fixtures.
Focused tests cover removals/reclassification, negative coordinates, empty and
partial residency, overlapping bounds, extreme spans and large-brush fallback.
The optional toggle-stress workload was not rerun; buffer lifetime was not changed.

Stage costs and the native streamed-edit run are recorded in
[performance](streaming-performance.md#october-14-focused-review-work-removal).
Logs remain under ignored `target/audit-*`. Initial compilation/lint findings
were repaired before these final passing checks; no failed run is counted as a
pass. Documentation changes were in progress during final native checks.

### Readiness and remaining limits

**Solid base for the next controlled experiment; not a finished generic engine.**
The production data flow has coherent owners and tests for stale work, unknown
input, sparse updates, retention and renderer generations. Direct sun and diffuse
transport intentionally have different estimators; CPU/GPU references and
display/measurement separation are necessary duplication, not dead code.

Remaining work is explicit and does not justify another unmeasured rewrite:

- Full region-map cloning, occupancy-dependent top-level rebuilds, directional
  sun dependency scans and some receiver planning still scale with residency.
  Publication is cooperatively budgeted; one atomic page can exceed its budget.
- GPU display uses two key-to-slot maps and monolithic capacities. Consolidating
  indices or adding paging needs topology/generation qualification, not just a
  smaller struct. GPU-resident measurement state with bounded tick delivery is
  the best already-supported next architecture experiment; keep explicit paths.
- Tracked overlays need authoritative host acknowledgment before safe retirement.
  Runtime orchestration is still a broad state owner; the crate remains tied to
  one axis-aligned BVW world, not arbitrary planets or multiple world IDs.
- Analytic light enable semantics, CPU-visible/invisible-band policy and the
  debug harness's process globals remain open contracts. Do not use camera
  visibility to disable world simulation energy. Analytic evaluation is still
  linear in source count after the snapshot optimization.
- No renewed proof of every-world visual convergence or always-60-FPS gameplay
  is claimed. Keep perceptual response, overshoot, sample freshness and frame
  tails in the next experiment; coverage alone cannot validate it.

Current architecture and validation docs now describe the actual whole-range
equation and distinguish it from archived controls. Research docs identify which
source versions were read and which newer papers were only screened by abstract
and integration guide. This is the handoff point for further experiments, not a
reason to restart history calibration or remove the stable presentation filter.
