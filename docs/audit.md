# Architecture audit — October 8, 2026

Historical checkpoint. The [October 10 scene-history audit](scene-history-audit.md)
supersedes the emitter-reset rationale below with controlled removal/streaming
evidence. The global voxel-emitter reset and its material-delta bookkeeping have
been removed; analytic-source and model-replacement policies remain explicit.

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
[research decisions](research.md#why-indirect-corners-are-still-an-experiment).

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
configuration in [validation](validation.md#shared-cargo-cache). No simultaneous
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
