# Two-way view/world RGB cache

Decision, 2026-10-22: **keep the two-way design in the +cache experiment** on
`codex/two-way-face-cache`. It substantially improves reveal/return stability,
preserves the cave's appearance and keeps offscreen emitter response responsive.
Do not replace the hybrid default yet: windowed frame pacing is unqualified.

Run it with:

```sh
VOXEL_GI_SETTINGS=config/gi-view-cached.json cargo run --locked
```

## Design and boundaries

The visible and background schedulers share one authoritative raw RGB record
per resident face. First admission transfers the better-supported valid estimate
without summing overlapping histories; subsequent visits always load the
authority, including its edit-retained count. View completions update that same
record, and newly revealed faces can import it. A separate display prior avoids
cold presentation without turning reconstructed support into physical evidence.

The residency radius remains 32 voxels, with 32-voxel retirement hysteresis;
the total transport share and 25% world / 75% view split are unchanged. This
does not add persistent records for all loaded terrain or implement spatial
path reconnection. IR/radiation is outside GI's gameplay contract.

GPU generation guards reject stale render writes across maintenance. Both
schedulers advance one sample sequence; unfinished admitted paths keep their
reservation. Unstarted reservations are released within the same command buffer.
An independent, pre-admission tail is cancelled when importing another prefix,
since it may duplicate a sample already in that history. There is no lighting
CPU readback/merge or player-thread wait. See the full
[ownership contract](../architecture.md#experimental-two-way-face-cache).

## Qualification

Control: `913ab35`, preserved in the managed `two-way-cache-baseline` worktree.
`c3494f8` adds only the opt-in cache-admission hallway workload to that control;
house, response and cave transport are identical. Candidate implementation:
`a4384a5`, rank-refresh cleanup `c21e4ee`, sequence/tail correction `bd12f0e`,
and independent admission-tail guard `594f37b`.

M4 / Metal, 16 GB, rustc 1.95.0; development opt-level 1, dependencies opt-level
3; 2560×1440 offscreen targets, background terrain loading, shared-corner sun,
corner indirect, textures on, baked AO off, uncapped 1024-evidence display.
No other game, build or image-analysis job ran during measured phases. The
candidate's dirty changes were documentation and formatting of `handoff.rs`
only; those do not change the workload. Native tests/builds ran separately.

All accepted runs have valid varied captures, the intended camera/target and
clean render logs. Images were inspected as well as measured. Official FLIP
1.7 at 67 pixels/degree compares each early capture with its own later capture;
these are **remaining visible change, not accuracy against a converged oracle**.
House uses the upper 40% (ceiling), excluding the outdoor doorway. Other profiles
use full images. No raw sampler threshold or reconstruction target is changed.

| Remaining change, mean FLIP | One-way control | Two-way |
| --- | ---: | ---: |
| House, 400 ms after turn | .03837 | .01032 |
| House, 400 ms after return | .04611 | .00429 |
| Cave, 15 s → own 60 s | .01039 | .01000 |
| Cave, 30 s → own 60 s | .00624 | .00609 |
| Visible cache admission, 400 ms → own later | .01757 | .01266 |
| Visible cache admission, 1 s → own later | .00801 | .00975 |
| Visible cache admission, 2 s → own later | .00568 | .00470 |
| Offscreen emitter add, 1 s (two runs) | .02086 / .02904 | .01143 / .01279 |
| Offscreen emitter remove, 600 ms (two runs) | .02208 / .15636 | .02455 / .01651 |
| Offscreen emitter remove, 1 s (two runs) | .00884 / .07916 | .00634 / .00785 |

House reveal and return improve 3.7× and 10.7× at the captured 400 ms offsets.
These are not continuous settle-time measurements. The final cross-run cave
difference is .00565 mean FLIP; displayed mean luminance differs by 0.034%.
Admission is mixed at individual early offsets, not a universal speedup:
the important contracts are successful transfers (1,045 promotions counted),
no cold clear, and continuous visible history. This is not a claim that the
unchanged hybrid's boundary handoff has been fixed.

Emitter removal at 300 ms was worse in the first pair (.31355 vs .25212), but
better in the repeat (.30710 vs .48468). By 1 s both candidate runs are close to
their own four-second final image. The baseline varies more; do not claim a
precise removal speedup or a repeatable early regression from these pairs.
Final response images differ by mean FLIP .00529/.02554 (off) and .01733/.02472
(on). These are live finite-time comparisons, not certified physical solutions.

The new GPU exchange passes cost median .059 ms house, .093 ms cave, .056 ms
response and .066 ms admission (logged samples; full range .042–.113 ms).
They add at most 256 KiB of per-camera sequence scratch, no second persistent
face history. Whole-process peak MB: house 976→981, cave 966→967, response
937→942 / 941→944, admission 915→930. These allocation peaks vary with streaming
and scheduling; they are not a static-memory accounting of the bridge alone.

| Offscreen app cadence p95 / p99, ms | Control | Two-way |
| --- | ---: | ---: |
| House | 25.36 / 27.09 | 25.24 / 27.78 |
| Cave | 26.28 / 27.49 | 25.97 / 27.04 |
| Admission | 25.46 / 26.92 | 25.36 / 26.80 |
| Response repeat | 25.18 / 25.93 | 24.95 / 25.84 |

No material offscreen cadence regression is apparent; no small performance gain
is claimed. The windowed admission attempt at `594f37b` stopped on its first
capture (`focused=false`, `varied=false`). It is excluded, not an FPS pass.

## Correctness gates and rejected intermediate designs

- Five native ownership tests cover adoption, non-double-counting, raw/display
  separation, material identity, stale generations, retained weights and pending
  change exclusion from bounce lookup, alternating schedulers and long tails.
- A raw analytic furnace reaches RGB `(1.000815, 2.891734, .666205)` versus
  `(1, 2.888889, .666667)`, under 0.1% worst relative error after 172,442 sequence
  indices. It uses no reconstruction. The count also proves that no unstarted
  reservation becomes a sample or a path-length-dependent sequence gap.
- Restarting unfinished view paths whenever background advanced the cursor was
  rejected: it preferentially discarded long paths. Reserving indices fixed
  ownership, but keeping unused reservation suffixes failed the analytic gate
  (~1.7% blue error). Returning that suffix fixed the new experiment's defect;
  this is not a newly discovered defect in the baseline sampler.
- Raw view-batching parity, single/MSAA runtime pipelines, headless material
  generation integration, 244 regular tests, production Clippy `-D warnings`
  and development build passed at `bd12f0e`. At `594f37b`, the extended native
  admission/pending-tail test, shader composition and build passed; unchanged
  analytic/ownership evidence is reused. The guard only cancels an unowned
  first-admission tail, not already-admitted pending work.
- A test harness initially encoded thousands of compute passes into one Metal
  command buffer and lost its test device. The analytic test now submits bounded
  batches like the runtime. Failed runs were not used as rendering evidence.

## Reproduction and local evidence

Use the commands in [validation](../validation.md), selecting `house`, `response`,
`cave` or `hallway` with `VOXEL_GI_VALIDATE_CACHE_HANDOFF=1` for admission.
`tools/view_cache_quality.py` generates paired pictures and JSON from actual
material captures, not CPU world-field diagnostics. Large artifacts stay ignored.

Run directories below are under each worktree's `target/gi-validation/`:

| Profile | Control run | Candidate run / commit |
| --- | --- | --- |
| House | `house-1792675968064839000-14521` (`913ab35`) | `house-1792678688078865000-17146` / `bd12f0e` |
| Cave | `cave-1792679226901199000-18710` | `cave-1792678872456592000-17781` / `bd12f0e` |
| Admission | `hallway-1792676727882402000-15590` | `hallway-1792679083432038000-18208` / `594f37b` |
| Response 1 | `response-1792677177270594000-16273` | `response-1792678808925171000-17593` / `bd12f0e` |
| Response 2 | `response-1792679290399488000-18697` | `response-1792679351662110000-18982` / `594f37b` |

Other controls are `c3494f8`. Summaries/images are in the candidate's
`target/two-way-final-comparison/` and `target/two-way-repeat-comparison/`.
Logs are `target/two-way-{final,repeat}-*.log`; native gates use
`target/two-way-{native,view,material,unit,clippy}-final.log` and
`target/two-way-admission-tail.log`. Earlier preliminary pairs are retained but
are not substituted for the measurements above.

## Grazing-face coverage correction (2026-10-22)

The reduced grid's fixed pixel centres could permanently miss a thin visible
plane. Cache admission then appeared to fix it because the world cache supplied
the previously unrepresented face. Discovery now scans each 2x2-cell footprint
cooperatively and trades redundant plane samples for missing planes. Selection
uses deterministic distance/slot/pixel ordering and verifies transport geometry
before replacement. Broad flat surfaces retain their original centre identities.
Ray allowance, persistent buffer capacity, raw sampling and reconstruction rules
are unchanged. Four slots cannot represent unlimited subpixel complexity.

The native fixture reproduces the old miss with one-pixel-wide side planes:
the `23a3775` control selects only the background, while the correction represents
both side orientations and the background. It also covers odd grid edges,
unchanged flat identities and raster geometry absent from the transport scene.
Native single/MSAA pipelines, shader composition, rank bounds through 16K
viewports and production Clippy pass. Release build passes with the existing 33
release-only unused/dead-code warnings (`target/grazing-release.log`). This is
identity/coverage evidence, not a rendered lighting oracle.

Cave pair: control `23a3775`; candidate `1a0516d` plus this discovery change,
before documentation/test-only finishing changes. Same M4/Metal, development
profile, 2560x1440 offscreen target, +cache settings and background loading as
above. Runs are `cave-1792685114166712000-25726` (control worktree) and
`cave-1792685416097221000-26096` (candidate); comparison is under
`target/grazing-comparison/`. All captures are valid and render logs clean.

- Cave 15s-to-own-60s mean FLIP: .00999 -> .00930; 30s: .00609 -> .00547.
  Final cross-run mean FLIP .00348, mean display luminance ratio 1.00002.
- Logged view presentation GPU cost median .632 -> 1.578 ms. A rejected serial
  scan cost 4.429 ms; the retained implementation classifies pixels in parallel.
  This is added coverage work, not a free improvement or a transport-budget change.
- Offscreen cadence p95/p99: 26.07/26.82 -> 22.89/23.54 ms. One pair excludes an
  obvious regression here; it does not establish a repeatable performance gain
  or windowed frame pacing. Whole-process peak allocation: 966 -> 944 MB; discovery
  adds only per-workgroup scratch, not a new persistent allocation.

The same development +cache pair also ran the 35-second `streaming` workload
(turn 180 degrees, move across terrain, stop). Both sets of three captures are
valid and visually inspected, with clean render logs. After 15s, excluding capture
frames, offscreen cadence p95/p99 is 21.80/24.09 -> 21.46/22.23 ms; whole-process
peaks 864 -> 865 MB. Final logged visible owners increase 2058 -> 2453 without
growing the grid; this measures representation, not sample independence or
convergence. Captured moving cameras differ slightly in position, so no raw
pixel/FLIP quality claim is made from that pair. Runs:
`streaming-1792685693713156000-26645` (control worktree) and
`streaming-1792685805050648000-26850` (candidate). Logs are
`target/grazing-{baseline,candidate}-streaming.log` in their respective worktrees.

Native logs: `target/grazing-{native,pipelines,rank,clippy,shader}.log`.
Cave logs: `target/grazing-baseline-cave.log` in the control worktree and
`target/grazing-parallel-cave.log` here. The initial serial attempt remains in
`target/grazing-candidate-cave.log`; do not substitute it for the retained design.

## Amortized discovery (2026-10-22)

Keep compact plane classification plus memoized, four-phase footprint checks.
The grid may move receivers between slots because transport history is keyed by
exact face, but simply jittering away from a thin face would drop its one-frame
view history. Instead, retain discovered receivers between checks and keep every
lighting stage running. Moving cameras still scan immediately; a new plane in a
stationary view waits at most three intervening frames (50 ms at 60 Hz). Existing
faces do not disappear between phases. Geometry-invalid selections and pending
transport uploads force a retry, independently of the cadence. This is not an
edit debounce or a lighting sample-count change. See the architectural contract.

Control: `7634ddc` = `8022e73` plus discovery timestamp reporting only. Its cave
run was made before that instrumentation-only commit, with the identical diff.
Candidate: `b56fd89` plus the four-phase shader/test changes in this section's
commit. Same M4/Metal, 16 GB, development profile, 2560x1440 offscreen +cache,
background terrain, corner look and 1024-evidence reconstruction. No other game,
build or image-analysis job ran during measured phases. Captures are valid,
inspected, and render logs are clean.

| Cave logged GPU median, ms | Full scan | Every-frame fingerprint | Four-phase |
| --- | ---: | ---: | ---: |
| Discovery only | 1.280 | .701 | .402 |
| All view presentation | 2.677 | 1.589 | .913 |

The targeted discovery saving is .878 ms / 69%. Presentation includes other
stages and more scheduling noise; it is not all attributable to discovery.
The intermediate smaller-workgroup/compact-classification run alone cost
1.065 ms discovery, so it was not the principal win. Lighting ray allowance
and receiver capacity are unchanged. Added metadata occupies 512 KiB per 1440p
camera (16 bytes per tile, rounded by the existing allocator), with no new GPU
readback. Discovery uses the already-recorded timestamps, not new timing trips.

Cave early-to-own-60s mean FLIP: at 15s .01059 -> .01056; at 30s .00615 -> .00613.
Final cross-run mean FLIP .00234, display luminance ratio .999985. This is visible
stability, not raw sampler accuracy. Offscreen cadence p95/p99: 25.53/26.59 ->
24.88/25.80 ms; no frame-pacing speedup is claimed from one pair. Peak process
allocation 985 -> 951 MB is affected by streaming, not a metadata memory saving.

Cave run IDs: control `cave-1792688495858814000-30220` in the control worktree;
every-frame fingerprint `cave-1792688839548175000-30598`; retained four-phase
`cave-1792689209820987000-31045`. Logs are `target/discovery-{control,reuse,stagger}-cave.log`
in their respective worktrees. Paired images and FLIP JSON are under
`target/discovery-comparison/`. Earlier `discovery-fast{,32}-cave.log` attempts
are retained but are not the final result.

Native tests compare exact receiver records with the full-scan control after
the bounded discovery interval, and prove unchanged thin-face coverage between
checks. They also exercise odd edges, raster-only geometry, periodic refresh,
forced camera refresh, single/MSAA pipelines and unchanged raw batching/pending
tails. Production Clippy passes. Relevant logs are
`target/discovery-stagger-{native,pipelines,batch,clippy}.log`.

The same control/candidate pair also passed the 35-second streaming workload
(turn 180 degrees, walk, stop), with three valid captures per run and clean render
logs. Final captures were visually compared. Moving cameras intentionally keep
full discovery: six logged moving-phase samples range from .867–1.539 ms for the
control and .755–1.249 ms for the candidate, insufficient for a precise
performance-gain claim. After stopping,
the candidate's two logged discovery costs are .426/.429 ms, confirming reuse
also engages after ordinary player movement. After 15s, excluding capture frames,
offscreen cadence p95/p99 is 24.86/25.57 -> 24.79/25.54 ms. Final visible owners
are 2459 -> 2454; slightly different camera positions preclude exact pixel parity.
Runs: `streaming-1792689433194498000-31550` (control worktree) and
`streaming-1792689513079108000-31692` (candidate). Logs:
`target/discovery-{control,stagger}-streaming.log` in the respective worktrees.
Release build passes with the existing 33 release-only unused/dead-code warnings
(`target/discovery-stagger-release.log`).

## Moving discovery checkpoint (2026-10-22)

Experimental moving-view reuse adds a face-local UNORM16x2 visibility witness
per slot. Projecting this physical point, rather than a face centre or a pixel
that drifts each frame, validates an existing selection against current depth,
normal, voxel identity and transport geometry. Valid selections survive full
tile scans; only vacated slots start again from grid centres. Moving footprints
scan in four phases or immediately when a witness is lost. Vacant centres also
check for newly exposed terrain during camera motion. Existing lighting advances
every frame. Metadata is now 1 MiB at 1440p; receiver/ray limits are unchanged.

This is not yet a solution to all coverage holes. The new validation-only
`motion` profile and `VOXEL_GI_VALIDATE_VIEW_COVERAGE=1` mask show approximately
0.15% of terrain pixels with no indirect support: red means missing, yellow
coplanar-only support, green exact face support. Four slots still cannot represent
every plane in a dense tile. This checkpoint motivates broader allocation tests.

Controls retain `3b10c1b` discovery: streaming `2dae4a9`, fixed-eye motion `d3d70a8`,
coverage mask `a6e8f97`. Candidate is `a6e8f97` plus this checkpoint's
shader/runtime/test diff. Same
M4/Metal, development profile, 2560x1440 offscreen +cache, background loading,
preferred look. Control streaming run is `streaming-1792691210062578000-35905`
(`target/moving-control-streaming.log` in the control worktree);
candidate is `streaming-1792691766051647000-37032`. During movement (15–30s),
logged discovery medians are 1.087 ms (five receipts) and .857 ms (six), a preliminary
21% saving. The final vacant-centre check was added after that candidate run.

Fixed-eye motion appearance runs: control `motion-1792691579067898000-36666`,
candidate `motion-1792691829076776000-37243`. Base-return mean FLIP over two seconds
is .000193 -> .000199; this does not establish a flicker improvement. Coverage-mask
runs: control `motion-1792691700830776000-36865`, candidate
`motion-1792691865539565000-37413`. Turned-view missing support is .155% -> .165%:
reprojection by itself does not cure the missing-face issue. Earlier motion runs
had different eye heights due to startup physics and are excluded from paired
quality evidence. Native tests cover witness reuse, thin orientations, bounded
discovery, raster-only geometry and disappearance from the prepass. Native
coverage/pipelines and production Clippy pass after the vacant-centre change
(`target/moving-{native,pipelines,clippy}.log`).

### Moving reuse qualification and raster lookup (2026-10-22)

The final moving-discovery implementation is `fa6d562`. `7b95fc8` adds interval
averages of all existing timestamp receipts; it does not change scheduling or
add GPU trips. This supersedes the sparse last-receipt estimate above. The
`3bf0790` motion-analysis tool separates visible raster coverage, exact-return
stability and an analytically aligned small-turn residual from raw accuracy.

The material lookup now uses actual world-space donor distances, not screen
slot distances: discovery can relocate a representative within its tile, and
that slot is not the sample location. An empty ordinary 4x4 lookup searches one
additional ring (6x6 total), with unchanged orientation, plane and 16-voxel
distance gates. This is presentation only; raw estimates and path budgets are
unchanged. It reduces missing support but does not guarantee every visible face
is represented. Four local receiver slots remain a capacity constraint.

Two broader approaches were rejected, not silently promoted:

- `codex/wider-view-discovery` / `a78627f` shares nine slots across 3x3 cells.
  It reduces missing turned-view pixels from 3470 to 1966, but moving discovery
  rises from 1.087 to 1.242 ms in the sparse-receipt comparison; a parallel-mask
  variant costs 1.424 ms. Native stepped-plane coverage passes. This branch is
  retained for future allocation work, not merged.
- Expanding empty raster lookups to 10x10 reduces missing pixels to 2677, but
  process footprint rises: streaming peak 910 -> 1068 MB and cave 973 -> 1426 MB.
  Returning to 6x6 restores streaming peak to 910 MB. The increase is much larger
  than discovery's extra 512 KiB of metadata; its exact Metal allocation cause
  is not established. The additional coverage does not justify retaining it.

Coverage control: `motion-1792691700830776000-36865` in the baseline worktree.
The retained 6x6 footprint gives 2969 unsupported turned-view pixels versus 3470
(14.4% fewer), from `motion-1792692929602136000-39048`. The later world-weight
correction changes values, not the geometry-validity mask. The rejected 10x10
mask is `motion-1792693053415036000-39466`; its normal motion run is
`motion-1792693344792483000-40100`. That run's aligned luminance residual is
essentially unchanged from the control (.007118 versus .007123), so neither
the fewer holes nor reprojection is claimed to solve overall distant flicker.
Zero-angle self-alignment returns no pixels above .02 luminance residual.

Final performance pair: baseline `a3ee4e4` in the baseline worktree retains
`3b10c1b` discovery plus identical harness/timing instrumentation; candidate is
`7b95fc8` plus this section's one-ring/world-weight material diff (the intervening
`3bf0790` changes only tools/docs). Same M4/Metal, 16 GB, development profile,
2560x1440 offscreen, explicit `config/gi-view-cached.json`, background loading,
corner look and 1024-evidence reconstruction. Frame timing is enabled in both.
No other game/build/analysis ran during measured phases. Captures are valid and
render logs clean.

| Streaming, fully moving intervals ending 18–30s | Control | Retained candidate |
| --- | ---: | ---: |
| Discovery mean, 480 GPU receipts each | 1.0757 ms | .9070 ms |
| Offscreen cadence p50 / p95 / p99 after 15s, capture frames excluded | 23.071 / 25.305 / 26.433 ms | 22.974 / 25.155 / 26.253 ms |
| Peak process footprint | 910 MB | 913 MB |

Moving discovery saves .169 ms / 15.7%. A preliminary repeat of the final material
without CPU frame-timeline instrumentation measured .9404 ms / 479 GPU receipts
and 910 MB; it is not used for cadence comparisons. The earlier 10x10 material
measured .9131 ms / 480 receipts, but is rejected on memory above. These runs do
not establish windowed interactive frame pacing. Receiver count/ray allowance
and raw estimator are unchanged; discovery metadata adds 512 KiB per 1440p view,
not a new readback or CPU wait.

Final streaming run IDs: control `streaming-1792693176004804000-39702`, candidate
`streaming-1792694119128168000-41701`. Logs:
`target/moving-control-streaming-mean.log` in the baseline worktree and
`target/moving-final-ring1-streaming.log` here. The preliminary no-frame-timeline
repeat is `streaming-1792693927407330000-41034` / `target/moving-memory-ring1.log`.

Cave control `cave-1792693488227939000-40351` and final candidate
`cave-1792694003150381000-41259` have clean, valid and visually inspected captures.
Candidate frame-timeline instrumentation was omitted, so this is a quality/memory
check, not a cave cadence comparison. Early-to-own-60s mean FLIP is .010288 ->
.009624 at 15s and .006284 -> .005884 at 30s. Final cross-run mean FLIP is .003055,
display luminance ratio .999969. Peak footprint is 973 -> 977 MB. These are
presentation/stability comparisons, not raw transport accuracy or convergence.

Final normal motion run `motion-1792694224792115000-41941` also has clean, valid
and inspected captures. Exact-return mean FLIP is .000201 versus .000193 control;
aligned mean absolute display-luminance residual is .007118 versus .007123.
This confirms no meaningful overall motion-stability gain is established by
the small-turn diagnostic yet; the narrower supported claim is fewer uncovered
pixels and cheaper moving discovery. Detailed output is
`target/moving-final-ring1-quality.json`.

Native thin-face/moving-witness/disappearance and single/MSAA pipeline checks
passed at the unchanged discovery checkpoint. Raw-prefix/pending-tail and full
Bevy terrain-material composition checks passed; composition was rerun for the
retained one-ring material (`target/moving-ring1-material.log`). Production
Clippy passed on the unchanged Rust (`target/moving-clippy-final.log`).
The final `cargo build --release --locked --offline` passes with the existing
release-only unused/dead-code warnings (`target/moving-final-release.log`).

## Shared receiver pool experiment (2026-10-22)

Branch `codex/shared-view-receiver-pool` starts from `9950a99`; the clean control
is `codex/shared-pool-baseline` at that commit in the existing baseline worktree.
This checkpoint proves native allocation/identity correctness, not gameplay
performance or a promotion decision.

Storage slots no longer encode screen coordinates. A preparation dispatch
validates and reprojects old visibility witnesses, keeps surviving slots across
tile boundaries, publishes per-tile linked spatial lists and collects a global
free list. After that dispatch completes, discovery tiles borrow free slots for
new representatives, without cross-workgroup spinning or publishing partial
identities. Tile scans remain phased and fingerprinted. Occluded, invalid and
duplicate slots are reclaimed; transport history remains face-keyed. Pool
exhaustion is bounded, never an out-of-bounds allocation. The ordinary receiver
and ray ceilings are unchanged; the spatial/free-list metadata is additional.

Reconstruction and raster upsampling consume the spatial index rather than
physical slot coordinates. Exact faces remain hash-addressed, including corner
sunlight. The material checks the nearest four spatial bins for coplanar donors
under the existing 16-voxel gate. Raw moments, QMC sequences, unfinished paths,
retention and cache exchange are unchanged.

Native test `native_view_discovery_keeps_thin_surfaces_without_growing_storage`
now puts six depth planes plus one-pixel X/Y grazing faces in one crowded tile,
with empty tiles elsewhere. It verifies borrowing beyond four local slots,
stable stationary identities, cross-tile reprojection, occlusion/reappearance,
odd grid edges and bounded exhaustion with nine total slots. Real transport
geometry backs the synthetic prepass. Native single/MSAA pipelines, shader
layout validation and full Bevy material composition pass. Logs are
`target/pool-{native,pipelines,shader,material}.log`; the final cleanup only
removes unused helpers and shares the existing exact-face lookup.

### Pool qualification and benchmark correction

`75e9988` splits preparation and scanning into separately timed passes and adds
free/claimed/exhausted/retained counters to the existing receipt (224 bytes, not
192; no new callback or lighting readback). `5b99afe` builds the retained-face
index before discovery, avoids recreating globally represented faces, and treats
cross-bin motion separately from actual loss of visibility. `02ae4a8` tests raw
mean/weight and QMC-sequence preservation across motion, plus clean history when
a physical slot is recycled for an unrelated face.

Camera-position review caught a harness defect: translating only the character
Transform could be overwritten by controller synchronization. Some old streaming
runs moved 32 or 64 voxels while others remained near spawn. **Treat earlier
streaming cost comparisons as provisional, not matched-path qualification.**
The fixed-eye motion/coverage and fixed cave captures are unaffected. `711f247`
replays a detached absolute path; `710e066` only removes the now-unused mutable
look-component borrow. The same camera-path fix is `933909f` on the control.

Matched-path pair, M4/Metal 16 GB, development profile, 2560x1440 offscreen,
`config/gi-view-cached.json`, background terrain, preferred look and frame timing:

| Fully moving intervals ending 18–30s | `933909f` control | `711f247` shared pool |
| --- | ---: | ---: |
| Mean total discovery, 480 receipts each | .7925 ms | .9605 ms |
| Offscreen cadence p50 / p95 / p99 after 15s, capture frames excluded | 23.077 / 25.367 / 27.134 ms | 23.145 / 25.419 / 26.826 ms |
| Peak process footprint | 908 MB | 911 MB |

The pooled design still costs .168 ms more for discovery; it is not a discovery
speedup. Both cameras start at `(0,19.62,0)` and end at `(0,19.62,64.8)`, with
intermediate capture positions agreeing within .011 voxels. Runs are
`streaming-1792696118125527000-44446` in the control worktree and
`streaming-1792696193763104000-44583` here; logs are
`target/pool-{control,candidate}-fixed-streaming.log`. Candidate's dirty diff
at capture is the later `02ae4a8` native tests only, not gameplay code.
These timings do not establish windowed FPS. No other game/build/analysis ran
during the measured phases.

`codex/pool-batched-reclaim` / `18ecfe1` tries workgroup-batched free-list
compaction. Native history/coverage checks pass, but the same path measures
.9563 ms versus .9605 ms: no meaningful gain. The simpler reclamation stays in
the main pool experiment. Its run is `streaming-1792696345525913000-44847`, with
the shader diff against `02ae4a8`; log `target/pool-compact-streaming.log`.

The first pool's fixed-eye mask (`48f204d`,
`motion-1792695603142721000-43371`) reduced turned-view unsupported pixels from
2965 to 560 versus control `motion-1792694916580032000-42797`. This early 81%
reduction qualifies coverage only, not the later retained-index revision or
ordinary material cost. Final appearance and edit checks are recorded below.

### Final pool coverage and appearance

Final captures use `710e066` plus a shader-only cleanup (indentation and removal
of an already-true nested owner check), a Rust comment and the documentation
diff. The tested runtime cleanup is committed as `579a8c5`. Hardware, viewport,
profile and settings match the pair above. Control
appearance runs use `933909f`; its rendering remains `9950a99`. All runs completed
with valid, varied 2560x1440 offscreen targets and no render errors. The cave,
normal motion and emitter-response comparisons were visually inspected.

The final coverage-mask run `motion-1792696800868052000-45670` has **2 / 0 / 2**
unsupported pixels in `base-1 / turned-1 / base-2`, versus **2982 / 2965 / 2982**
in control `motion-1792694916580032000-42797`. The diagnostic counts strict red
pixels (`R > 240, G < 15, B < 15`) before tone mapping/fog. Green means corner
support and yellow means valid geometric upsampling; neither is missing light.
This is a coverage result in the tested views, not a claim of universal coverage.

Normal-motion runs are `motion-1792696473251582000-45154` (control) and
`motion-1792696834219473000-45869` (pool). Exact-return mean FLIP is .000213 ->
.000240. Aligned small-turn mean absolute display-luminance residual is
.007118 -> .007111. These are essentially unchanged; removing unsupported pixels
does not establish an overall motion-flicker improvement in this diagnostic.

| Appearance measurement, control -> pool | Result |
| --- | --- |
| Cave 15s versus own 60s, mean FLIP | .009672 -> .010683 |
| Cave 30s versus own 60s, mean FLIP | .005806 -> .006429 |
| Cave 60s cross-run mean FLIP / display-luminance ratio | .005079 / .999854 |
| Emitter add 200ms versus own settled, mean FLIP | .287266 -> .319604 |
| Emitter add 1000ms versus own settled, mean FLIP | .014213 -> .016574 |
| Emitter removal 600ms versus own recovered, mean FLIP | .019116 -> .017093 |
| Emitter removal 1000ms versus own recovered, mean FLIP | .007034 -> .008889 |

Early addition is slightly darker in this pair; removal recovers on the same
observed timescale. Cave appearance remains close, with a small increase in
early-to-late change rather than a quality gain. These self-change measurements
are not reference accuracy or evidence that the sampler improved. Near-steady
cave discovery is about .43 ms versus .46 ms control. Late cave pool counters
show 5982 retained faces, 59490 free slots and no allocation exhaustion: the
unused-capacity premise holds in the actual scene as well as the native fixture.

Cave runs are `cave-1792696506631858000-45280` / `cave-1792696867547930000-46042`;
response runs are `response-1792696570147438000-45143` /
`response-1792696931083513000-45653`. JSON and paired images are in
`target/pool-final-quality/`, motion analysis in
`target/pool-final-motion-quality.json`. Run logs are
`target/pool-{control,final}-{mask,motion,cave,response}.log`.

Qualification: keep this as a **coverage-winning experiment**, not a performance
win or an automatic merge. Moving discovery costs an additional .168 ms; added
GPU metadata is about 3.375 MiB at this viewport (4 MiB pool buffers versus 1 MiB
old tile metadata, plus 384 KiB published spatial lists). Receiver/path capacity
and transport budgets did not grow. Total capacity exhaustion is safety-tested,
but fairness between competing tiles under sustained full-pool pressure is not
qualified; ordinary scene tests had ample spare capacity. General motion flicker
remains an open target. The main branch and default hybrid mode are unchanged.

Focused native coverage/raw-history/QMC checks, raw-prefix/pending-tail parity,
single/MSAA layouts and full Bevy material composition pass. Final production
Clippy passes with warnings denied. Logs: `target/pool-history-native.log`,
`target/pool-batch.log`, `target/pool-final-{layout,pipelines,clippy}.log` and
`target/pool-material.log`. `cargo build --release --locked --offline` passes
with 33 existing unused/dead-code warnings (`target/pool-final-release.log`).

### Follow-up discovery experiments and motion diagnosis (2026-10-22)

Keep the shared-pool baseline at `3e8eb1c`. Focused experiments did not close the
moving-cost gap, so neither changed scheduling nor wider workgroups is promoted:

| Candidate | Mean moving discovery | Decision |
| --- | ---: | --- |
| `3e8eb1c` control | .9619 ms | Tested pool baseline |
| Lane frontier, `c2ca08d` | .9695 ms | Reject; no gain |
| Coarse-cell repair masks, `d5f34cc` | .9470 ms | Unpromoted; too little gain for changed scheduling |
| Fused retained indexing, `52b7390` | .9486 ms | Keep redundant-dispatch removal; no measured speedup claim |
| Fused indexing with 64 discovery lanes | 1.0333 ms | Reject; retain 32 lanes |

All use M4/Metal 16 GB, dev, 2560x1440 offscreen, explicit
`config/gi-view-cached.json`, deterministic `streaming` plus background terrain,
480 receipts each in fully moving intervals ending 18–30s. The control omitted
frame-timeline instrumentation; compare GPU discovery only. Control run is
`streaming-1792698200872467000-47093` in the baseline worktree, candidates are
`streaming-1792698303369975000-47334`, `streaming-1792698533984130000-47757`,
`streaming-1792698777558887000-48136`, `streaming-1792698901359967000-48400`
respectively. Each capture precedes its commit and has only the named candidate
diff against `3e8eb1c`. The 64-lane follow-up adds the workgroup-size/loop-stride
change to the fused-index diff; it was reverted. Logs are
`target/frontier-{control,candidate}-streaming.log`, `target/cell-mask-streaming.log`,
`target/fused-index-streaming.log` and `target/pool64-streaming.log`.

The retained change folds indexing into preparation. Only canonical previous
owners insert keys; no two such owners share an identity. CAS collision handling
therefore needs no reads from current receivers while preparation writes them.
The dispatch boundary publishes all receiver contents before discovery reads
the index. One full-capacity dispatch is removed without changing discovery,
receiver/ray capacity or presentation. Native grazing/crowded coverage,
history/QMC preservation, single/MSAA pipelines and shader layout pass
(`target/fused-index-{native,pipelines,shader}.log`).

`57b4698` adds component-specific motion captures. Sunlight return/repeated-view
mean FLIP is .0000191 / .0000051; indirect is .001482 / .001357. Those runs are
`motion-1792699093629026000-48738` and `motion-1792699126994063000-48730`, with
clean logs and valid, visually inspected offscreen captures. This isolates
remaining same-view change primarily to indirect presentation in this workload;
it does not yet distinguish refinement from camera-induced donor changes.
Analysis: `target/motion-components-quality.json`.

Stationary indirect control `motion-1792699242689033000-49141` uses `57b4698`
plus the static-camera/report/analysis diff. Return/repeated mean FLIP is
.001288 / .001254 versus .001482 / .001357 with turns. Much of the same-view
change therefore continues without camera motion; do not attribute it all to
discovery or missing faces. The aligned one-second stationary residual is
.0000225 with zero pixels above .02, versus .000347 and 1837 pixels with the
turn. The latter still includes silhouette/raster interpolation error and does
not by itself prove a lighting bug. Sunlight's aligned residual also contains
those edges despite its almost unchanged exact-view captures. These diagnostics
do not establish a fix for all observed gameplay flicker or raw-path accuracy.

Final coverage `motion-1792699276145770000-49335` has **1 / 3 / 1** unsupported
pixels in base/turned/returned views, retaining the pooled coverage result.
Normal motion `motion-1792699309512997000-49140` has return mean FLIP .000224
(previous pool .000240) and aligned mean residual .007107 (previous .007111):
no meaningful stability difference. Both use the same runtime and static-harness
diff above, with ordinary turns enabled. All capture targets are valid/varied
2560x1440, logs clean; normal, sunlight and indirect images were inspected.
Logs `target/motion-component-{sun,indirect,static}.log`,
`target/fused-index-{mask,motion}.log`; analysis
`target/fused-final-motion-quality.json`. Normal material and transport settings
remain unchanged. Production Clippy passes (`target/fused-final-clippy.log`).
The static harness is committed as `1ab8810`. The final release build passes
with the same 33 existing unused/dead-code warnings (`target/fused-final-release.log`).
This follow-up remains on `codex/view-pool-index-fusion`; the tested shared-pool
baseline and original two-way-cache branch are unchanged.

The next diagnostic target is indirect refinement/reconstruction, with component
captures and a stationary/background-streaming control. Do not add stronger
smoothing based only on these small aggregate changes; first reproduce the
visibly objectionable changes and separate raw estimates from reconstruction.

### Persistent camera-angle shadow report (2026-10-22)

The user narrowed the remaining symptom to mostly up-facing, partially shadowed
faces. They can stay fully dark indefinitely at one angle and change after a
tiny pan. This reopens view-dependent lookup/coverage, including sunlight:
returning to an identical view cannot detect a deterministic wrong lookup.
Do not interpret the earlier same-view results as clearing sunlight during turns.

`e3f2931` adds capture-only `sun-reference`, tracing the displayed face's own
canonical corners instead of using the view cache's copied corner values. Normal
shading, discovery and transport are unchanged. Code inspection found a plausible
mechanism: a missing exact face may borrow sunlight from coplanar donors up to
16 voxels away, clamping target coordinates to each donor's corners. Shared plane
identity does not imply shared shadow visibility. This is not yet established as
the cause of the user's persistent dark patches.

On that exact clean commit, M4/Metal 16 GB, dev, 2560x1440 offscreen, motion plus
background terrain and `config/gi-view-cached.json`, compare cached sunlight
`motion-1792709609340745000-53564` with exact corner reference
`motion-1792709648944147000-53776`. Both completed with six valid, varied captures
and clean render logs. Base/turned/returned cross-run mean FLIP was
.000711/.000679/.000701. At the returned view, only 22 pixels were darker than
reference by >.1 display luminance; 235 were brighter by >.1. Visual inspection
shows small edge differences, not a reproduction of large persistent black top
faces. This mostly open spawn view is insufficient to qualify the reported bug.
The reference traces per fragment and is deliberately expensive; its cadence
and indirect history are not gameplay performance/transport evidence.

Logs: `target/shadow-lookup-{control,reference}.log`; paired images and amplified
signed error maps: `target/shadow-lookup-{base-1,turned-1,returned}{,-error}.png`.
Diagnostic selection and full Bevy material composition tests passed (one each):
`target/shadow-{diagnostic,material}-test.log`. No shading fix or smoothing change
is promoted. Reproduce the reported location/angle before choosing a remedy.

### Subpixel face identity and the coverage metric (2026-10-22)

The user's two snow-field screenshots clarify the target: roughly 3–8 tiny
faces per view, not large dark patches. They persist at a fixed angle and change
with pixel-sized pans. Whole-frame FLIP and a count of fully red coverage pixels
are insufficient to clear this symptom.

`93e3492` adds a native 4x-MSAA fixture: six face orientations, eight subpixel
translations, exact voxel-key lookup. Pixel-centre interpolation produces 36
wrong-voxel pixels; centroid produces zero away from exact raster-edge ties.
At exact edge ties, centroid still produces 12 wrong-voxel pixels due to boundary
rounding. Keep that limitation explicit. Log: `target/centroid-native.log`.

The new `face-validity` diagnostic (`ef625d3`) independently encodes missing
indirect support and invalid exposed-geometry lookup. The old palette had another
blind spot: MSAA mixing red failure with green support produces yellow, the same
color as valid coplanar borrowing. Counts now include partial contributions and
resolved channel sums; they are not numbers of visibly black faces.

M4/Metal 16 GB, dev, 2560x1440 offscreen, fixed-eye `motion` with background
terrain and `config/gi-view-cached.json`:

| Diagnostic, base / turned view | Pixel centre | Centroid |
| --- | ---: | ---: |
| Pixels containing an invalid face lookup | 49,625 / 49,691 | 324 / 79 |
| Pixels containing missing indirect support | 7,390 / 7,444 | 850 / 875 |

Validity runs in the two-way-cache-baseline worktree are
`motion-1792710950787462000-55740` (control) and
`motion-1792711016223130000-55895` (candidate). Both use `b549b3e` plus the
capture-only shader/parser diff later committed in `ef625d3`; the candidate adds
only the centroid annotation/comment. All six captures are valid/varied, logs
clean, and the images were inspected. The user's existing game remained running
for these correctness captures; their cadence is not performance evidence.
Analysis: `target/centroid-validity.json`, generated by `tools/view_validity.py`.

After the user approved closing their game, ordinary appearance was paired at
the same settings: main `93e3492`, clean, run
`motion-1792711123810758000-56123`, versus baseline worktree `b549b3e` plus the
diagnostic and centroid diff, run `motion-1792711226294745000-56475`. Diagnostics
are off in both. Offscreen cadence p50/p95/p99 was 23.19/25.17/26.67 ms versus
22.92/25.29/26.23 ms: no meaningful performance difference claimed from this pair.
Return mean FLIP .000239 versus .000245, aligned mean residual .007108 versus
.007113: the broad metric remains dominated by normal resampling. Ordinary
images and known-hole crops were inspected; no gross appearance regression.
Logs: `target/centroid-control-motion.log` in main and
`target/centroid-candidate-motion.log` in the baseline worktree. Analysis/crops:
`target/centroid-motion-quality.json`, `target/centroid-appearance-*.png`.

Keep centroid as a concrete raster correctness fix, not proof that every reported
snow-field speck is resolved. It adds no receiver storage, transport samples,
history or sample-rate shading. Remaining valid-face coverage holes, exact grid
boundary ties, and borrowed sunlight accuracy are not solved by this change.
The material composition test now checks the actual centroid interpolation
contract (`target/centroid-material-test.log`); selection/composition also passed
for the diagnostic (`target/centroid-validity-{test,material}.log`).

The retained fix is `780e824` on `codex/centroid-face-lookup`; previous baselines
are unchanged and the temporary baseline checkout is restored to
`codex/shared-pool-baseline` at `933909f`. The final release build passes with
the same 33 existing warnings (`target/centroid-release.log`). No live GI process
is left running. The validity analyzer rejects non-view ownership configurations
instead of treating ordinary hybrid shading as a diagnostic mask.

### Edit bootstrap and asynchronous geometry handoff (2026-10-22)

Branch `codex/edit-face-bootstrap`, baseline `f79b80c`. `473b90e` adds the user's
one-wide/two-high hillside tunnel: twenty depths, forty individual removals,
upper block followed 250 ms later by the lower block. All forty edits hit solid
voxels. Captures sample depths 1/2/4/8/12/20, with background terrain loading.

`504e17f` initializes new view faces from supported **raw** local face data at
one effective history weight, with no QMC index advance. Matching normals take
precedence. A native Metal test checks zero-ray initialization, perpendicular
and covered donors, rejection of remote/unmeasured donors, repeated handoff
without duplicate weight, and exact 1/(N+1) dilution by sixteen real paths.
The existing two-way ownership test also passes. No display-filtered donor data
is admitted, and no independent-path accuracy threshold is changed.

Raw seeding alone was **not** a reliable gameplay fix. In two ordinary tunnel
pairs, the baseline had 831,623 / 824,816 black-then-lit pixels in the sampled
400 ms views, while the seed-only candidate had 59,048 / 885,676. These counts
use pixels at <=2/255 in all early RGB channels that are later >.02 luminance,
not a physical radiance threshold. Runs in `target/gi-validation/`:

- control (baseline worktree, clean `473b90e`):
  `tunnel-1792717848472801000-58753`, `tunnel-1792718292879742000-59733`;
- raw seed (main, `473b90e` + the feature/test diff committed as `504e17f`):
  `tunnel-1792718013635734000-59061`, `tunnel-1792718189553402000-59426`.

The diagnostic extension in `215f905` distinguishes missing support (red) from
invalid exposed transport geometry (green). Its tiny blue execution witness is
excluded by the analyzer; it permits a legitimate zero-failure indoor mask
without weakening normal black-frame rejection. The initial witness-free run
`tunnel-1792718571227497000-60052` correctly failed the blank-capture check and
is not gameplay evidence. The completed diagnostic run
`tunnel-1792718736877736000-60189` showed 832,537 pixels with **both** flags on
the new floor at 400 ms. The new mesh existed before its transport geometry;
discovery rejected the face, so a raw seed alone could not help it.

The retained handoff change (following `215f905`) keeps an explicit pending
raster identity in the existing receiver record. It can hold a raw seed or
existing exact-face history, but cannot trace, import a resident authority or
write back until exposure agrees with the current transport scene. Each retained
witness is reclassified every frame. The native coverage fixture now delays the
transport model for three frames, checks zero queued rays, then checks the same
raw means and QMC sequence through admission, motion and occlusion. The cache
test also verifies pending faces cannot modify resident state. All pass on Metal.

Two normal tunnel runs with the pending-identity diff on `215f905`:
`tunnel-1792719210082405000-60725` and
`tunnel-1792719342972030000-61047`. Both had **zero black-then-lit pixels across
the six sampled 400 ms views**, without increasing ray budgets or allocations.
All twenty captures in each run were valid/varied at 2560x1440, logs clean and
images inspected. This is a dropout result, not proof that every intervening
frame or every edit is correct. Early brightness still evolves: depth-one
400-to-1700 ms mean FLIP was .198/.179 versus .176/.234 in the controls; depth-two
was .091/.099 versus .048/.093. The later frame is not an oracle. The first
candidate's bright-then-dimmer mask covered 11,182 pixels at depth one (2,829 in
the repeat); do not describe the feature as eliminating all brightness transients.

M4/Metal, 16 GB, development profile, offscreen 2560x1440,
`config/gi-view-cached.json`, background streaming, preferred corner look:

| Paired workload | Control p50/p95/p99 ms | Seed + pending p50/p95/p99 ms |
| --- | --- | --- |
| Tunnel pair 1 | 23.16 / 25.07 / 25.82 | 23.15 / 24.94 / 25.65 |
| Tunnel pair 2 | 23.07 / 25.10 / 25.96 | 23.18 / 24.96 / 25.70 |
| Small camera turns | 23.20 / 25.22 / 26.34 | 23.26 / 25.10 / 26.25 |

No meaningful cadence change is claimed. Motion runs are baseline worktree
`motion-1792719500285476000-61544` (clean `473b90e`) and main
`motion-1792719435570319000-61373` (`215f905` + pending-identity diff).
No receiver-pool exhaustion occurred. Return mean FLIP .000225/.000288 and
aligned mean residual .007106/.007118 do not establish a small visual difference;
the ordinary images were also inspected. Persistent subpixel identity/coverage
issues from the preceding investigation are not declared solved.

Artifacts: `target/bootstrap-{tunnel,pending}-quality*/tunnel.{json,png}`,
`target/bootstrap-tunnel-validity.json`, `target/bootstrap-motion-quality.json`;
logs `target/bootstrap-*.log` in the owning worktree. Native logs:
`target/bootstrap-pending-{pipeline,coverage,native}.log` and
`target/bootstrap-cache-ownership.log`. Baseline runs remain reproducible without
the estimator change; no BVW changes, debounce, additional lighting readbacks or
main-thread waits were introduced.

Final retained handoff commit: `391ac9c`. The targeted `edits` pair also covers
restoration, placement, removal and a surface opening: baseline worktree clean
`473b90e`, `edits-1792719684926882000-63254`, versus main `215f905` plus the
pending diff (and a profile-error-message-only correction),
`edits-1792719606854339000-61673`. All eighteen captures per run are valid/varied,
with clean render logs. At 150 ms the control's new cavity and placed block are
black; the candidate has initialized lighting on both. These images were
inspected directly, not inferred from the CPU world-field mirror. Offscreen
p50/p95/p99: 23.08/25.23/26.33 ms control, 23.13/25.14/25.96 ms candidate.
The final release build at `391ac9c` passes with the same 33 existing warnings
(`target/bootstrap-release.log`). The comparison worktree is restored clean to
`codex/shared-pool-baseline` at `933909f`; no GI test process is left running.

### Discovery additions after the shared-corner sun fix (2026-10-23)

With unowned faces taking sunlight from shared corners (`2d0e143`), each
addition made while chasing the distant flicker was retested on `1c53aea`
(M4, headless 2560x1440, default +cache 64). The sun component matched the
exact corner reference (0 pixels off by >.1) in every variant.

| Variant | Streaming presentation / discovery ms | Static combined vs HEAD | Walk black-flash px (indirect) | Decision |
| --- | ---: | --- | --- | --- |
| HEAD | 2.57 / 1.21 (cave .94 / .43) | — | 1414, 1389, 1714, 3344 | |
| Footprint scan removed (coarse, `3fdc8ff` discovery only) | 1.48 / .58 (cave .90 / .37) | 74 px darker: specks on distant ledge tops; indirect-only 13.6k px darker | 7127, 8977 (all 28 frames) | Keep the scan: it is the indirect half of the flicker fix |
| Four-phase stagger off | 2.79 / 1.68 (cave 1.30 / .73) | identical | 1954, 1385 | Keep: turn-burst black frames match HEAD across 3 runs each |
| Lone-sample rule off | 2.38 / 1.28 (cave .94 / .43) | identical | 1298, 2027, 1544, 1061 | Removed: no longer reduces flashes |

The footprint scan therefore still pays for itself, and memoization, the
stagger and witnesses remain its cost controls.

A cheaper scan was tried on `daac913`: one pass reducing each pixel to a plane
key (face, plane, 16-voxel block) in workgroup memory, with the full decode and
coverage check once per key instead of once per pixel per allocation round.
With a 256-entry table, streaming discovery stayed 1.28 ms and cave rose
.43 -> .88 ms (workgroup memory is reserved by every tile, including those that
exit early). With 64 entries: streaming 1.28 -> 1.01 ms, cave .43 -> .48 ms, but
577 more darker pixels in the indirect-only static view and walk flashes
8181/2314. Most tiles finish in one round, so the cost is the per-pixel decode
itself, not the rounds. Not kept. A real saving needs either fewer checked
pixels or a correct indirect fallback for missed tiny faces (the
any-orientation fallback of `3fdc8ff` tinted them sky-blue).
