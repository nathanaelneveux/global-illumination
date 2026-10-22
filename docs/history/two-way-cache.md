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
