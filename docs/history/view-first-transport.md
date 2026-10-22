# View-first receiver experiment

Follow-up: [two-way cache exchange](two-way-cache.md) tests persistent exact-face
history for the cached candidate. The evidence below predates that exchange;
its cold-reveal limitation describes this original one-way implementation.

Baseline: `9e52b7a` on `gi-far-view-transport`, preserved in the managed
`view-first-baseline` worktree. This branch keeps the hybrid default unchanged.

## Question and controls

Visible receiver ownership and persistent bounce-cache residency are separate
decisions. Both view candidates use the same face-identified view frontend,
shared path estimator, geometry, materials, corner sunlight and display target.
Only cache residency and the division of the existing transport allowance differ
between the two view-first candidates:

| Configuration | Visible owner | World cache radius | World / view budget |
| --- | --- | ---: | ---: |
| Unchanged hybrid | Near world + far view | 64 | 75% / 25% |
| `config/gi-view-cached.json` | View everywhere | 32 | 25% / 75% |
| `config/gi-view-cacheless.json` | View everywhere | None | 0% / 100% |

Radii admit whole 32-voxel pages by nearest distance, with the existing
32-voxel retirement hysteresis. They are not exact spherical face cutoffs.
The total transport share remains 20%. Presentation is measured separately.

View-first does not build or upload the world display directory and does not
build world presentation donor lists. Cached mode retains world probes, raw GPU
statistics, scheduling, sky measurements and edit-flux measurement for secondary
ray termination. Cacheless admits no world probes and submits no world sampling.
The zero-face worker still publishes geometry, sources and edit packets; its
resident event processor and binding-sized empty cache buffers are infrastructure,
not a populated bounce cache. Without outgoing-light observations an edit's flux
is explicitly unknown, so retention uses the existing distance curve rather
than falsely treating no observations as a measured zero change.

This is not spatial path reuse, a new sampler or a new calibrated accuracy
threshold. View histories remain visibility-dependent. The bounded coplanar view
filter feeds the same connected virtual-corner interpolation used by world
receivers. View-first publishes its existing GPU face index with the display
values, so close-up faces find exact neighbours instead of blurring nearby
screen samples. Missing reduced-grid faces still use geometric upsampling.
The fixed index adds 512 KiB to output allocations (including placeholders),
no CPU directory or lighting readback. CPU world-display diagnostics do not
measure view-first presentation; use actual captures and view GPU counters.

View-first can spend its allowance on repeated sequential visits to a visible
face (up to 32 three-hit waves per owner per dispatch, at most 65,536 visits
overall). The feedback target is its frame's transport share, capped at 4 ms;
actual asynchronous GPU receipts repay the same allowance. Hybrid retains its
one-visit policy. A native furnace test compares 128 one-visit dispatches with
16 eight-visit and four 32-visit dispatches: receiver sequence, last path, pending
tail and raw moments are bit-identical. There is no display filtering in this
check. This removes an incidental one-path-per-frame limit, not path termination.

## Decision

Keep **view-first with a small persistent bounce cache** as the preferred
experimental candidate. Do not replace the hybrid default yet: it is better
on first reveal, and the attempted windowed comparison was invalid. Cacheless
is functional and useful as a control, but its modest additional memory saving
does not offset slower emitter response and greater reveal error in these runs.
The runtime controls stay independent so cache radius/share can be qualified
without another frontend rewrite. No sampler accuracy threshold changed.

### Corrected comparison

Baseline `9e52b7a` vs clean candidate `5fe9a3b`. Same hardware, profile, 2560×1440
offscreen targets, appearance and background loading as below. Candidate
`6e19c1c` changes only the frame-modulo spelling to satisfy the current compiler's
production lint; no rendering/sampling behavior changes. All six corrected
candidate runs and the three baseline runs have valid varied captures and clean
render logs. No other game, build or analysis job ran during measured phases.

| Measurement | Hybrid | View + small cache | View, no cache |
| --- | ---: | ---: | ---: |
| Cave peak physical footprint, MB | 1189 | 967 | 900 |
| House peak physical footprint, MB | 1231 | 978 | 933 |
| Response peak physical footprint, MB | 1075 | 940 | 905 |
| Cave 15s → own 60s, mean FLIP | .01742 | .01077 | .01887 |
| Cave 30s → own 60s, mean FLIP | .01114 | .00643 | .01334 |
| House ceiling: 400ms turn → own later view, mean FLIP | .00897 | .03494 | .05407 |
| Emitter addition 1s → own final, mean FLIP | .08672 | .01712 | .02848 |
| Emitter removal 600ms → own final, mean FLIP | .40795 | .01164 | .13250 |
| Emitter removal 1s → own final, mean FLIP | .11061 | .00574 | .06813 |

These use the official FLIP evaluator at 67 pixels/degree on actual material
captures. **Self differences measure remaining visible change, not accuracy
against a converged reference or a continuously observed settle time.** House
comparison uses the upper 40% of the image to exclude the streamed outdoor
view through the doorway. Response uses fixed-camera whole images. The final
response targets are four seconds after the edit, not certified convergence.
Against the hybrid's live 60s cave image, mean FLIP is .01010 cached / .01914
cacheless. The final room images also differ (.06413/.07116 for emitter-on,
.03482/.02929 for emitter-off); do not claim identical physical solutions or
use these comparisons to recalibrate raw sampling accuracy.

Visual inspection confirms smooth torch halos and floor gradients after shared
corner reconstruction. Cold house turns remain the clearest weakness: visible
history is discarded when a face leaves the view. The small bounce cache helps
subsequent paths, but it is not yet used as an exact-face presentation prior.
That is the next targeted experiment if this direction is pursued; no spatial
path reconnection or invisible history count is being claimed here.

Post-15s offscreen cadence p95 ranges from 24.9 to 26.2 ms, including the
harness's sleep; it establishes no interactive FPS result. The windowed hybrid
attempt (`target/view-baseline-gameplay-window.log` in the baseline worktree)
failed its first capture: unfocused and uniformly black. The harness exited
with failure. That run is excluded; no windowed performance win is claimed.
The screensaver/security settings were not changed.

Evidence: candidate `target/view-final-{cached,cacheless}-{cave,house,response}.log`;
baseline `target/view-first-baseline-{cave,house,response}.log`. Each records the
unique `target/gi-validation/` output path. Combined sheets and numeric summaries
are under `target/view-final-comparison/`; scratch reproduction scripts are
`target/analyze_view_first.py` (argument `final`) and `target/view_capture_flip.py`.
Large captures and logs remain ignored. The baseline worktree is preserved.

Verification: 244 regular tests pass, 82 native/benchmark tests remain explicitly
ignored; selected native batching, view pipelines, event lifecycle and edit-flux
parity tests passed separately. The native headless material/cache-generation
test also passes with the enlarged display placeholder. Production-target strict Clippy and development
build pass. All-target strict Clippy still reports nine inherited test-only
style lints (default-field reassignment, test-module placement, single-element
loop); it is not reported as lint-clean.

### Scope of the experimental controls

- The world-display CPU directory and donor topology are absent in both view
  modes. The cache, when enabled, feeds raw bounce transport, not screen pixels.
- The ordinary HUD's coverage/history describes world receivers, not view
  convergence. Use view GPU counters plus captures when qualifying these modes.
- View histories retain RGB presentation only. Cacheless does not provide a
  persistent world-space IR/radiation field for simulation queries. A separately
  scheduled simulation receiver service is not implemented by this experiment.
- Sparse/subpixel coverage still uses geometric upsampling; visible-corner
  neighbours are not a complete world topology. Camera-reveal behavior remains
  part of the qualification, even when stationary images are good.

### First ownership comparison, before corner presentation

Exact baseline `9e52b7a` vs `1880379`, clean tracked trees, development opt1 /
dependencies opt3, Apple M4/Metal, 16 GB. Offscreen 2560×1440, preferred appearance,
background terrain, 20% total transport allowance. VS Code and its Clippy build
were closed; no competing build or game during captures. All nine runs had
valid varied captures and no render errors. Logs `target/view-first-*.log` in
each checkout identify their exact output directories.

| Mode | Cave peak footprint | House peak footprint | Response peak footprint |
| --- | ---: | ---: | ---: |
| Hybrid | 1189 MB | 1231 MB | 1075 MB |
| View + 32-voxel cache | 965 MB | 980 MB | 938 MB |
| View without cache | 889 MB | 932 MB | 903 MB |

Broad cave gradients agree visually. Both view candidates expose a presentation
mismatch at close range: face-flat torch halos, ceiling and floor lighting.
The house's 400 ms cold turn is much patchier than hybrid. Cacheless emitter
response is also substantially slower in both directions; in its one-second
removal capture the floor is still strongly orange, unlike hybrid/cached view.
These are not qualified replacements. The following corner change addresses
presentation only; it does not dismiss the cacheless response weakness.

The first cacheless response comparison is **invalid as an architectural
judgment**: inspection found scene publication pruning events by CPU world-cache
pages. Cacheless therefore received changed geometry but no geometry-retention
events. Publication now keeps the scene-derived affected domain for both
receiver owners, including changed-sunlight events. Native regression covers
scene publication through worker submission and GPU event bytes with no world
pages, not just manually queued packets. No receiver allocations are recreated.

Sparse-cache measurement also distinguishes missing support from measured
darkness: the outgoing-flux atomic carries an observation bit. An edit whose
one-voxel neighbourhood has no sampled world faces uses unknown-flux distance
retention, even when a cache exists elsewhere. A genuinely sampled dark
neighbourhood still reports measured zero. The native regression covers both;
the existing measured-flux/urgency CPU-parity test also passes. This matters
for any bounded world cache, including the hybrid, not just cacheless mode.

Offscreen post-15s cadence p95 was 25.0–26.0 ms across the nine runs (including
the harness's sleep); this is not windowed frame pacing. World-display CPU
site metrics are unavailable in view-first and must not be reported as zeros
or perfect stability. Inspect the actual captures instead.
