# View-first receiver experiment

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

## Qualification

In progress. Start with matched cave, house turns/returns and edit-response captures,
with background terrain loading. Compare visible stability, reveal response,
lighting progress, memory and GPU costs before a windowed frame-pacing pair.
Do not promote either candidate based on counts or lower allocation alone.

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
