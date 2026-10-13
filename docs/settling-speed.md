# Settling speed — October 10, 2026

Target: halve time to visually close-to-settled lighting without increasing
overshoot, starving other receivers, or blocking the player. Coverage is not
convergence. The starting runtime is `febcaf4`; the edit-guided experiment in
that commit is test-only, not the runtime policy.

## Baseline and isolated execution study

Apple M4, 16 GiB, Metal, Rust 1.95, Bevy 0.19.1, registry BVW 0.17. Dev profile
(root opt1, dependencies opt3), shared Cargo target. No concurrent compiler or
GPU workload during measurements. Initial study adds only test code.

Native lane-layout sweep: 2,048 receivers, 48 rotating-order batches (first
eight warmup), identical paths, all five means/second moments/direct values
checked with 17-receiver tails at 1/7/16/64 paths. GPU p50 milliseconds at 16
paths per face:

| Scene | 4 lanes (baseline) | 1 | 2 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open sky | .220 | .324 | .223 | .217 | .221 |
| Many-emitter cave | 3.978 | 6.070 | 4.050 | 3.301 | 3.480 |
| Skylight ceiling | .985 | 1.614 | 1.044 | .945 | .920 |

Eight lanes save about 17% cave kernel time, **not 2×**. At four paths they
waste lanes: cave 1.223 ms (four lanes) versus 1.874 ms (eight). A production
layout would need to respect batch size. This first sweep is not yet promotion
evidence for end-to-end speed.

Removing roulette at the default three-hit depth was independently tested with
four live sequence offsets, 128 batches of 16 paths, and independent 65,536-path
references. At 2,048 paths, pooled five-band relative p95 error changes:
skylight floor .138→.110, ceiling .0865→.0742, cave .0610→.0512. Cave GPU batch
time increases .951→1.031 ms. Ceiling worst bright excursion actually increases
4.388→4.720 relative to reference luminance, despite better aggregate error.
Analytic absorbing emitters are unchanged. **Not a universal winner** and not
promoted. PBRT explains both the variance cost of roulette and why terminated
GPU lanes may not save proportional execution time:
[Monte Carlo efficiency](https://pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency),
[GPU execution](https://pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Mapping_Path_Tracing_to_the_GPU).

Baseline streamed cave: `cave-1791681216380638000-87350`, clean `febcaf4` at
build/launch, offscreen 2560×1440, long cave/background/temporal/reference/frame
timings enabled. Defaults: 16 paths, 4 ms worker target, 8,192-receiver ceiling,
2 ms scene publication, 75 ms display response, 50 ms display upload. Five valid
nonblank captures, clean render logs. Post-timing reference replicas: 262,144
paths each, relative RGB p95 disagreement .413%. At 30/60 seconds raw receiver
p95 error is 15.05%/9.62%; display 12.51%/9.80%. At 60 seconds 35/36 raw and
display receivers are within 10%. Corner excess variation p95 over 40–60 s is
.08230, with all 144 corners present and new paths continuing. Repeated history
cuts remain visible in the measurements.

After 15 s, excluding captures: offscreen app intervals p50/p95/p99/max
21.30/27.33/30.23/73.00 ms; GI main-system-body sum
1.68/5.60/7.75/18.03 ms. This is **not windowed FPS or a 60 FPS guarantee**.
288.72 M paths, 8,492 batches, up to 1,540 pages / 1.28 M probes.

Ignored evidence: `target/settling-{layout,roulette}-study.log`,
`target/settling-baseline-{build,cave}.log`,
`target/settling-baseline-{reference,temporal}-analysis.json`, and the run's
captures/JSON in `target/gi-validation/`. Reproduce isolated studies with
`cargo test --locked --offline settling_lane_layout_study -- --ignored --nocapture`
and `settling_roulette_study` respectively. Both passed; neither changes runtime.

## Local refinement-index invalidation

The nearby/regional index previously followed the renderer's global topology
generation. Any distant streaming change recopied the local receiver lists and
reset their fallback cursor. It now follows receiver-membership changes within
the two-page focus window only. Rebuilds look up at most 125 page coordinates,
instead of scanning every loaded page, and preserve the cursor unless the
player changes focus page. Face/material changes still invalidate lighting as
before; all-world fairness, history, rays and GPU display lifetime are unchanged.
All 23 runtime policy/publication tests passed, including new distant/local
addition/removal/movement coverage. This eliminates work by construction; an
end-to-end speed claim requires the streamed comparison below.

Streamed follow-up `cave-1791681993393562000-87849`, runtime `1088081` (only
test-only spatial-study edits made during the run), identical launch settings:
292.81 M paths versus 288.72 M baseline. All five captures valid, clean logs,
reference p95 disagreement .413%. Display p95 relative error at 30/60/180 s:
13.11%/12.73%/8.50%, versus baseline 12.51%/9.80%/6.66%. Sustained display
settling median is essentially unchanged (20.96 versus 20.86 s); p95 is worse
(166.15 versus 119.26 s). Corner variation p95 at 40–60 s falls .08230→.05785,
but streaming revisions differ, so this does not establish a general accuracy
gain. Both runs finish with 35/36 receivers sustaining 10% error.

Offscreen interval p50/p95/p99/max: 21.88/26.72/30.23/36.83 ms; GI body sum
2.11/6.16/7.86/15.63 ms. Maximum trace-dispatch body falls 14.37→9.42 ms,
but medians/tails do not support a broad speedup claim. Keep the local
invalidation for its bounded work and cursor-correctness improvement, **not as
the 2× settling result**. Remove the now-unused global topology counter; the
renderer already uses its independent dirty/lifetime bookkeeping.

## Spatial reconstruction pilot — not promoted

Test-only `settling_spatial_study` compares the existing connected four-face
corner reconstruction with an additional coplanar five-tap filter (center
weight four, edge neighbors one), both fixed and guided by per-channel sample
variance. Hard geometric boundaries exclude neighbors. This draws on the
variance/edge-stopping principle of [SVGF](https://cg.ivd.kit.edu/publications/2017/svgf/svgf_preprint.pdf),
not its screen-space implementation. Simulation values remain untouched.

M4/Metal/dev profile; four disjoint live sequence prefixes, 16-path batches
through 2,048 paths. Dense one-voxel neighborhoods around every third receiver
of the floor, ceiling and cave fixtures. Reference: 65,536 disjoint paths per
face, **unfiltered** connected corners. Reference submissions are bounded to
1,024 paths after the initial monolithic cave submission timed out; that failed
attempt is not accepted evidence. The complete bounded run passed.

At 256 paths, RGB relative corner-error p95 baseline/fixed/guided:
floor .369/.282/.320, ceiling .279/.216/.242, cave .0525/.1035/.0470.
The fixed filter's cave high-sample approximation error is 10.54% p95 (13.39%
worst): lower noise conceals real gradients near sources. Guided high-sample
deviation is only .051% p95, but its cave improvement is small and worst early
64-path error increases .193→.208. No filter cost or temporal startup is included
in this pilot; there is no 2× result and neither candidate is promoted.
Evidence: `target/settling-spatial-study.log`. Production remains unchanged.

## Fair refinement reservation

Experimental commit `0aba755` prevents a full streaming-coverage queue from
excluding adaptive refinement. A quarter of opportunities stays with the global
sweep, a quarter with initial coverage, and half with refinement; empty queues
lend their slots. Explicit edits still preempt all non-global slots. All 24
focused runtime/publication tests and the dev build passed.

Run `cave-1791846614020288000-29475`, same M4/profile/viewport/settings as baseline,
clean code at launch. Layout source edits made after the timed workload finished
were not compiled into this process. Five valid captures, native post-timing
reference. At 15 s, display p95 error improves .681→.283 and 28/36 receivers are
within 10% versus 13/36. Sustained raw settling median improves 27.53→12.80 s
(2.15×), but display median improves only 20.86→15.02 s and its p95 worsens
119.26→137.93 s. **Not yet a 2× visual-settling result.** At 60 s display p95
error is 9.38%; at 180 s 8.48%. Corner variation p95 at 40–60 s is .05670,
minimum 128 new paths; at 160–180 s .03772, minimum 64. No hidden pause in work.

306.34 M paths, final coverage 99.22%. After 15 s excluding captures: offscreen
interval p50/p95/p99/max 20.99/26.85/29.89/127.96 ms; GI body sum
1.98/6.16/8.45/24.11 ms. Large maxima remain; this is not a frame-pacing win.
Evidence: `target/settling-reservation-{tests,build,cave}.log` and
`target/settling-reservation-{reference,temporal}-analysis.json`.

## Batch-aware cooperative lanes

The execution-only candidate uses eight lanes for batches of at least 16 paths,
four for smaller batches. Quality/layout pipelines are bounded (3 depths × 2
reach modes × 2 layouts) and compiled on the worker. Samples, spectra, history,
ray lengths, bounce depth, scheduling budget and output buffer formats are
unchanged. Legacy test shaders retain their original layout. Automatic layout
switches invalidate auto-layout bindings correctly.

Confirmation at the same native fixture/work settings as the initial sweep:
cave four/eight-lane GPU p50 4.655/4.275 ms, worker 6.859/6.797 ms. Gains varied
between runs (roughly 4–17% kernel time), not a 17% guaranteed whole-engine gain.
Native automatic/fixed layout parity covers 17-face tails at 1/7/16/64 paths,
all five means/moments/direct values and continuing sequences. Standard and
emissive native CPU/GPU references report RMS 0.000000; resource reuse/growth
tests, formatting and strict all-target Clippy pass. End-to-end qualification
follows separately. Logs: `target/settling-layout-{confirm,pair,parity,lifetime,clippy}.log`.

## Corner-accuracy oracle

The offline reference now traces exposed coplanar neighbors as well as the fixed
center faces and records their connected-corner reconstruction at converged
equal confidence. This matches the preferred display's spatial definition;
face-center accuracy alone is not visual-corner accuracy. Unknown/missing
corners remain missing in the analysis, not a black measurement or a pass.

Schema 2 uses two disjoint **65,536-path** replicas per member face (previous
center-only reports used 262,144). Reference-replica disagreement is reported
for corners separately. The extra neighborhoods run only after timing/capture
completion. `python3 tools/reference_analysis.py RUN --corners` reports error,
sustained settling and positive luminance peaks relative to that independent
final-scene target. This is a presentation-space proxy, not tone-mapped pixel
error; early frames can have different residency from the final scene. No
last-frame or blurred-candidate target is substituted for truth.

Verification: native-independent reconstruction test (shared corners and
disconnected diagonals), three Python analysis tests (including missing spans
and known overshoot), fmt, strict all-target Clippy and dev build passed.

Combined runtime `c18380b`, run `cave-1791847590335826000-30282`: independent
corner replica disagreement p95 .472%; 140/144 corners sustain 10% error to
the end, median 11.68 s, p95 17.26 s. At 15 s 119/144 are within tolerance;
at 30 s all 144 are. Positive luminance peak p95 .1027, maximum .3045.
The diagnostic-only baseline backport `5c2d333` (runtime `febcaf4`) ran as
`cave-1791847874844864000-30513` in the temporal-baseline worktree. Its
143/144 sustained corners give median 15.89 s, p95 21.99 s and peak p95 .1021,
maximum .3045. However, its 15 s capture is effectively black and every
measured corner is absent then. **Reject that startup visual comparison**;
the later data alone does not establish a 2× improvement. Do not turn missing
lighting into a favorable baseline. Candidate 15 s capture is valid.

## Exact empty-donor culling

Page publication now skips display-prior searches when no receiver page can
exist within the donor algorithm's maximum three-cell coordinate reach.
Existing local receivers and all reachable border pages use the original
connectivity/measurement checks. This does not change history, seeds, rays or
work budgets. The search-bound test enumerates all six donor searches; page
tests cover boundaries, negative coordinates and existing local data.

Paired CPU test, M4/dev, `c18380b` plus this change, 1,024 newly loaded floor
faces beside a measured page, 40 alternating rounds (eight warmup): exact RGB,
sky and confidence parity; hash lookups 41,292→2,856; seed-stage median
233.625→33.000 µs (7.08×). This is **not a whole-frame or settling speedup**.
Both new regular tests, all history tests and the ignored parity/benchmark
passed. Evidence: `target/settling-donor-{tests,history,bench}.log`.

## Larger batches — not promoted

Equal 32,768-path native batches, M4/dev, five rotating modes, 48 rounds/eight
warmup (`12f887a` plus test-only study): cave GPU medians for 16/8, 32/8,
64/8, 32/16, 64/16 paths/lanes are 4.241/4.503/4.836/4.172/4.495 ms;
worker medians 6.670/6.606/6.627/6.500/6.578 ms. Ceiling and open sky are
also approximately tied. Larger batches sacrifice receiver refresh frequency
without a compelling throughput gain. Keep the 16-path default and limit.
The native study passed; log `target/settling-batch-study.log`.

## Cardinal light-node importance

`e8f2fd0` plus the shader reduction replaces six conditional normal projections
with signed-axis vector operations. It computes the same maximum cosine bound;
no proposal, PDF, path, history or work-budget change is intended. Absent normals
contribute -1 before adding the bounded angular spread.

M4/Metal/dev, 2,048 cave receivers × 16 paths, 64 alternating rounds (eight warm):
baseline/candidate GPU medians 4.362/3.665 ms; confirmation 3.075/2.509 ms.
Clock-dependent absolute times differ, but both pairs save 16–18% kernel time.
All five means/moments/direct values match at 1/7/16/64/256 paths; the separate
3,872-choice native selection/forward/reverse-PDF CPU oracle has zero differing
choices. Both native tests passed. This is not a whole-frame or settling claim.
Logs: `target/settling-light-importance.log` and
`target/settling-exact-importance-{confirm,pdf}.log`.

## Private-cache integration and facing-priority qualification

M4/Metal/dev, 2560×1440 headless, background streaming, preferred corner look,
16 paths and 4 ms transport target throughout. Every cave run below has valid
captures and its own disjoint native corner reference. App intervals are not FPS.

| Runtime / run | Sustained 10% corner error median / p95 | Within 10% at 15 s | Positive peak p95 / max |
| --- | --- | --- | --- |
| `c18380b` / `cave-1791847590335826000-30282` | 11.68 / 17.26 s | 119/144 | .1027 / .3045 |
| `3e2d459` / `cave-1791849222156685000-31687` | 9.08 / 18.27 s | 124/144 | .1044 / .3043 |
| `e3d1d42` / `cave-1791849592310651000-31991` | 8.47 / 13.65 s | 144/144 | .0975 / .2639 |

`3e2d459` is the private-cache synchronization candidate, not a different
estimator. First display publication advances from 11.81 to 9.08 s. Total paths
are approximately tied (375.13 versus 372.82 M). After 15 s excluding captures,
its offscreen interval p50/p95/p99/max is 20.93/27.00/30.00/127.89 ms; GI body
sum is 2.16/6.51/8.61/69.62 ms. The worst selection event is 63.63 ms. Thus the
isolated upload improvements are useful but do **not** establish stable frames.
A windowed streaming attempt (`streaming-1791849182476730000-31621`) correctly
rejected its unfocused black capture: no interactive 60 FPS evidence.

The separate `codex/gi-facing-refinement` branch prioritizes local receivers
with at least one camera-facing face, while tracing every due face once selected
and retaining global/coverage/urgent reservations. Although its early cave
accuracy is better, corner temporal variation p95 at 40–60 s rises to .1454
versus .0683 for `c18380b`; nearby refresh-age p95 becomes 36.81 s because hidden
faces lose opportunistic work. It is **not promoted** as a general default.
All 11 scheduling and 25 runtime-policy tests, Clippy and dev build passed.

Paired 30 s headless/background `response` runs (14 valid captures each):
facing `response-1791849801774352000-32121`, diagnostic-only `febcaf4` control
`response-1791850047067551000-32354` in the temporal-baseline worktree. Two
floor receivers' last red-light residual above 10% after emitter removal improves
2.719→1.774 s (1.53×), normalized using pre-add and pre-remove display means.
This is an event-response proxy, **not an independent accuracy reference**.
An earlier control launch failed the compiled-root guard and is excluded.
Logs/analyses use the `target/settling-private-*`, `settling-facing-*` and
`settling-control-response-verified.log` prefixes. No 2× overall claim follows.

## Exact-kernel integration and time-bounded catch-up

Same settings/hardware as above. `59d721c` ran clean at launch as
`cave-1791850540212693000-32968`; later documentation/test-only source edits were
not compiled into that process. All five captures are valid. Corner settling
median/p95 9.28/16.31 s; 128/144 within 10% at 15 s, p95 error .1296; peak
p95/max .1021/.3044. 386.95 M paths. Variation p95 at 40–60 and 160–180 s is
.07388/.05911, with at least 176/128 new paths. After 15 s excluding captures,
offscreen interval p50/p95/p99/max 20.88/26.88/30.22/89.29 ms, GI body
2.26/6.96/9.41/47.89 ms. First publication 9.28 s, replacement 6.55 s.

The isolated `0ebc3c8` experiment removes only the 8,192-record catch-up ceiling,
retaining the 1.5 ms deadline checked every 512 records. It ran clean at launch as
`cave-1791850901451970000-33399` (test-only study edits later, not compiled).
All five captures valid. Median/p95 9.52/15.87 s; 138/144 within tolerance at
15 s, p95 error .08886; peak .09887/.30451. 451.62 M paths. Variation
.09264/.06343, minimum fresh paths 192/96. Offscreen interval
20.54/27.31/31.19/68.93 ms, GI body 2.33/6.99/9.17/32.21 ms. First publication
9.68 s is not better; replacement 4.69 s is faster. Run-to-run throughput/temporal
differences exceed the narrow change's expected effect: do not attribute all of
them to removing the cap. This preserves a buildable experimental checkpoint,
not a demonstrated 2× result. Logs/analyses: `target/settling-{exact,time-budget}-*`.

## Power-MIS pilot — not promoted

Test-only study on `0ebc3c8` compares balance with exponent-two power weighting
for paired light/cosine techniques. The terminal one-sample mixture retains
balance, following [PBRT's MIS discussion](https://www.pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency).
No history or presentation change. Four disjoint live sequences, 16-path batches
through 2,048 paths; analytic two-emitter truth or disjoint 65,536-path baseline
references. Candidate high-sample results are checked separately, never used to
move the target. Native references are split into bounded 1,024-path submissions.

On M4/Metal/dev, analytic emitter p95 error at 256 paths improves .00541→.00176;
peak p95/max .05328/.21219→.01547/.03240 at effectively equal GPU cost.
Both nonemissive skylight fixtures are identical. However, the many-emitter cave
barely improves at 256 paths (.19403→.19188) and early peak p95/max worsens
.55505/1.08611→.56092/1.11080. No general/cave win is established, so the runtime
estimator remains unchanged. Ignored native study passed;
`target/settling-power-mis.log`. This also refactors the previous roulette study's
harness; its historical numbers above are from the earlier harness, not a rerun.

## Conservative light support pilot — not promoted

`settling_light_support_study`, test-only diff on `28624e0`, uses the axis-aligned
receiver plane and emitting normal mask to exclude a child whose entire bounds
cannot connect. No energy threshold, visibility assumption or band-specific
rejection; other branches retain the power mixture. This is a restricted voxel
case of orientation-aware [many-light bounds](https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf),
not a port of adaptive tree splitting.

Same fixed-work native fixtures and independent references as the power-MIS
pilot. Analytic CDF emitters and both skylight fixtures are unchanged. Cave
256-path p95 error only improves .19403→.19118; peak p95/max improves
.55505/1.08611→.53581/1.04392, but GPU batch median rises .74079→.77796 ms.
The extra bound tests do not pay for themselves here. Native study passed;
`target/settling-light-support.log`. Keep runtime proposals unchanged.

## Valid baseline repeat

Diagnostic-only `5c2d333` (runtime `febcaf4`), baseline worktree,
`cave-1791851270749646000-33719`: compiled-root guard passed and all five captures
are valid; the 15 s image was inspected. This replaces the earlier rejected
black-startup comparison, not its historical record. Same M4/dev/background/
2560×1440/default settings. Independent corner reference disagreement p95 .472%.

Sustained corner error median/p95 12.80/21.24 s, all 144 eventually within 10%;
at 15 s 103/144 within tolerance, p95 error .4293. Peak p95/max .10205/.30430.
Variation at 40–60/160–180 s .09921/.05821, min new paths 224/128.
343.42 M paths. After 15 s excluding captures, offscreen interval
21.81/26.39/30.08/89.03 ms and GI body 2.29/6.73/8.76/17.51 ms.
The candidates substantially reduce 15-second error, but the sustained settling
times are still only about 1.3–1.4× better and occasional CPU tails remain.
Evidence: `target/settling-control-repeat-*`; no 2× settling/60 FPS claim.

## Continuous-distance ranking pilot — not promoted

Isolated branch `codex/gi-distance-refinement`, `e982427`, applies soft inverse
square distance to opportunistic refinement only; reserved global/coverage and
urgent work stay unchanged. Eleven scheduling and 24 runtime tests, strict
Clippy and dev build pass. Streamed long cave
`cave-1791848585119270000-31043` has valid 15 s imagery, 141/144 corners
sustained, median 11.31 s and p95 22.36 s. At 15 s error p95 .359 versus .178
for `c18380b`; positive peak p95 .1021/max .3045 is essentially unchanged.
40–60 s corner variation .07327 versus .06831. This pilot also includes exact
donor culling, so it is not a clean isolation of that change. There is no
compelling settling win: keep the distance policy off the retained branch.

## Private mapped-view batching

Opening a native mapped view for each sparse write is unnecessary: replacement
buffers are private and remain mapped through catch-up. Open each whole-buffer
view once per bounded synchronization batch instead. All views still end before
installation/unmapping. The CPU fallback, atomic publication, buffer generation
and exact bytes are unchanged.

M4/dev, `4398697` plus this change: 512 scattered 208-byte writes, 48 alternating
rounds/eight warmup, exact byte checks after each mode. Per-range versus
per-batch median 30.750→1.875 µs. This is a **view-management microbenchmark**,
not a 16× engine claim. The native generation/material test passes with mapped
byte parity, growth/shrink/equal-size replacement, actual post-unmap submission
and CPU fallback. Logs `target/settling-private-{views,native}.log`.

Private catch-up now updates its own indices, mirrors and initial allocation
in place. Unlike live publication, it needs no rollback-safe preflight maps,
dirty upload sets or sorted/coalesced driver transactions. Removals precede
additions, so released slots remain reusable. Capacity/hash exhaustion poisons
the private generation; installation rejects it and the caller retains the
live generation while rebuilding. Live transactions are unchanged.

M4/dev, `c3be7bf` plus in-place change: paired native 512 changed values plus
32 additions/removals, 48 alternating rounds/eight warmup, 8,192 initial probes.
Previous transactional route (already using one mapped view per batch) versus
in-place: 79.500→42.958 µs median (1.85×). Every round compares slot/hash maps,
free slots, high water, all mirror and mapped bytes. Regular edit/duplicate/
exhaustion parity, all 17 regular render tests, native generation/material
lifetime, strict Clippy and dev build pass. Logs
`target/settling-private-inplace-{tests,study,render-tests,native,clippy,build}.log`.
End-to-end timing remains separate; the 1.5-ms catch-up budget is unchanged.
