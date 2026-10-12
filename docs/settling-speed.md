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
