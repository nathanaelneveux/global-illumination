# Temporal stability regression checks

Coverage and sample history are not convergence. A faster noisy response is not
a win over a slightly slower smooth fade. Use fixed world-space receivers and
keep the scene, display mode, settings, hardware and timing domain explicit.

## Recorder

Build first, then run the existing house workload with
`VOXEL_GI_VALIDATE_TEMPORAL=1`, `VOXEL_GI_VALIDATE=1`,
`VOXEL_GI_VALIDATE_HEADLESS=1`, `VOXEL_GI_VALIDATE_PROFILE=house`
(or `house-edits`), and `VOXEL_GI_VALIDATE_EXIT=1`.
Also set `VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$PWD"`: the compiled manifest
directory must match, or the app exits before recording evidence. When switching
worktrees sharing a Cargo build directory, invalidate the application artifact
(for example `touch src/main.rs` before building). Confirm focused tests actually
ran: zero matching tests is not verification. This root check does not certify
the commit or dirty diff; record those separately with the experiment.
`target/gi-validation/<run>/temporal.json` contains:

- 25 fixed underside receivers, observed every Update after GI publication;
- measured RGB, second moments, history count and sequence, revisions, freshness;
- filtered display and confidence;
- CPU render-cache mirror values and four connected, confidence-weighted corners.

One worker result can be accepted per Update. These observations therefore do
not skip accepted batch transitions, unlike the broad 250 ms field sampler.
Unchanged states are event-compressed. Missing receivers are explicit nulls.
The 200,000-event limit fails the workload rather than silently truncating it.
The recorder does no filesystem writes until the workload finishes. It is opt-in
and does not alter transport, histories, scheduling or smoothing.

The mirror is *queued publication*, not GPU readback. Reconstructed corners are
pre-material/pre-tonemapping and exclude the shader's unpublished-center prior.
Validated Bevy captures remain the end-to-end check. The current recorder does
not retain pre-dispatch history caps or individual incoming path batches; use
sequence/history changes to locate an anomaly before adding such diagnostics.

## Analysis

```
python3 tools/temporal_analysis.py --test
python3 tools/temporal_analysis.py target/gi-validation/<run>/temporal.json \
  --field corners --interval 10:30 --interval 30:50 --interval 50:70
```

Analyze `measured`, `display`, `mirror`, then `corners` to localize excursions.
RGB is compressed as `x / (0.25 + x)` to expose changes in dark receivers without
letting bright ones dominate. This is not an image-space perceptual score.
**Excess variation** is accumulated absolute change minus net endpoint change,
computed separately for each RGB channel; the worst channel is reported.
A monotone fade scores zero regardless of its speed. Reversals score positively.
Repeated identical observations do not affect the score. Missing spans break
continuity and are reported instead of being interpreted as black.

This diagnoses stability, not accuracy: frozen wrong lighting also scores zero.
Always pair it with freshness/sequence progress, screenshots and reference error.
`--reference reference.json` accepts an independently established RGB map keyed
by `x,y,z` (or `x,y,z:corner` for corners). With a reference, analysis reports
time-weighted error, overshoot and the last threshold excursion, requiring at
least two seconds of subsequent in-tolerance observations. This is sustained
settling *within the observed window*, not proof it stays settled indefinitely.
Without an independent reference no settling time is claimed; the last noisy
frame is never silently promoted to ground truth.

For the reproduced house-turn regression, use `--field corners --interval 30:50
--max-p95 0.10 --min-new-paths 32`. This explicitly requested stability/freshness
gate exits nonzero if variation exceeds the limit, a receiver is missing, the
interval is incomplete, or any receiver accepts fewer than 32 new paths. It
rejects a frozen cache even if it looks temporally quiet. These thresholds are
calibrated to this workload, not universal quality defaults; accuracy and
rendered-image checks remain separate. See [the initial investigation](temporal-regression-investigation.md).
Use `VOXEL_GI_VALIDATE_BACKGROUND=1` for the additional continuously moving loader
and compare it only against the same configuration. The
[scene-history audit](scene-history-audit.md) records improvement that still
misses this strict threshold under background load; do not report it as a pass.

## Comparison discipline

- Validate assets, dimensions, nonblank captures and clean render logs first.
- Compare identical scenes. `ad6ba8e` predates the nearby emissive cave, so a
  default-world revision comparison also changes geometry and source load.
  Current code offers `VOXEL_GI_VALIDATE_HOUSE_ONLY=1` to omit cave/approach
  structures only during validation. The report identifies this scene control.
- Keep the preferred corner look, textures on, AO off. Record exact commits and
  dirty patches. Separate normal-world content changes from algorithm changes.
- Offscreen app cadence is not interactive FPS. Recorder/capture overhead also
  means this is a quality diagnostic, not an uninstrumented throughput result.
- Reproduce a failing observation before changing smoothing or accepting a fix.
  Test only the implicated workload; toggle stress is unnecessary here.
