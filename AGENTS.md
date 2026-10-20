# GI development workflow

## Checkpoints and regression tracking

- The user wants small, useful commits throughout the work, not a large uncommitted accumulation. Commit completed, coherently scoped changes after focused verification, without waiting for another request to commit.
- Preserve a tested baseline before starting a new performance or architecture experiment. Do not mix unrelated algorithm, storage, presentation, and harness changes in one commit when they can be verified independently.
- Keep intermediate commits buildable. If existing changes are inseparable, make an honest integrated checkpoint; do not manufacture untested intermediate states or claim measurements for them.
- Explain the change and verification performed in the commit message. Mark experimental work and known limitations explicitly. Do not claim tests were rerun when relying on previous verification of unchanged code.
- Associate performance evidence with the exact commit, any dirty diff, profile, hardware, viewport, and settings. Keep a compact result summary in the commit message or tracked documentation; large captures/logs stay under ignored `target/`.
- Preserve user changes; do not sweep unrelated work into commits. Keep commits local unless the user requests publishing. Do not rewrite existing history to conceal failed experiments.

## Lean, risk-based verification

- The priority is improving real gameplay performance and lighting response, not repeatedly exercising solved problems. Use the smallest relevant check for each change; do not run every validation profile or the full suite by default.
- Documentation/workflow-only changes need a diff review, not a build or GPU run. Reuse valid evidence for unchanged code.
- During iteration, use focused unit tests or a relevant isolated compute benchmark. Run the gameplay profile when end-to-end behavior/performance needs checking, holding the preferred corner look fixed.
- Compare frame-time tails together with lighting freshness, edit response, and visible stability. Faster frames obtained by starving transport are not an improvement. Sample coverage/history is not convergence.
- A preliminary baseline/candidate pair is enough to guide an iteration. Repeat paired runs when noise makes the result uncertain or before claiming a small performance gain; do not mandate three pairs for every edit.
- Use visual comparisons for presentation changes, source tests for emitter/multi-band transport changes, and native parity/upload tests for their affected paths. Run broader checks at meaningful integration checkpoints or when risk justifies them.
- Qualify sampler/path-sharing accuracy using raw matched prefixes, without temporal smoothing or adaptive spatial reconstruction. Report that separately from presentation quality with reconstruction enabled. Borrowed support is not independent samples; never use the smoothed image to lower the sampler's accuracy threshold. See `docs/validation.md` for controls and metrics.
- Judge gameplay changes by what the player sees, against a baseline worktree run of the same profile. Measure neighbour blotchiness, settle time and late flicker (`tools/live_site_quality.py` on `live-hallway`/`cave`), edit response (`tools/response_timing.py` on `response`) and frame-time tails. Accuracy against a reference is a separate check, not the goal.
- The surface cache is biased by design (multi-bounce light arrives over time). Qualify its raw noise against a frozen converged cache (`VOXEL_GI_VALIDATE_CACHE_SWEEP=1`, `tools/cache_flip_gate.py`), not against the finite three-bounce oracle.
- Do not edit `assets/` while a validation run is active: the asset watcher aborts the run.
- The known toggle/buffer-generation failure is fixed and regression-covered. Keep the optional `stress` profile, but do not routinely run it for sampling, scheduling, quality tuning, documentation, or ordinary performance work. Use it for relevant buffer lifetime/cache-generation/material extraction/display-mode changes, a renderer dependency upgrade, or a returning symptom. Prefer the targeted headless regression test when sufficient.
- Never accept missing-asset, black, unfocused, or incomplete captures as gameplay evidence. Do not change the user's screensaver/security settings. Completion of a workload is not a quality/performance pass.
- Prefer `VOXEL_GI_VALIDATE_HEADLESS=1` for unattended image/lighting checks. This uses Bevy's offscreen image capture with no OS window; focus is inapplicable, not a waived window-capture check. Require valid camera/target dimensions, assets, nonblank captures and clean render logs. Keep offscreen app-cadence timings separate from windowed FPS; never compare them against a windowed baseline or claim they establish interactive frame pacing. System sleep can still suspend the run.

## Preferred behavior

- Default to shared-corner direct sunlight and corner-reconstructed indirect GI, textures on, baked AO off. Keep the per-pixel sunlight comparison available.
- Resident indirect RGB uses uncapped 1024-evidence reconstruction by default. The user asked for it to be chosen with the calibrated FLIP gate after the surface cache halved the raw path count needed (2048 before). Keep `config/gi-display-unfiltered.json` as the explicit no-adaptive-spatial control; uncapped still has the existing radius/work bound.
- Do not add debounce or promote corner-based indirect transport or other architectural experiments without relevant evidence and the user's intended scope.
