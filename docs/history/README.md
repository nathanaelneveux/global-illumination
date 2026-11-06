# Historical studies

These records explain how the renderer reached its current design. Read
measurements in the context of the revision and settings in the study.
Use [architecture](../architecture.md) for current behavior and
[validation](../validation.md) to choose a check for a new change.

## How we light a view

- [Lighting ownership comparison](option-comparison-2026-10-23.md) compares
  world, hybrid and view-driven lighting while moving and turning.
- [Two-way face cache](two-way-cache.md) shows how visible faces share
  history with the persistent cache. For the earlier candidates, see
  [view-first transport](view-first-transport.md).
- [Far tier](far-per-pixel.md) measures the cost of lighting small faces in
  screen space and the motion flicker tradeoff.

## How we tested the light

- [Cave reference and sample sweep](cave-sampling.md) calibrates image
  error against a visually accepted reference for that estimator.
- [Hallway outliers](hallway-path-outliers.md) investigates rare, bright
  paths in a dark hallway and records the roulette and guiding experiments.
- [Path reuse](resident-path-reuse.md) tests sharing fresh paths between
  faces.
- [Whole-range history](whole-range-history.md) measures how much history
  to keep after an edit at different distances.
- [Early spatial reconstruction](early-spatial-reconstruction.md) explores
  how neighboring faces can help when each face has few samples.

## How we measured cost

- [Streaming performance](streaming-performance.md) traces frame stalls and
  the changes that reduced them.
- [Experiment results](experiments.md) covers tracing, scheduling and
  presentation.
- [Windows first run](windows-first-run.md) records the RX 7900 XT
  validation on Vulkan and DX12.

## Design and review archives

- [Architecture design notes](architecture-design-notes-2026-10.md)
- [Research notes](research-design-notes-2026-10.md)
- [Direction plan](direction-2026-10-21.md)
- [Validation notes](validation-design-notes-2026-10.md)
- [Code review from 2026-10-13](CODE_REVIEW_2026-10-13.txt)

## Recover an experiment branch

Superseded experiments may have an `archive/<branch>` tag. Use
`git branch <name> archive/<name>` to restore one. Use the branch name
directly when a study points to an active branch.
