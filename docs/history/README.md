# Historical studies

These records preserve experiment evidence and rejected alternatives. Many
older studies measured the explicit three-bounce estimator before the
[surface cache](../surface-cache.md); newer records qualify the current transport
and view/cache architecture. A study's defaults apply to its recorded commit,
not necessarily today's runtime. For the current design
read [architecture](../architecture.md), [validation](../validation.md) and
[research decisions](../research.md).

Notable records:

- [Architecture design notes](architecture-design-notes-2026-10.md),
  [research notes](research-design-notes-2026-10.md),
  [direction plan](direction-2026-10-21.md), and
  [validation notes](validation-design-notes-2026-10.md): detailed October
  development records, including superseded APIs and removed harnesses.

- [Ownership comparison, 2026-10-23](option-comparison-2026-10-23.md): all-near,
  hybrid, cacheless and +cache 32/48/64, windowed uncapped and visible behavior.
- [Two-way face cache](two-way-cache.md): shared RGB history between visible
  sampling and a small persistent cache, with rendered and analytic gates.
- [View-first transport](view-first-transport.md): hybrid, +cache and cacheless
  comparison before two-way history exchange.
- [Hallway outliers](hallway-path-outliers.md): heavy-tailed sparse-hallway
  paths, roulette and the guiding experiment (removed).
- [Path reuse](resident-path-reuse.md): fresh spatial path sharing (removed).
- [Whole-range history](whole-range-history.md): the retention curve still in use.
- [Early spatial reconstruction](early-spatial-reconstruction.md): fixed
  evidence-target experiments; current presentation uses variance/bias weights.
- [Streaming performance](streaming-performance.md) and
  [experiments](experiments.md): measured wins that remain in the engine.
- [Code review, 2026-10-13](CODE_REVIEW_2026-10-13.txt).
- [Windows first run, 2026-10-30](windows-first-run.md): RX 7900 XT baseline
  on Vulkan and DX12, the reference for Windows fixes.

Dead experiment branches no document cites are kept as `archive/<branch>`
tags (2026-10-30: 31 superseded `experiment/*` and `codex/*` branches), so
their commits stay reachable without crowding the branch list; `git branch
<name> archive/<name>` restores one. Branches the documents cite by name,
and those checked out in worktrees, remain branches.
