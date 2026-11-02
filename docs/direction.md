# Lighting and simulation direction

Voxel GI is a client-side RGB lighting system. The current implementation is
in [architecture](architecture.md); the release integration API is in the
[core crate README](../crates/voxel_gi/README.md).

## Current lighting design

Every visible surface is handled by the camera's view tier. It discovers
voxel faces from depth and normal prepasses, preserves their world-space
identities across camera motion, and accumulates raw light on the GPU.
Small projected faces use the per-pixel far tier instead.

A persistent world cache within `cache_radius` (64 voxels by default) keeps
nearby offscreen faces warm. View receivers and resident faces exchange the
same raw history without adding duplicate samples. Secondary rays may end on
supported cache entries; uncached view paths retain unfinished continuations
across bounded dispatches. Geometry includes all pages declared by the host,
regardless of cache radius.

Reconstruction and temporal display filtering smooth the result separately
from raw accumulation. Borrowed display support is not independent samples.
The transport estimator remains oriented per face; corner reconstruction
controls its appearance rather than changing its sampling origin.

The [ownership comparison](history/option-comparison-2026-10-23.md) and
[two-way cache study](history/two-way-cache.md) record the decisions behind
this design. The [original plan](history/direction-2026-10-21.md) preserves
the earlier hybrid layout and measurements; those are not current API or
memory-layout descriptions.

## Future simulation

Heat, ionizing radiation, material transitions and dose are outside this
crate. A multiplayer server must be able to decide those world changes
without a client camera or GPU lighting cache.

A future simulation may use deterministic visibility and bounded-range
source transport on a slower tick. It can share geometric concepts with GI,
but needs its own units, sampling guarantees, persistence and authority.
No such simulation field or query API is provided here. A simulation that
melts a block should update the host voxel material and notify GI through
the normal edit interface; its visible emission then participates in GI.

## Next experiments

[Research priorities](research.md#after-01) list the planned work. Large-world
scene-build latency and far-tier motion stability are the main current
integration limits. A cache-radius setting limits lighting state, not geometry
storage or scene-build cost. Performance changes must preserve edit response
and the preferred corner lighting under the [validation protocol](validation.md).
