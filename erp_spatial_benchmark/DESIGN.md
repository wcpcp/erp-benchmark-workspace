# ERP Spatial Benchmark Design

## Positioning

This benchmark is not meant to be a generic ERP VQA set.

Its purpose is narrower:

> Evaluate whether a model taking ERP panoramas directly has learned the
> spherical, panoramic, observer-centered, and representation-level structure
> encoded by ERP images.

The key design principle is that benchmark tasks should correspond to genuine
properties of ERP spatial understanding rather than arbitrary QA categories.

## Why the benchmark should not mirror the training pipeline

Using the same template family and answer style as the training set would make
the benchmark too close to the SFT distribution.

Therefore the benchmark deliberately changes:

- question wording
- answer format
- task realization
- filtering strictness
- public release selection protocol

The training pipeline can provide broad supervision.
The benchmark should instead be:

- smaller
- stricter
- more closed-form
- more manually verified

## Core ability groups

The benchmark is organized around four ability groups.

### 1. Spherical localization and panoramic topology

This group measures whether the model understands that ERP is a spherical
representation with a continuous 360 horizontal ring.

Included tasks:

- `referring_grounding_bfov`
- `absolute_direction_mc`
- `relative_direction_mc`

What this tests:

- single-target localization on the sphere
- absolute ERP direction sectors
- two-target angular relations on the panoramic ring

### 2. Viewpoint-conditioned spatial updating

This group measures whether the model can update spatial relations when the
observer's front direction changes.

Included tasks:

- `camera_rotation_transform_mc`
- `object_conditioned_reorientation_mc`

What this tests:

- explicit observer rotation
- target-conditioned reference-frame reset

This is a spatial-updating ability, not a representation-consistency ability.

### 3. Observer-centered 3D layout understanding

This group measures whether the model can infer observer-centered 3D layout
from ERP direction and depth-like geometry.

Included tasks:

- `observer_distance_choice`
- `relative_3d_position_mc`

What this tests:

- which object is nearer to the observer
- which option is closest to the observer
- camera-centered multi-axis relative layout

This group is intentionally stricter than training:

- compact objects only for `relative_3d_position_mc`
- ERP-consistent geometry for relative 3D relations
- review queue for fragile camera-centered 3D cases

### 4. ERP representation understanding

This group measures whether the model understands ERP-specific representation
properties that do not arise in ordinary perspective images.

Included tasks:

- `seam_continuity_mc`
- `polar_shape_recovery_mc`

What this tests:

- seam wrap-around continuity through hard cross-boundary adjacency
- high-latitude / polar distortion awareness
- ERP-specific representation effects that do not appear in ordinary perspective images

## Why rotation consistency is not a scored v1 task

`view_transform` and `rotation_consistency` are conceptually different:

- `view_transform`: the observer turns
- `rotation_consistency`: the ERP representation is reparameterized around yaw

However, after review we do **not** include rotation consistency as a standalone
v1 scored QA task.

The issue is not that rotation consistency is unimportant. The issue is that
the obvious QA formulations are not clean enough:

- if the prompt tells the model that two ERP panoramas come from the same
  capture point, the task leaks the target relation
- if the prompt does not tell the model that, the task becomes a mixture of
  scene matching, seam relocation, and viewpoint-change interpretation
- if the prompt simply asks localization before and after rotation, strong
  localization alone can hide whether representation consistency was truly
  learned

So the current benchmark treats rotation consistency as **future diagnostic
protocol work**, not as a released standalone task.

The cleaner future direction is:

- build paired original / yaw-rotated ERP items
- evaluate whether predictions on tasks like grounding or relative direction
  remain equivariant or invariant after applying the known yaw transform
- treat seam-sensitive items as a seam-relocation stress test rather than as
  ordinary invariant QA

That would measure representation consistency more directly than any current
single QA formulation.

The current recommendation is documented in:

- [ROTATION_PROTOCOL.md](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/ROTATION_PROTOCOL.md)
- [rotation_protocol.py](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/rotation_protocol.py)

## Why `counting` is not part of the core benchmark

Counting across a panorama is useful, but it is not central to the paper's
claim about ERP spatial understanding.

It primarily measures:

- global visual aggregation
- duplicate avoidance
- whole-panorama integration

These are valuable, but they do not define the core spatial essence of ERP.

Therefore `counting` is excluded from the core benchmark.

## Leakage and same-distribution controls

### 1. Scene-level non-overlap with training

The 1k benchmark scenes must not overlap with training or SFT scenes.

This is more important than prompt holdout.

### 2. Optional group-level provenance control

If near-duplicate scenes or the same capture source exist, use `group_id` in
the scene manifest so the builder can track related scenes together and support
future stratified selection if needed.

### 3. Benchmark-only template family

The benchmark does not directly reuse the training template family.
It uses simpler and more closed-form wording.

### 4. Closed-form official answers

Whenever possible, the benchmark uses:

- multiple choice
- binary yes/no
- fixed label sets

instead of open generation.

### 5. Manual review queue

The builder exports a `review_queue.jsonl` so fragile tasks can be verified
before official release.

Current default review tasks:

- seam nearest neighbor
- seam relative direction
- seam dedup count
- seam structure continuity
- seam same-entity judgement
- polar shape recovery
- relative 3D position

## Release protocol

The current official release mode is a **single public benchmark with public
answers**.

Use:

- `benchmark_public.jsonl`
- `benchmark_public_prompts.jsonl`
- `benchmark_public_references.jsonl`

This is the simplest option for paper release and open reproduction.

## Important benchmark principle

This benchmark is not trying to maximize task count.
It is trying to maximize confidence that a strong score really means:

> the model has learned ERP panoramas as meaningful omnidirectional spatial
> representations, rather than merely matching templates or exploiting generic
> image priors.
