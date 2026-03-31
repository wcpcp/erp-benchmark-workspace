# ERP Spatial Benchmark

This folder contains the benchmark builder for evaluating whether a model
really understands the omnidirectional 3D space represented by ERP panoramas.

It is intentionally different from the training QA pipeline:

- it uses benchmark-only templates
- it keeps official answers closed-form whenever possible
- it reports scores by core ERP-spatial ability groups
- it exports a review queue for fragile items
- it treats ERP-specific representation understanding as a first-class target

For a paper-ready appendix table of every current task, including question
schema, ground-truth rule, filtering rule, and scoring rule, see:

- [APPENDIX_TASK_TABLE.md](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/APPENDIX_TASK_TABLE.md)

## Core ability groups

The benchmark is organized around four ability groups that directly reflect the
spatial meaning of ERP panoramas.

1. `spherical_localization_and_panoramic_topology`
   - `referring_grounding_bfov`
   - `absolute_direction_mc`
   - `relative_direction_mc`

2. `viewpoint_conditioned_spatial_updating`
   - `camera_rotation_transform_mc`
   - `object_conditioned_reorientation_mc`

3. `observer_centered_3d_layout_understanding`
   - `observer_distance_choice`
   - `relative_3d_position_mc`

4. `erp_representation_understanding`
   - `seam_continuity_mc`
   - `polar_shape_recovery_mc`

## What each group measures

### 1. Spherical localization and panoramic topology

This group asks whether the model has aligned ERP pixels with the spherical
structure they represent.

- `referring_grounding_bfov`: predict the target BFOV directly
- `absolute_direction_mc`: identify the correct absolute ERP sector
- `relative_direction_mc`: judge the angular relation of two targets on the full 360 ring

This is the most basic layer of ERP understanding: knowing where things are on
the sphere and how the panoramic ring is organized.

### 2. Viewpoint-conditioned spatial updating

This group asks whether the model can update spatial relations when the
observer's front direction changes.

- `camera_rotation_transform_mc`: explicit left/right turn by a stated angle
- `object_conditioned_reorientation_mc`: use one object as the new front

This is not the same as ERP rotation consistency. Here the observer reference
frame changes.

### 3. Observer-centered 3D layout understanding

This group asks whether the model can recover observer-centered 3D layout from
ERP direction plus depth-like signals.

- `observer_distance_choice`: among four options, which is nearest to the observer
- `relative_3d_position_mc`: camera-centered multi-axis relation such as left/right, above/below, in front of/behind

This group is intentionally stricter than the training pipeline:
- large BFOV objects are filtered
- the relation is defined from ERP-consistent geometry
- fragile cases still enter the review queue

### 4. ERP representation understanding

This group measures whether the model understands ERP as a representation, not
just as another image.

- `seam_continuity_mc`: seam-aware diagnostic family with five subtypes: cross-boundary nearest neighbor, cross-boundary relative direction, seam dedup counting, structure continuity, and same-entity judgement
- `polar_shape_recovery_mc`: high-latitude distortion should not change the inferred true shape

Current seam subtypes and question templates:

- `nearest_neighbor`
  - `Which listed object is actually nearest to {target_ref} near the {target_side} image edge?`
- `relative_direction`
  - `What is the relation of {neighbor_ref} relative to {target_ref} across the left-right image boundary?`
- `dedup_count`
  - `The left-edge and right-edge visible parts of {target_ref} should be counted as:`
- `structure_continuity`
  - `For the {target_ref} touching both image sides, which explanation is more reasonable?`
  - This subtype is only built for structure-like categories such as walls, tabletops/counters, roads/floors, ceiling lines, and railings.
- `same_entity_judgement`
  - `The left-edge and right-edge appearances of {target_ref} are best described as:`

Rotation consistency is **not** currently included as a scored core task in
this v1 benchmark release.

After review, we treat it as a future paired diagnostic protocol rather than a
standalone QA task, because the standalone versions were too easy to phrase in
a way that leaks the intended equivalence relation.

## Why this differs from the training pipeline

The benchmark should not simply mirror training QA.

To reduce same-distribution leakage, it applies these rules:

- benchmark scenes must not overlap training scenes
- benchmark templates are separate from training templates
- most official items are multiple choice or fixed-label
- large or ambiguous cases are filtered more aggressively
- fragile tasks are queued for manual review

Anchor selection also applies a benchmark-specific exclusion list for very
repetitive, low-distinctiveness categories that often generate poor-quality or
ambiguous referring expressions. Current blocked anchor substrings include:

- `tree`
- `window`
- `leaf`
- `branch`
- `foliage`
- `bush`
- `shrub`
- `plant`

These exclusions currently apply at anchor selection time only.

## Inputs

The builder expects a directory tree containing many `metadata.json` files, for
example:

```text
/path/to/erp_benchmark_metadata/scene_00001/1753781394/metadata.json
```

It can also optionally consume a scene manifest JSONL with fields such as:

- `scene_id`
- `group_id`
- `source_id`
- `domain`
- `split_lock`

`group_id` is useful for describing related scenes and for future stratified
selection if needed.

## Outputs

The script writes:

- `candidate_pool.jsonl`
- `review_queue.jsonl`
- `benchmark_public.jsonl`
- `benchmark_public_prompts.jsonl`
- `benchmark_public_references.jsonl`
- `summary.json`

## How scoring works

Scoring is handled by:

- [evaluate_predictions.py](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/evaluate_predictions.py)

The evaluator accepts predictions with:

- `item_id`
- `prediction`

and compares them against:

- `benchmark_public_references.jsonl`
  or any equivalent answer-bearing benchmark export

Accepted prediction forms:

- option key such as `A` / `B` / `C`
- exact option text
- normalized reference `answer_text`
- for `referring_grounding_bfov`, a BFOV prediction such as `[yaw, pitch, x_fov, y_fov]`

The primary headline metric is now:

- `ability_group_macro_accuracy`

The report also includes:

- `task_macro_score`
- `overall`
- `by_task`
- `by_ability_group`
- `by_diagnostic_slice`

For `referring_grounding_bfov`, the scorer additionally reports:

- seam-aware spherical BFOV IoU
- center angular error

The current correctness threshold for grounding is:

- `BFOV IoU >= 0.5`

## Rotation consistency in the current release

We do **not** currently score rotation consistency as a standalone QA task.

The reason is conceptual rather than implementation-related:

- if the question explicitly states that two ERP images come from the same
  capture point, the task leaks the key relation
- if the question does not state that, many formulations become ambiguous about
  whether the task is scene matching, seam relocation, or viewpoint change
- a cleaner future design is to evaluate rotation consistency as a paired
  protocol over existing tasks such as grounding or relative direction

So in this release:

- seam continuity and polar distortion remain in the ERP-representation group
- rotation consistency is documented as future protocol work, not as an
  official scored task
- the recommended supplementary protocol is described in
  [ROTATION_PROTOCOL.md](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/ROTATION_PROTOCOL.md)
- a runnable builder/evaluator is provided in
  [rotation_protocol.py](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/rotation_protocol.py)

Example:

```bash
python3 benchmark/erp_spatial_benchmark/evaluate_predictions.py \
  --predictions /path/to/predictions.jsonl \
  --references /path/to/erp_spatial_benchmark_out/benchmark_public_references.jsonl \
  --report /path/to/report.json
```

Supplementary yaw-shift protocol:

```bash
python3 benchmark/erp_spatial_benchmark/rotation_protocol.py build \
  --references /path/to/erp_spatial_benchmark_out/benchmark_public_references.jsonl \
  --output-dir /path/to/erp_rotation_protocol_out
```

```bash
python3 benchmark/erp_spatial_benchmark/rotation_protocol.py evaluate \
  --base-predictions /path/to/base_predictions.jsonl \
  --base-references /path/to/erp_spatial_benchmark_out/benchmark_public_references.jsonl \
  --shifted-predictions /path/to/shifted_predictions.jsonl \
  --shifted-references /path/to/erp_rotation_protocol_out/rotation_protocol_references.jsonl \
  --report /path/to/rotation_report.json
```

## Run

```bash
python3 benchmark/erp_spatial_benchmark/build_benchmark.py \
  --input-root /path/to/erp_benchmark_metadata \
  --output-dir /path/to/erp_spatial_benchmark_out \
  --target-public-per-task 250 \
  --seed 20260327
```

### What the main arguments mean

- `--input-root`
  - A directory containing many `metadata.json` files, or a single `metadata.json` file for local smoke testing.
- `--output-dir`
  - Output directory for all benchmark artifacts, including candidate pools, public prompts, public references, and summary files.
- `--target-public-per-task`
  - Target number of selected public benchmark items per task, after candidate generation and per-scene caps.
  - A practical default is `250`.
- `--max-per-scene-per-task`
  - Maximum number of selected items contributed by one scene to one task.
  - Default `1` prevents a single scene from dominating a task.
- `--seed`
  - Random seed controlling deterministic benchmark item selection.
- `--scene-manifest`
  - Optional JSONL manifest with fields such as `scene_id`, `group_id`, `source_id`, `domain`, and `split_lock`.
  - In the current public-only release, this mainly preserves provenance metadata and supports future stratified selection if needed.

With an optional manifest:

```bash
python3 benchmark/erp_spatial_benchmark/build_benchmark.py \
  --input-root /path/to/erp_benchmark_metadata \
  --output-dir /path/to/erp_spatial_benchmark_out \
  --scene-manifest /path/to/scene_manifest.jsonl \
  --target-public-per-task 250 \
  --seed 20260327
```

### Local single-metadata smoke test

For quick local verification, you can point `--input-root` directly at one
`metadata.json` file instead of a whole benchmark root.

Example:

```bash
python3 benchmark/erp_spatial_benchmark/build_benchmark.py \
  --input-root /Users/wcp/code/erp_data_pipeline/data_generation/dataset/metadata.json \
  --output-dir /tmp/erp_benchmark_smoke \
  --seed 20260327
```

This is useful for:

- checking that the builder runs end to end
- inspecting generated question schemas locally
- validating scorer compatibility before running the full benchmark build

For a single-scene smoke test, the most convenient files to inspect are usually:

- `benchmark_public.jsonl`
- `benchmark_public_prompts.jsonl`
- `benchmark_public_references.jsonl`
- `summary.json`

## Recommended release workflow

1. Build the full candidate pool.
2. Inspect `review_queue.jsonl`.
3. Manually verify all review-required items.
4. Freeze:
   - `benchmark_public.jsonl`
   - `benchmark_public_prompts.jsonl`
   - `benchmark_public_references.jsonl`
