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

- `observer_distance_choice`: among three or four options, which is nearest to the observer
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
- `polar_shape_matching_mc`: match a high-latitude distorted target to another object with the same true geometry
- `polar_cross_latitude_matching_mc`: match a high-latitude distorted target to a lower-latitude object with the same true geometry

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
  - `In the 360 scene, the left-edge and right-edge appearances of {target_ref} are best described as:`

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

The broader entity-level benchmark filter also rejects highly repetitive
categories such as:

- `tree`
- `window`
- `grass`
- `sky`
- `cloud`

These exclusions are now applied both at anchor selection time and at the
general entity eligibility level.

Entity reliability filtering now prioritizes detection and reground quality,
not VLM self-reported semantic confidence. The current hard gate is:

- `best_score >= 0.65`
- `local_reground.pred_score >= 0.65`
- valid BFOV
- non-trivial area

For direction-sensitive tasks (`absolute_direction_mc`,
`relative_direction_mc`, `camera_rotation_transform_mc`,
`object_conditioned_reorientation_mc`), the builder also filters out large
targets to avoid multi-sector ambiguity:

- `x_fov <= 35°`
- `y_fov <= 30°`
- `area_ratio <= 0.08`

In addition, sector and relation items require larger angular clearance than
before. The effective margin is now measured after accounting for target BFOV,
so boundary cases such as `right` vs `back-right` are filtered more
aggressively.

For `absolute_direction_mc`, we now intentionally prefer panorama-native
challenge cases instead of perspective-like easy cases:

- the public benchmark target mix is:
  - about `10%` `right`
  - `back-right`
  - `back`
  - `back-left`
- about `10%` `left`
- the effective sector margin must be at least `15°`
- immediate neighboring distractors such as `left` versus `front-left` are no
  longer used
- distractors are drawn from more clearly separated sectors so the task tests
  true 360-degree orientation understanding rather than simple image-left /
  image-right heuristics
- if natural candidates are insufficient, the builder may synthesize additional
  absolute-direction stress items by yaw-rotating high-quality targets into
  this target sector mix

When a scene contains multiple similar instances of the same category, the
builder now disambiguates only when needed:

- it first checks whether the natural referring phrases are actually too similar
- for mild duplicate cases, it adds a light natural cue such as
  `near the right side`
- for heavier duplicate cases, it prefers natural disambiguators such as
  left/right or upper/lower wording
- if a duplicate-heavy case still cannot be described cleanly without leaking
  too much positional information, the item is filtered out instead of forcing
  explicit coordinates into the question text

This is only used for duplicate-heavy cases; it is not added to every item.

For most tasks, benchmark prompts continue to use entity-level referring
expressions such as `reground_query` or `caption_brief`, because those richer
references are useful for measuring language-object understanding and do not
normally leak the answer.

High-latitude distortion tasks now also use natural referring expressions. The
builder starts from the same `reground_query` / `caption_brief` path as other
tasks, then strips both direct canonical shape words and the entity's own raw
`semantic.attributes["shape"]` terms from the final reference. This keeps the
target description natural and semantically rich without copying the exact
geometry answer into the question text.

For derived yaw/pitch rotations, the builder also rewrites the main
geometry-bearing metadata so derived scenes remain internally consistent for
benchmark construction. Updated fields include:

- `lon_lat`
- `bfov.yaw_deg`
- `bfov.pitch_deg`
- `entity_bfov`
- `bbox_erp`
- `entity_xyz_camera`
- `spatial.yaw_deg`
- `spatial.pitch_deg`
- `spatial.xyz_camera_m`
- `seam_crossing_flag`
- `pole_proximity_flag`

Depth-related scalars such as `entity_center_depth` are preserved because a
rigid camera rotation changes viewpoint but not distance. Detector/reground
scores are also preserved. Pixel-space masks are cleared in derived scenes
because the original mask is no longer aligned after spherical reprojection.

## High-Quality Filtering Strategy

This benchmark is intentionally quality-first rather than quantity-first. The
builder does not force every task to reach its nominal target count. Instead,
it first constructs a candidate pool under aggressive filtering, and only then
selects the final public benchmark from the surviving candidates.

### Shared filtering rules

These rules apply broadly across the benchmark before task-specific logic runs.

- **Metadata validity**
  - Empty or invalid `metadata.json` files are skipped and logged to
    `skipped_invalid_metadata.jsonl`.
- **Entity reliability**
  - `best_score >= 0.65`
  - `local_reground.pred_score >= 0.65`
  - valid BFOV must exist
  - area must be non-trivial
- **Low-value semantic category filtering**
  - Anchor-level blocked substrings:
    - `tree`
    - `window`
    - `leaf`
    - `branch`
    - `foliage`
    - `bush`
    - `shrub`
    - `plant`
  - Broader entity-level blocked substrings:
    - `tree`
    - `window`
    - `leaf`
    - `branch`
    - `foliage`
    - `bush`
    - `shrub`
    - `plant`
    - `grass`
    - `sky`
    - `cloud`
- **Duplicate-instance disambiguation**
  - If a scene contains multiple instances of the same label:
    - mild duplicate cases receive a natural side cue such as
      `near the right side`
    - heavier duplicate cases additionally append a compact locator derived
      from normalized box coordinates or BFOV
  - This is only applied when needed; references are not made artificially
    verbose for every item.
- **Direction-task large-object filtering**
  - For direction-sensitive tasks
    (`absolute_direction_mc`, `relative_direction_mc`,
    `camera_rotation_transform_mc`, `object_conditioned_reorientation_mc`):
    - `x_fov <= 35°`
    - `y_fov <= 30°`
    - `area_ratio <= 0.08`
- **Boundary ambiguity suppression**
  - Directional tasks use BFOV-aware effective angular margins so that targets
    near sector boundaries such as `right` vs `back-right` are dropped instead
    of kept as ambiguous items.
- **Manual review routing**
  - Fragile tasks are exported to `review_queue.jsonl` for additional checking:
    - `relative_3d_position_mc`
    - `seam_continuity_mc`
    - `polar_shape_recovery_mc`
    - `polar_shape_matching_mc`
    - `polar_cross_latitude_matching_mc`

### Task-specific filtering rules

#### `referring_grounding_bfov`

- Requires a valid target BFOV.
- No large additional task-specific semantic filtering beyond the shared
  entity-quality checks.

#### `absolute_direction_mc`

- Target must pass the shared direction-task size filter.
- The target sector mix is approximately:
  - `10%` `right`
  - `80%` distributed across `back-right`, `back`, and `back-left`
  - `10%` `left`
- The effective sector margin must be at least `15°` after accounting for the
  target BFOV width.
- Immediate neighboring sectors are excluded from distractors so linguistically
  plausible but low-value alternatives such as `left` versus `front-left` do
  not appear together.
- If the natural candidate pool is underfilled, derived yaw rotations are used
  to supplement this target sector mix.
- These derived absolute-direction candidates are drawn from the broader pool
  of high-quality, resolvable entities rather than only the top anchor subset,
  so conservative anchor selection does not accidentally starve the task.

#### `relative_direction_mc`

- Both target and reference must pass the shared direction-task size filter.
- `abs(delta_yaw)` must be large enough to define a clear relation.
- Effective margin to relation boundaries must be at least `15°` after BFOV
  clearance using the larger target/reference horizontal extent.

#### `camera_rotation_transform_mc`

- Target must pass the shared direction-task size filter.
- After applying the observer turn, the target must still land in a stable
  relation bin.
- Effective post-rotation margin must be at least `15°`.

#### `object_conditioned_reorientation_mc`

- Both the facing object and the target must pass the shared direction-task
  size filter.
- The reoriented relation must remain stable after BFOV-aware clearance.
- Effective margin must be at least `15°`.

#### `observer_distance_choice`

- At least `3` depth-valid candidate entities are required.
- The builder takes the nearest `6` by depth, then uses the first `4` when
  available, otherwise the first `3`.
- Adjacent depths among the selected set must differ by at least `0.35m`.

#### `relative_3d_position_mc`

- Both entities must be compact:
  - `x_fov <= 35°`
  - `y_fov <= 35°`
  - `x_fov * y_fov <= 1200`
- Large structure-like categories are filtered out entirely for this task:
  - `building`
  - `wall`
  - `ceiling`
  - `roof`
  - `floor`
  - `ground`
  - `road`
  - `sidewalk`
  - `facade`
  - `fence`
  - `railing`
  - `gate`
- A relation is only kept if the geometry resolves to a clean 1-axis or 2-axis
  camera-centered relation.
- If no axis is sufficiently clear, or more than two axes become active, the
  sample is dropped.

#### `seam_continuity_mc`

- Only boundary-valid seam cases are kept.
- The five seam subtypes use different structural conditions:
  - `nearest_neighbor`
    - requires a boundary-touching target
    - requires a correct opposite-boundary neighbor with small wrap gap
    - requires a flat-image lure and additional distractors
  - `relative_direction`
    - requires a valid close wrap-around pair
  - `dedup_count`
    - requires `seam_crossing_flag = true`
  - `structure_continuity`
    - requires `seam_crossing_flag = true`
    - further restricted to structure-like targets only
  - `same_entity_judgement`
    - requires `seam_crossing_flag = true`
- If natural seam samples are too scarce, the builder synthesizes seam stress
  items through yaw-rotated ERP scenes.
- Derived seam samples may reuse the same source scene, but they must come from
  different target entities. The builder will not generate multiple seam
  variants for the same entity.

#### `polar_shape_recovery_mc`

- Requires a valid canonical geometric `shape` label such as:
  - `round`
  - `rectangular`
  - `square`
  - `oval`
  - `cylindrical`
  - `spherical`
  - `triangular`
  - `arched`
- Non-geometric subtype labels such as vehicle/body-style terms are rejected.
- Natural items require either:
  - `abs(lat) >= 60°`
  - or `infer_pole_proximity(entity)`
- This task uses the target's natural referring expression
  (`reground_query` / `caption_brief`) together with the standard duplicate
  disambiguation logic used elsewhere in the benchmark.
- Direct canonical shape words and the entity's raw shape-attribute terms are
  stripped from that reference before the question is rendered, so the prompt
  remains descriptive without spelling out the answer.
- Distractors prefer geometry-near alternatives rather than arbitrary fallback
  words.
- If natural polar cases are too scarce, the builder synthesizes polar stress
  items through pitch-rotated ERP scenes.
- Derived polar targeting is intentionally hard:
  - target latitude band: stable-random within roughly `75°-85°`
  - candidate pitch rotations: searched over non-zero pitch shifts until the
    rotated target lands inside that band
- Derived polar samples may reuse the same source scene, but they must come
  from different target entities. The builder will not generate multiple polar
  variants for the same entity.

#### `polar_shape_matching_mc`

- Requires the same high-latitude target eligibility as `polar_shape_recovery_mc`.
- The target still uses the cleaned natural referring expression path:
  `reground_query` / `caption_brief` with shape-leak terms removed.
- Options are other scene objects, not shape labels.
- The correct option must be another resolvable entity whose canonical
  geometric shape matches the target.
- Distractors must come from different canonical shape classes, with preference
  for geometry-near alternatives when available.
- This task is especially useful when explicit shape-label recovery is too
  sparse but same-shape object pairs are still available.

#### `polar_cross_latitude_matching_mc`

- Uses the same high-latitude target definition as the other polar tasks.
- The correct option must come from a lower-distortion latitude band:
  `abs(lat) <= 35°`.
- Distractors are also drawn from the lower-latitude pool, but must come from
  different canonical shape classes.
- This task measures whether the model can match a strongly distorted polar
  target to a more canonical lower-latitude object without relying on an
  explicit shape label.

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
- `derived_metadata/*.json` when derived stress items are synthesized

For all multiple-choice tasks, option order is stable-shuffled per item. This
prevents global answer-position bias such as many items always having answer
`A`, while keeping rebuilds deterministic for the same data and code version.

If natural absolute-direction, seam, or polar cases are too scarce, the builder now automatically
creates derived stress samples:

- absolute-direction stress: yaw-shifted ERP panoramas that move strong targets
  into the challenge sectors used by `absolute_direction_mc`
- seam stress: yaw-shifted ERP panoramas that move strong targets toward the
  left/right seam
- polar stress: pitch-rotated ERP panoramas that move strong targets toward
  high latitude

To keep the derived set from becoming too synthetic, the current builder uses
multiple derived seam/polar items only when they come from different target
entities. Polar targeting is intentionally hard: it uses a stable-random
target latitude in the `75°-85°` band and searches non-zero pitch rotations
until the chosen target lands there.

The rotated ERP image is written next to the original image file, while the
corresponding transformed metadata is exported under `derived_metadata/` in the
benchmark output directory. Derived items are tagged with the
`derived_rotation` diagnostic slice and retain a pointer to the source scene.
For derived scenes, geometry-bearing metadata such as `lon_lat`, `bfov`,
`entity_bfov`, `bbox_erp`, `entity_xyz_camera`, and `spatial.xyz_camera_m` are
rewritten into the rotated camera frame; `bbox_erp` is re-estimated from
sampled box points after spherical reprojection rather than by simple center
translation. Derived metadata also preserves explicit `erp_width` /
`erp_height`, so seam checks and normalized `0-1000` boxes remain valid after
rotation. `summary.json` now reports every benchmark task explicitly,
including tasks whose current candidate count is `0`, and separates original
scene count (`num_scenes`) from the number of generated derived scenes
(`derived_scene_count`).

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

Run from the repository root. The builder is self-contained inside this repo and
does not require a sibling `data_generation/` checkout.

```bash
python3 erp_spatial_benchmark/build_benchmark.py \
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
  - Target number of selected public benchmark items per task, after candidate generation and optional per-scene caps.
  - A practical default is `250`.
- `--max-per-scene-per-task`
  - Maximum number of selected items contributed by one scene to one task.
  - Default `0` means no per-scene cap.
  - Set this to a positive integer only if you explicitly want to stop a few dense scenes from dominating one task.
- `--seed`
  - Random seed controlling deterministic benchmark item selection.
- `--scene-manifest`
  - Optional JSONL manifest with fields such as `scene_id`, `group_id`, `source_id`, `domain`, and `split_lock`.
  - In the current public-only release, this mainly preserves provenance metadata and supports future stratified selection if needed.

With an optional manifest:

```bash
python3 erp_spatial_benchmark/build_benchmark.py \
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
python3 erp_spatial_benchmark/build_benchmark.py \
  --input-root /path/to/metadata.json \
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

## Prune released items

If you manually review rendered benchmark images and decide that some public
items should be removed, use:

- [scripts/prune_benchmark_jsonl.py](/Users/wcp/code/erp_data_pipeline/benchmark/scripts/prune_benchmark_jsonl.py)

This script supports the two delete-list patterns used in our review workflow:

- an exact rendered filename such as `.../foo.jpg`
  - removes the single matching benchmark item whose `item_id` corresponds to that rendered image
- a prefix ending in `*` such as `.../source_scene_id*`
  - removes all benchmark items derived from that source prefix, including rotated or task-specific variants

Example `delete.txt` entries:

```text
/workspace/.../images/panoramax__0b046ad7-bec7-4ec3-ab56-8a4ff66befca__seam_yaw_-21_E000027_seam_continuity_mc_dedup_count_E000027.jpg
/workspace/.../images/openverse__5ddee573-0ad7-4dd4-b0e7-0100438e5259*
```

Recommended usage is to prune the public benchmark, prompts, and references
together so the three JSONL files remain aligned:

```bash
python3 scripts/prune_benchmark_jsonl.py \
  --jsonl \
  /path/to/erp_spatial_benchmark_out/benchmark_public.jsonl \
  /path/to/erp_spatial_benchmark_out/benchmark_public_prompts.jsonl \
  /path/to/erp_spatial_benchmark_out/benchmark_public_references.jsonl \
  --delete-txt /path/to/erp_spatial_benchmark_out/delete.txt \
  --in-place \
  --backup
```

Behavior:

- without `--in-place`, the script writes sibling `*.filtered.jsonl` files
- with `--in-place`, the input JSONL files are overwritten
- with `--backup`, each overwritten JSONL is first copied to `*.bak`

The script prints a JSON summary showing:

- how many delete rules were loaded
- how many rows were removed from each JSONL file
- example removed `item_id` values and the delete rule that matched them

## Recommended release workflow

1. Build the full candidate pool.
2. Inspect `review_queue.jsonl`.
3. Manually verify all review-required items.
4. Freeze:
   - `benchmark_public.jsonl`
   - `benchmark_public_prompts.jsonl`
   - `benchmark_public_references.jsonl`

## Candidate generation policy

The current builder does not treat one scene as “one question” or “one
ability”. A single scene can contribute many benchmark items across many tasks.

- Anchor coverage
  - All entities that pass the anchor-pool filter are considered as anchor candidates.
  - The anchor pool is no longer truncated to a small top-k set during scene-level question generation.
  - Shared anchor-pool eligibility requires:
    - `best_score >= 0.65`
    - `local_reground.pred_score >= 0.65`
    - a valid resolved BFOV
    - no match to the global low-value blocked labels such as `tree` or `window`

- Per-scene selection
  - During candidate generation, one scene can emit multiple items for the same task and many items across different tasks.
  - During final public selection, `--max-per-scene-per-task 0` keeps this unconstrained by default.
  - If a positive cap is supplied, it is applied per scene per task, not per scene overall.

- Answer-key balance
  - All multiple-choice tasks use deterministic option shuffling.
  - The final public benchmark then applies an additional deterministic answer-key rebalance pass so that `A/B/C/D/E` are not heavily skewed toward one letter inside a task.
