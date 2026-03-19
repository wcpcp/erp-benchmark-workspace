# ERP Foundation Benchmark

This directory defines a benchmark-only evaluation suite for an ERP-native
vision-language foundation model.

It is intentionally different from the training-data project:

- benchmark samples should be fewer but much stricter
- answers should be uniquely verifiable
- most official scores should use closed-form outputs
- ERP-specific stress slices should be explicit
- hidden-test discipline matters more than sample count

## Goal

The benchmark should answer one question:

Can a model that takes ERP panoramas directly build the right aligned
representation for language, omnidirectional reasoning, and 3D/search-oriented
understanding?

## What should be measured

The official benchmark should prioritize three ability groups:

1. `language_erp_alignment`
   - object semantics
   - referring expressions
   - grounding
   - existence and counting
   - scene composition

2. `erp_native_spatial_understanding`
   - absolute omnidirectional direction
   - relative direction
   - seam continuity
   - rotation consistency
   - polar distortion robustness

3. `spatial_3d_and_search_preconditions`
   - depth ordering
   - distance bucket estimation
   - coarse relative 3D position
   - object-search initial turning decision
   - path-search initial action

## Benchmark structure

The recommended structure is:

- `core_v1`
  - 13 official tasks
  - 500 hidden-test items per task
  - 6,500 scored test samples

- `advanced_extension`
  - 3 harder tasks that require stronger geometry or search metadata
  - 500 hidden-test items per task
  - 1,500 extension samples

- `diagnostic_slices`
  - seam
  - pole
  - rotation
  - same-class distractor
  - low-margin depth

This gives:

- `6,500` core official test items
- `1,500` extension items
- `8,000` total benchmark items if the full suite is released

## Why not use a generic VQA benchmark design

This benchmark is not meant to measure generic caption quality. It should
measure whether ERP-specific world understanding emerges.

Therefore:

- free-form long answers should be auxiliary only
- core official answers should be structured or closed-form
- ERP-specific tasks should require the panorama, not just a local crop
- seam, pole, and rotation slices should be reported separately

## Files

- [`docs/design_principles.md`](/Users/wcp/code/erp_data_pipeline/benchmark/custom_erp_foundation_benchmark/docs/design_principles.md)
  benchmark design principles and why they fit ERP evaluation
- [`config/task_blueprint.json`](/Users/wcp/code/erp_data_pipeline/benchmark/custom_erp_foundation_benchmark/config/task_blueprint.json)
  task definitions, quotas, metadata tiers, and answer formats
- [`config/benchmark_protocol.json`](/Users/wcp/code/erp_data_pipeline/benchmark/custom_erp_foundation_benchmark/config/benchmark_protocol.json)
  release protocol, balancing rules, and acceptance criteria
- [`config/question_templates.json`](/Users/wcp/code/erp_data_pipeline/benchmark/custom_erp_foundation_benchmark/config/question_templates.json)
  benchmark question templates for each task
- [`config/filter_rules.json`](/Users/wcp/code/erp_data_pipeline/benchmark/custom_erp_foundation_benchmark/config/filter_rules.json)
  concrete thresholds for candidate filtering and bucketization
- [`docs/task_generation_rules.md`](/Users/wcp/code/erp_data_pipeline/benchmark/custom_erp_foundation_benchmark/docs/task_generation_rules.md)
  concrete generation and filtering rules for each task
- [`docs/annotation_guidelines.md`](/Users/wcp/code/erp_data_pipeline/benchmark/custom_erp_foundation_benchmark/docs/annotation_guidelines.md)
  manual verification checklist for official benchmark items
- [`templates/benchmark_item_example.json`](/Users/wcp/code/erp_data_pipeline/benchmark/custom_erp_foundation_benchmark/templates/benchmark_item_example.json)
  example benchmark item schema
- [`scripts/generate_candidates.py`](/Users/wcp/code/erp_data_pipeline/benchmark/custom_erp_foundation_benchmark/scripts/generate_candidates.py)
  generate candidate benchmark items from one scene metadata file
- [`scripts/assemble_pool.py`](/Users/wcp/code/erp_data_pipeline/benchmark/custom_erp_foundation_benchmark/scripts/assemble_pool.py)
  assemble a benchmark pool from multiple candidate files with per-task quotas

## Minimal production flow

Generate candidate items scene by scene:

```bash
python3 benchmark/custom_erp_foundation_benchmark/scripts/generate_candidates.py \
  --input /path/to/scene_metadata.json \
  --output /path/to/candidates/scene_a.jsonl
```

Build a quota-controlled pool:

```bash
python3 benchmark/custom_erp_foundation_benchmark/scripts/assemble_pool.py \
  --input-dir /path/to/candidates \
  --output /path/to/core_v1_pool.jsonl \
  --phase core_v1 \
  --target-per-task 500
```

For small internal dry runs, start with:

```bash
python3 benchmark/custom_erp_foundation_benchmark/scripts/assemble_pool.py \
  --input-dir /path/to/candidates \
  --output /path/to/core_v1_pool_lite.jsonl \
  --phase core_v1 \
  --target-per-task 50
```

## Recommended release policy

For an internal or public benchmark, use:

- hidden official test answers
- optional small public dev subset
- scene-level holdout from all training and SFT data
- manual verification on every official item

## Minimum corpus guidance

For the full `500 per task` target, a healthy source corpus is usually:

- `>= 600` scenes if scene overlap is controlled tightly
- `>= 800` scenes if you want strong diversity and lower scene repetition

If the corpus is smaller, start with:

- `300 per task` for `core_v1`

and scale to `500 per task` after the metadata coverage and verification
pipeline stabilize.
