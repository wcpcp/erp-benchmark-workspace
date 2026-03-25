# H*Bench-ERP Protocol

`H*Bench-ERP` is the ERP-native variant used in this workspace.

## What is kept

- `perspective_multiturn`
  - close to the official H* setup
  - useful when you want to compare repeated perspective rotations

- `erp_rotated_submit`
  - the main ERP-native protocol for this project
  - each sampled initial yaw rotates the ERP input itself
  - the question text stays the same
  - the gold target yaw is rotated into the current ERP image coordinate system
  - the model only outputs the final angles as `(yaw,pitch)`

## Main idea

For ERP input, changing the initial yaw is not just metadata.

If the ERP panorama is horizontally rotated by `delta_yaw`, then the answer must
rotate with it:

- `yaw_rotated = normalize(yaw_original - delta_yaw)`
- `pitch_rotated = pitch_original`

This protocol therefore measures:

- ERP-native final direction prediction
- rotation consistency
- object/path search under full-panorama input

It does not measure intermediate `rotate(...)` policy quality.

## Generated files

After:

```bash
uv run --project benchmark python benchmark/scripts/prepare_benchmarks.py \
  --benchmarks hstar-bench-erp
```

the workspace generates:

- `benchmark/data/hstar-bench-erp/manifests/erp_rotated_submit.jsonl`
- `benchmark/data/hstar-bench-erp/manifests/perspective_multiturn.jsonl`

If you create smoke subsets:

```bash
uv run --project benchmark python benchmark/scripts/create_smoke_subsets.py
```

it also generates:

- `benchmark/data/hstar-bench-erp/manifests/smoke_rotated_submit_20.jsonl`

## Evaluation

The ERP-native protocol only accepts final answers as angles:

```bash
(yaw,pitch)
```

Evaluation checks whether the submitted yaw/pitch falls inside the rotated
target window.

Example:

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py evaluate \
  --benchmark hstar-bench-erp \
  --references benchmark/data/hstar-bench-erp/manifests/erp_rotated_submit.jsonl \
  --predictions /path/to/predictions.jsonl
```
