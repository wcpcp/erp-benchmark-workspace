# H*Bench Unified Protocols

This note defines how `H*Bench` is represented inside the unified benchmark
workspace.

## Why two protocols are kept

The official `H*Bench` setup is not a single-image QA benchmark. It is a
stateful environment:

- one ERP panorama is stored per scene
- the environment renders a perspective crop from the ERP panorama according to
  the current `(yaw, pitch)`
- the model takes one action at a time
- the model eventually calls `submit(yaw,pitch)`

For your project, we keep both:

1. `perspective_multiturn`
   - close to the official benchmark
   - useful for comparing repeated perspective rotations

2. `erp_direct`
   - your ERP-native variant
   - useful for testing whether a model can solve the same task directly from
     the full ERP panorama

## Manifest files

After:

```bash
uv run --project benchmark python benchmark/scripts/prepare_hstar_protocols.py
```

the data directory will expose:

- `benchmark/data/hstar-bench/manifests/perspective_multiturn.jsonl`
- `benchmark/data/hstar-bench/manifests/erp_direct_initial_action.jsonl`
- `benchmark/data/hstar-bench/manifests/erp_direct_submit.jsonl`
- `benchmark/data/hstar-bench/manifests/test.json`

## Protocols

### `perspective_multiturn`

Fields:

- `image_path`: original ERP panorama
- `start_yaw`, `start_pitch`
- `instruction`
- `target_yaw`, `target_pitch`
- `render_config`: perspective renderer config
- `preferred_submit`

This manifest is intended for stateful evaluation pipelines.

### `erp_direct_initial_action`

Fields:

- full ERP input
- current agent direction provided in text
- answer format: one action, usually `rotate(dyaw,dpitch)` or `submit(yaw,pitch)`

This variant measures whether full ERP context lets the model choose a better
first action than a perspective-only setup.

### `erp_direct_submit`

Fields:

- full ERP input
- answer format: `submit(yaw,pitch)`

This variant measures whether the model can directly infer the target direction
from the ERP panorama without iterative perspective search.

## Evaluation

Original `H*Bench`:

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py evaluate \
  --benchmark hstar-bench \
  --predictions /path/to/exported_metrics.jsonl
```

ERP-direct:

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py evaluate \
  --benchmark hstar-bench-erp \
  --references benchmark/data/hstar-bench/manifests/erp_direct_submit.jsonl \
  --predictions /path/to/predictions.jsonl
```

For `erp_direct_initial_action`, rotate predictions are counted as effective if
they reduce angular distance to the target window.

