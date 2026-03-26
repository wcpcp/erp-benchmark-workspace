# H*Bench-ERP Protocol

`H*Bench-ERP` in this workspace is a direct aggregation of the official `hos_bench`
and `hps_bench` benchmark entries.

## What is kept

- one official `annotation.json` item becomes one ERP benchmark sample
- the original task text is kept
- the official yaw window is kept
- the official pitch window is kept when provided

## What is not done

- no rotation expansion
- no synthetic `initial yaw` copies
- no multi-turn `rotate(...)` policy benchmark

This benchmark only asks:

- given the current ERP panorama
- where is the target direction for the official HOS/HPS task?

## Generated files

After:

```bash
uv run --project benchmark python benchmark/scripts/prepare_benchmarks.py \
  --benchmarks hstar-bench-erp
```

the workspace generates:

- `benchmark/data/hstar-bench-erp/manifests/test.jsonl`
- `benchmark/data/hstar-bench-erp/manifests/erp_direct_submit.jsonl`

If you create smoke subsets:

```bash
uv run --project benchmark python benchmark/scripts/create_smoke_subsets.py
```

it also generates:

- `benchmark/data/hstar-bench-erp/manifests/smoke_20.jsonl`

## Prompt format

Each sample uses a direct ERP question such as:

```text
Find the ice cream fridge. Return only the target direction angles in this ERP panorama as (yaw,pitch).
```

`question` and `prompt` are kept identical so the model sees the same text either way.

## Evaluation

Expected output:

```text
(yaw,pitch)
```

The evaluator also accepts:

```text
submit(yaw,pitch)
```

Success means the predicted yaw and pitch fall inside the official target window.
