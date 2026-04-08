# Benchmark Workspace

这个 `benchmark/` 仓库同时做两件事：

1. **统一准备、预测、评测公开 benchmark**
2. **构建并评测我们自己的 ERP 空间 benchmark**

所以这里不是只有自建 benchmark，也不是只有公开 benchmark 适配层，而是两条工作线共用一个仓库。

## 目录概览

### 公开 benchmark 统一评测层

- [src/erp_benchmarks](/Users/wcp/code/erp_data_pipeline/benchmark/src/erp_benchmarks)
- [registry.yaml](/Users/wcp/code/erp_data_pipeline/benchmark/registry.yaml)
- [scripts/prepare_benchmarks.py](/Users/wcp/code/erp_data_pipeline/benchmark/scripts/prepare_benchmarks.py)
- [scripts/predict_benchmark.py](/Users/wcp/code/erp_data_pipeline/benchmark/scripts/predict_benchmark.py)
- [scripts/run_benchmark.py](/Users/wcp/code/erp_data_pipeline/benchmark/scripts/run_benchmark.py)

### 自建 benchmark 层

- [erp_spatial_benchmark](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark)
  - 当前正式版自建 ERP benchmark
- [custom_erp_foundation_benchmark](/Users/wcp/code/erp_data_pipeline/benchmark/custom_erp_foundation_benchmark)
  - 早期草案，不再作为正式主入口

## 当前支持的公开 benchmark

当前 registry 和评测层里可统一管理的 benchmark 包括：

- `osr-bench`
- `panoenv`
- `omnispatial`
- `hstar-bench-erp`
- `hstar-bench`
- `360loc`
- `habitat-nav`

其中：

- **支持统一 `predict + evaluate` 的**
  - `osr-bench`
  - `panoenv`
  - `omnispatial`
  - `hstar-bench-erp`

- **当前主要支持 `evaluate` 的**
  - `hstar-bench`
  - `360loc`
  - `habitat-nav`

## 三步工作流

无论是公开 benchmark 还是自建 benchmark，建议都按这三个阶段理解：

1. **prepare**
   - 下载或整理原始数据
   - 生成 manifest / references
2. **predict**
   - 用模型读取 benchmark 输入
   - 输出 `predictions.jsonl`
3. **evaluate**
   - 用参考答案对预测打分

`evaluate` **不会**替你跑模型，它只负责读预测文件并评分。

---

## 公开 benchmark：准备数据

### 1. 默认准备方式

```bash
python3 scripts/prepare_benchmarks.py \
  --benchmarks osr-bench,panoenv,omnispatial,hstar-bench-erp
```

这会：

- 准备原始数据
- 生成各 benchmark 的本地 manifest

### 2. 复用你已经下载好的 raw 数据

如果服务器上 raw 数据已经在别的目录，使用：

```bash
python3 scripts/prepare_benchmarks.py \
  --benchmarks osr-bench,panoenv,omnispatial,hstar-bench-erp \
  --raw-dir osr-bench=/abs/path/to/OSR-Bench \
  --raw-dir panoenv=/abs/path/to/PanoEnv \
  --raw-dir omnispatial=/abs/path/to/OmniSpatial \
  --raw-dir hstar-bench-erp=/abs/path/to/thinking_in_360
```

脚本会把这些外部 raw 目录软链接到当前仓库的标准 `data/<benchmark>/raw/` 位置，再生成 manifest。

### 3. 只准备某一个 benchmark

```bash
python3 scripts/prepare_benchmarks.py --benchmarks osr-bench
python3 scripts/prepare_benchmarks.py --benchmarks panoenv
python3 scripts/prepare_benchmarks.py --benchmarks omnispatial
python3 scripts/prepare_benchmarks.py --benchmarks hstar-bench-erp
```

### 4. 先看支持的 benchmark

```bash
python3 scripts/run_benchmark.py list
```

查看某一个 benchmark 的 registry 信息：

```bash
python3 scripts/run_benchmark.py describe osr-bench
```

---

## 公开 benchmark：生成预测

统一预测脚本是：

- [scripts/predict_benchmark.py](/Users/wcp/code/erp_data_pipeline/benchmark/scripts/predict_benchmark.py)

它当前支持：

- `osr-bench`
- `panoenv`
- `omnispatial`
- `hstar-bench-erp`

### 1. 先用 mock 跑通

```bash
python3 scripts/predict_benchmark.py \
  --benchmark osr-bench \
  --model mock \
  --references data/osr-bench/manifests/smoke_20.jsonl \
  --predictions-out results/osr_predictions_smoke.jsonl \
  --skip-download
```

### 2. 用 Transformers 跑本地/HF 模型

例子：`PanoEnv + Qwen3-VL-4B`

```bash
python3 scripts/predict_benchmark.py \
  --benchmark panoenv \
  --model transformers-vlm \
  --model-path /path/to/Qwen3-vl-4B \
  --references data/panoenv/manifests/test.jsonl \
  --predictions-out results/panoenv_predictions_qwen3vl4b.jsonl \
  --torch-dtype bfloat16 \
  --device-map auto \
  --skip-download
```

### 3. 用 vLLM OpenAI 兼容接口

```bash
python3 scripts/predict_benchmark.py \
  --benchmark osr-bench \
  --model vllm-openai \
  --model-name Qwen/Qwen3-VL-4B-Instruct \
  --api-base http://127.0.0.1:8000/v1 \
  --references data/osr-bench/manifests/smoke_20.jsonl \
  --predictions-out results/osr_predictions_vllm.jsonl \
  --skip-download
```

### 4. 用 OpenAI 风格 API

```bash
python3 scripts/predict_benchmark.py \
  --benchmark omnispatial \
  --model openai-api \
  --model-name gpt-4.1 \
  --references data/omnispatial/manifests/smoke_20.jsonl \
  --predictions-out results/omnispatial_predictions_openai.jsonl \
  --skip-download
```

### 5. 最常用参数

- `--benchmark`
- `--model`
- `--model-path`
- `--model-name`
- `--api-base`
- `--references`
- `--predictions-out`
- `--skip-download`
- `--limit`

如果不传 `--references`，脚本会默认使用该 benchmark 本地 `test` manifest。

---

## 公开 benchmark：评测打分

统一评测脚本是：

- [scripts/run_benchmark.py](/Users/wcp/code/erp_data_pipeline/benchmark/scripts/run_benchmark.py)

### 1. 评测 OSR-Bench

```bash
python3 scripts/run_benchmark.py evaluate \
  --benchmark osr-bench \
  --references data/osr-bench/manifests/test.jsonl \
  --predictions results/osr_predictions.jsonl
```

### 2. 评测 PanoEnv

```bash
python3 scripts/run_benchmark.py evaluate \
  --benchmark panoenv \
  --references data/panoenv/manifests/test.jsonl \
  --predictions results/panoenv_predictions.jsonl
```

### 3. 评测 OmniSpatial

```bash
python3 scripts/run_benchmark.py evaluate \
  --benchmark omnispatial \
  --references data/omnispatial/manifests/test.jsonl \
  --predictions results/omnispatial_predictions.jsonl
```

### 4. 评测 H*Bench-ERP

```bash
python3 scripts/run_benchmark.py evaluate \
  --benchmark hstar-bench-erp \
  --references data/hstar-bench-erp/manifests/test.jsonl \
  --predictions results/hstar_erp_predictions.jsonl
```

### 5. 评测 H*Bench 原始版

```bash
python3 scripts/run_benchmark.py evaluate \
  --benchmark hstar-bench \
  --predictions /path/to/exported_metrics_or_predictions.jsonl
```

### 6. 评测 360Loc

```bash
python3 scripts/run_benchmark.py evaluate \
  --benchmark 360loc \
  --predictions /path/to/360loc_predictions.jsonl \
  --coordinate-system cartesian
```

### 7. 评测 Habitat-Nav

```bash
python3 scripts/run_benchmark.py evaluate \
  --benchmark habitat-nav \
  --predictions /path/to/habitat_nav_metrics.jsonl
```

---

## 公开 benchmark：最小 smoke 流程

如果你只是想确认链路跑通，推荐：

### OSR-Bench smoke

生成预测：

```bash
python3 scripts/predict_benchmark.py \
  --benchmark osr-bench \
  --model mock \
  --references data/osr-bench/manifests/smoke_20.jsonl \
  --predictions-out results/osr_predictions_smoke.jsonl \
  --skip-download
```

评测：

```bash
python3 scripts/run_benchmark.py evaluate \
  --benchmark osr-bench \
  --references data/osr-bench/manifests/smoke_20.jsonl \
  --predictions results/osr_predictions_smoke.jsonl
```

---

## 我们自己的 ERP benchmark

当前正式版在：

- [erp_spatial_benchmark](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark)

它不是训练集，而是一个质量优先的、ERP-native 的空间 benchmark builder。

### 当前测什么

四类核心能力：

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

### 构建命令

```bash
python3 erp_spatial_benchmark/build_benchmark.py \
  --input-root /path/to/erp_benchmark_metadata_root \
  --output-dir /path/to/erp_spatial_benchmark_out \
  --target-public-per-task 250 \
  --seed 20260327
```

### 输出文件

最重要的文件是：

- `benchmark_public_prompts.jsonl`
  - 给模型做题
- `benchmark_public_references.jsonl`
  - 给评测器判分
- `summary.json`
  - 看每个 task 最终有多少题

中间文件：

- `candidate_pool.jsonl`
- `review_queue.jsonl`
- `derived_metadata/`
- `skipped_invalid_metadata.jsonl`

### 如何删除人工判坏的 public items

如果你在可视化复核后整理出一个 `delete.txt`，可以用：

- [scripts/prune_benchmark_jsonl.py](/Users/wcp/code/erp_data_pipeline/benchmark/scripts/prune_benchmark_jsonl.py)

推荐一起同步裁剪：

- `benchmark_public.jsonl`
- `benchmark_public_prompts.jsonl`
- `benchmark_public_references.jsonl`

示例：

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

这个脚本支持两种 delete 规则：

- 精确 `.jpg` 文件名：删除单条 item
- 以 `*` 结尾的源图前缀：删除这一整组相关 derived/task item

### 如何拿它做推理

你的模型应读取：

- `image_path`
- `question`
- `options`（如果是多选题）

然后输出：

```json
{"item_id": "...", "prediction": "A"}
```

对于 `referring_grounding_bfov`，输出：

```json
{"item_id": "...", "prediction": "[yaw, pitch, x_fov, y_fov]"}
```

### 如何评测

```bash
python3 erp_spatial_benchmark/evaluate_predictions.py \
  --predictions /path/to/predictions.jsonl \
  --references /path/to/erp_spatial_benchmark_out/benchmark_public_references.jsonl \
  --report /path/to/report.json
```

### 质量筛选策略

这套自建 benchmark 的高质量筛选策略已经单独整理在：

- [erp_spatial_benchmark/README.md](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/README.md)

里面有：

- 公共筛选规则
- 每个任务的特殊过滤规则
- seam / polar 的 derived stress 生成策略

---

## 自建 benchmark 与训练集的区别

训练集是为了“教会模型”，benchmark 是为了“确认模型到底学会了什么”。

所以 benchmark 当前默认是：

- benchmark-only templates
- 质量优先
- 闭集/固定标签优先
- 更严格过滤
- review queue
- scorer 公开

而不是追求训练式的大规模自动扩增。

---

## 推荐理解方式

这个仓库目前最合理的定位是：

1. **公开 benchmark 适配层**
   - 用于外部对比、基线复现、横向评测
2. **`erp_spatial_benchmark`**
   - 用于你们论文和主项目的 ERP-native 核心验证

如果你是要跑已有 benchmark，对应看上面的：

- `prepare`
- `predict`
- `evaluate`

如果你是要构建和评测你们自己的正式 benchmark，对应看：

- [erp_spatial_benchmark](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark)
