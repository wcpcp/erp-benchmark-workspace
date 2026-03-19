# Benchmark README

这份 README 只讲你当前关心的四个 benchmark：

- `osr-bench`
- `panoenv`
- `omnispatial`
- `hstar-bench-erp`

使用顺序固定是：

1. 数据准备
2. 生成预测
3. 评测打分

## 1. 数据准备

先装环境：

```bash
uv sync --project benchmark
```

查看支持的 benchmark：

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py list
```

准备你当前这四个 benchmark 的数据：

```bash
uv run --project benchmark python benchmark/scripts/prepare_benchmarks.py \
  --benchmarks osr-bench,panoenv,omnispatial,hstar-bench-erp
```

这条命令默认假设原始数据和生成的 manifest 都放在：

- `benchmark/data/<benchmark>/raw`
- `benchmark/data/<benchmark>/manifests`

如果你服务器上已经从 Hugging Face 下载好了原始数据，但放在别的位置，可以直接把现有目录告诉脚本，它会先把那个目录软链接到标准位置，再生成 manifest：

```bash
uv run --project benchmark python benchmark/scripts/prepare_benchmarks.py \
  --benchmarks osr-bench,panoenv,omnispatial,hstar-bench-erp \
  --raw-dir osr-bench=/abs/path/to/OSR-Bench/raw \
  --raw-dir panoenv=/abs/path/to/PanoEnv/raw \
  --raw-dir omnispatial=/abs/path/to/OmniSpatial/raw \
  --raw-dir hstar-bench-erp=/abs/path/to/hstar_bench/raw
```

`--raw-dir` 的格式必须是：

- `benchmark_id=/absolute/path/to/raw_dir`

只要传了 `--raw-dir`，脚本就会直接复用该目录并跳过下载步骤。

对 `hstar-bench-erp`，`--raw-dir` 既可以指向：

- 含有 `hos_bench.zip` / `hps_bench.zip` 的原始目录
- 也可以直接指向已经解压好的 H*Bench 目录树

对于 `hstar-bench-erp`，`prepare_benchmarks.py` 已经会直接生成协议 manifest，不需要再单独跑别的准备脚本。

会生成：

- `benchmark/data/hstar-bench-erp/manifests/erp_rotated_submit.jsonl`
- `benchmark/data/hstar-bench-erp/manifests/perspective_multiturn.jsonl`

### OSR-Bench 先不要跑全量

`OSR-Bench` 当前全量 manifest 是：

- `78,426` 条 QA
- 对应 `4,100` 张图

所以先用仓库里的 smoke 子集：

- [benchmark/data/osr-bench/manifests/smoke_20.jsonl](/Users/wcp/code/erp_data_pipeline/benchmark/data/osr-bench/manifests/smoke_20.jsonl)

如果你想一次把四个 benchmark 的 smoke 子集都生成出来：

```bash
uv run --project benchmark python benchmark/scripts/create_smoke_subsets.py
```

注意：

- 这个脚本现在默认是离线优先的
- 对 `OmniSpatial`，它只会复用本地已有的 `smoke/test manifest` 或本地 `parquet`
- 如果本地还没有 `OmniSpatial` 原始数据，它会跳过并提示原因，不会自动联网下载

会生成：

- `benchmark/data/osr-bench/manifests/smoke_20.jsonl`
- `benchmark/data/panoenv/manifests/smoke_20.jsonl`
- `benchmark/data/omnispatial/manifests/smoke_20.jsonl`
- `benchmark/data/hstar-bench-erp/manifests/smoke_rotated_submit_20.jsonl`

## 2. 生成预测

生成预测的脚本是：

- [benchmark/scripts/predict_benchmark.py](/Users/wcp/code/erp_data_pipeline/benchmark/scripts/predict_benchmark.py)

它的作用是：

- 读取 benchmark 样本
- 调模型
- 输出 `predictions.jsonl`

支持直接 predict 的 benchmark：

- `osr-bench`
- `panoenv`
- `omnispatial`
- `hstar-bench-erp`

支持的模型适配器：

- `mock`
- `mlx-qwen-vl`
- `transformers-vlm`
- `vllm-openai`
- `openai-api`

### 2.0 先理解不同推理后端

现在支持四种真实推理方式：

- `mlx-qwen-vl`
  - 适用：Apple Silicon / MLX
  - 输入：`--model-path`

- `transformers-vlm`
  - 适用：Linux GPU / PyTorch / Transformers
  - 输入：`--model-path`
  - 可选：`--processor-path --device-map --torch-dtype --attn-implementation`

- `vllm-openai`
  - 适用：你已经用 `vllm serve` 起好了 OpenAI-compatible 服务
  - 输入：`--model-name --api-base`

- `openai-api`
  - 适用：直接调 OpenAI 风格 API，或者别的兼容接口
  - 输入：`--model-name`
  - 可选：`--api-base`，默认 `https://api.openai.com/v1`

统一参数约定：

- `--model-path`
  - 给本地模型目录，或者 Hugging Face 模型路径
  - 主要用于 `mlx-qwen-vl` / `transformers-vlm`

- `--model-name`
  - 给 API 服务侧的模型名
  - 主要用于 `vllm-openai` / `openai-api`

- `--processor-path`
  - 只在 `transformers-vlm` 下可选
  - 用于模型权重和 processor 不在同一路径时

如果你后面要测不同模型，比如：

- `Qwen/Qwen3-VL-4B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- 你自己部署的 `qwen3-vl` / `qwen3.5` 服务

本质上只需要换：

- `--model-path`
  或
- `--model-name`

benchmark 命令本身不用改。

注意：

- 这些 benchmark 都是视觉任务
- 如果你接的是纯文本模型，即使接口兼容，也不能正确处理图像
- 正式测时请确认你接的是视觉语言模型

### 2.1 OSR-Bench smoke，先用 mock 跑通

```bash
uv run --project benchmark python benchmark/scripts/predict_benchmark.py \
  --benchmark osr-bench \
  --model mock \
  --references benchmark/data/osr-bench/manifests/smoke_20.jsonl \
  --predictions-out benchmark/results/osr_predictions_smoke.jsonl \
  --skip-download
```

### 2.2 OSR-Bench smoke，用 MLX Qwen 跑

```bash
uv run --project benchmark python benchmark/scripts/predict_benchmark.py \
  --benchmark osr-bench \
  --model mlx-qwen-vl \
  --model-path benchmark/models/Qwen3-VL-4B-Instruct-4bit \
  --references benchmark/data/osr-bench/manifests/smoke_20.jsonl \
  --predictions-out benchmark/results/osr_predictions_smoke.jsonl \
  --skip-download
```

### 2.2b OSR-Bench smoke，用 Transformers 跑

```bash
uv run --project benchmark python benchmark/scripts/predict_benchmark.py \
  --benchmark osr-bench \
  --model transformers-vlm \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --references benchmark/data/osr-bench/manifests/smoke_20.jsonl \
  --predictions-out benchmark/results/osr_predictions_smoke.jsonl \
  --torch-dtype bfloat16 \
  --device-map auto \
  --skip-download
```

### 2.2c OSR-Bench smoke，用 vLLM OpenAI 兼容服务跑

假设你在服务器上已经启动了：

```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000
```

那么预测命令是：

```bash
uv run --project benchmark python benchmark/scripts/predict_benchmark.py \
  --benchmark osr-bench \
  --model vllm-openai \
  --model-name Qwen/Qwen3-VL-4B-Instruct \
  --api-base http://127.0.0.1:8000/v1 \
  --references benchmark/data/osr-bench/manifests/smoke_20.jsonl \
  --predictions-out benchmark/results/osr_predictions_smoke.jsonl \
  --skip-download
```

### 2.2d OSR-Bench smoke，用 OpenAI 接口跑

```bash
export OPENAI_API_KEY=your_key

uv run --project benchmark python benchmark/scripts/predict_benchmark.py \
  --benchmark osr-bench \
  --model openai-api \
  --model-name gpt-4.1 \
  --references benchmark/data/osr-bench/manifests/smoke_20.jsonl \
  --predictions-out benchmark/results/osr_predictions_smoke.jsonl \
  --skip-download
```

### 2.3 PanoEnv 预测

```bash
uv run --project benchmark python benchmark/scripts/predict_benchmark.py \
  --benchmark panoenv \
  --model mlx-qwen-vl \
  --model-path benchmark/models/Qwen3-VL-4B-Instruct-4bit \
  --references benchmark/data/panoenv/manifests/test.jsonl \
  --predictions-out benchmark/results/panoenv_predictions.jsonl
```

PanoEnv smoke：

```bash
uv run --project benchmark python benchmark/scripts/predict_benchmark.py \
  --benchmark panoenv \
  --model mock \
  --references benchmark/data/panoenv/manifests/smoke_20.jsonl \
  --predictions-out benchmark/results/panoenv_predictions_smoke.jsonl \
  --skip-download
```

### 2.4 OmniSpatial 预测

```bash
uv run --project benchmark python benchmark/scripts/predict_benchmark.py \
  --benchmark omnispatial \
  --model mlx-qwen-vl \
  --model-path benchmark/models/Qwen3-VL-4B-Instruct-4bit \
  --references benchmark/data/omnispatial/manifests/test.jsonl \
  --predictions-out benchmark/results/omnispatial_predictions.jsonl
```

OmniSpatial smoke：

```bash
uv run --project benchmark python benchmark/scripts/predict_benchmark.py \
  --benchmark omnispatial \
  --model mock \
  --references benchmark/data/omnispatial/manifests/smoke_20.jsonl \
  --predictions-out benchmark/results/omnispatial_predictions_smoke.jsonl \
  --skip-download
```

### 2.5 H*Bench-ERP 预测

`erp_rotated_submit`

```bash
uv run --project benchmark python benchmark/scripts/predict_benchmark.py \
  --benchmark hstar-bench-erp \
  --model mlx-qwen-vl \
  --model-path benchmark/models/Qwen3-VL-4B-Instruct-4bit \
  --references benchmark/data/hstar-bench-erp/manifests/erp_rotated_submit.jsonl \
  --predictions-out benchmark/results/hstar_erp_submit_predictions.jsonl \
  --skip-download
```

`erp_rotated_submit` smoke

```bash
uv run --project benchmark python benchmark/scripts/predict_benchmark.py \
  --benchmark hstar-bench-erp \
  --model mock \
  --references benchmark/data/hstar-bench-erp/manifests/smoke_rotated_submit_20.jsonl \
  --predictions-out benchmark/results/hstar_erp_submit_predictions_smoke.jsonl \
  --skip-download
```

### 预测文件格式

最常见格式：

```json
{"id": "sample_id", "prediction": "your answer"}
```

注意：

- `--predictions-out` 是输出路径
- 不是评测输入
- 这一步跑完后，这个文件应该非空
- `transformers-vlm` 需要当前环境里有 `torch`
- 如果是 Linux GPU，优先推荐 `transformers-vlm` 或 `vllm-openai`
- 如果是 Apple Silicon，本地优先用 `mlx-qwen-vl`

## 3. 评测打分

评测脚本是：

- [benchmark/scripts/run_benchmark.py](/Users/wcp/code/erp_data_pipeline/benchmark/scripts/run_benchmark.py)

这一步只负责：

- 读参考答案
- 读预测文件
- 输出分数

不会自动跑模型。

### 3.1 OSR-Bench smoke

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py evaluate \
  --benchmark osr-bench \
  --references benchmark/data/osr-bench/manifests/smoke_20.jsonl \
  --predictions benchmark/results/osr_predictions_smoke.jsonl
```

### 3.2 OSR-Bench 全量

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py evaluate \
  --benchmark osr-bench \
  --references benchmark/data/osr-bench/manifests/test.jsonl \
  --predictions benchmark/results/osr_predictions.jsonl
```

### 3.3 PanoEnv

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py evaluate \
  --benchmark panoenv \
  --references benchmark/data/panoenv/manifests/test.jsonl \
  --predictions benchmark/results/panoenv_predictions.jsonl
```

PanoEnv smoke：

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py evaluate \
  --benchmark panoenv \
  --references benchmark/data/panoenv/manifests/smoke_20.jsonl \
  --predictions benchmark/templates/predictions_panoenv_smoke_template.jsonl
```

### 3.4 OmniSpatial

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py evaluate \
  --benchmark omnispatial \
  --references benchmark/data/omnispatial/manifests/test.jsonl \
  --predictions benchmark/results/omnispatial_predictions.jsonl
```

OmniSpatial smoke：

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py evaluate \
  --benchmark omnispatial \
  --references benchmark/data/omnispatial/manifests/smoke_20.jsonl \
  --predictions benchmark/templates/predictions_omnispatial_smoke_template.jsonl
```

### 3.5 H*Bench-ERP

`direct_submit`

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py evaluate \
  --benchmark hstar-bench-erp \
  --references benchmark/data/hstar-bench-erp/manifests/erp_rotated_submit.jsonl \
  --predictions benchmark/results/hstar_erp_submit_predictions.jsonl
```

`erp_rotated_submit` smoke：

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py evaluate \
  --benchmark hstar-bench-erp \
  --references benchmark/data/hstar-bench-erp/manifests/smoke_rotated_submit_20.jsonl \
  --predictions benchmark/templates/predictions_hstar_erp_rotated_submit_smoke_template.jsonl
```

## 最小跑通流程

如果你现在只想确认链路是通的，先跑这个：

### 第一步：准备 OSR-Bench

```bash
uv run --project benchmark python benchmark/scripts/prepare_benchmarks.py \
  --benchmarks osr-bench
```

### 第二步：生成预测

```bash
uv run --project benchmark python benchmark/scripts/predict_benchmark.py \
  --benchmark osr-bench \
  --model mock \
  --references benchmark/data/osr-bench/manifests/smoke_20.jsonl \
  --predictions-out benchmark/results/osr_predictions_smoke.jsonl \
  --skip-download
```

### 第三步：评测

```bash
uv run --project benchmark python benchmark/scripts/run_benchmark.py evaluate \
  --benchmark osr-bench \
  --references benchmark/data/osr-bench/manifests/smoke_20.jsonl \
  --predictions benchmark/results/osr_predictions_smoke.jsonl
```
