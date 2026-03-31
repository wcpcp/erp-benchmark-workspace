# Benchmark Workspace

这个 `benchmark/` 目录当前承担两条工作线：

1. **统一封装并评测已有 benchmark**
   - 入口主要在 [src/erp_benchmarks](/Users/wcp/code/erp_data_pipeline/benchmark/src/erp_benchmarks)
   - 用于公开 benchmark 对比和外部基线

2. **构建我们自己的 ERP 空间 benchmark**
   - 当前正式版本在 [erp_spatial_benchmark](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark)
   - 用于测试模型是否真正理解 ERP 所表示的全景空间

所以这个仓库没有“改名成只做自定义 benchmark”，而是在保留原有公开 benchmark 评测层的同时，新增了一个正式的 ERP benchmark 构建层。

## 目录结构

### 1. 公开 benchmark 统一评测层

- [src/erp_benchmarks](/Users/wcp/code/erp_data_pipeline/benchmark/src/erp_benchmarks)
- [registry.yaml](/Users/wcp/code/erp_data_pipeline/benchmark/registry.yaml)
- [scripts/prepare_benchmarks.py](/Users/wcp/code/erp_data_pipeline/benchmark/scripts/prepare_benchmarks.py)
- [scripts/predict_benchmark.py](/Users/wcp/code/erp_data_pipeline/benchmark/scripts/predict_benchmark.py)
- [scripts/run_benchmark.py](/Users/wcp/code/erp_data_pipeline/benchmark/scripts/run_benchmark.py)
- [data](/Users/wcp/code/erp_data_pipeline/benchmark/data)
- [results](/Users/wcp/code/erp_data_pipeline/benchmark/results)
- [reports](/Users/wcp/code/erp_data_pipeline/benchmark/reports)

### 2. 自定义 ERP benchmark 构建层

- [erp_spatial_benchmark](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark)
  - 当前正式版 builder、design 和 scorer
- [custom_erp_foundation_benchmark](/Users/wcp/code/erp_data_pipeline/benchmark/custom_erp_foundation_benchmark)
  - 早期草案和探索，不再作为正式入口

### 3. 第三方参考资源

- [_third_party](/Users/wcp/code/erp_data_pipeline/benchmark/_third_party)

## 当前 ERP benchmark 在测什么

当前正式版 ERP benchmark 不想测“泛化视觉问答”，而是专门测：

> 模型是否把 ERP 图像学成了一个有意义的球面 / 全景 / observer-centered / ERP-specific 空间表示。

当前的四类核心能力是：

### 1. `spherical_localization_and_panoramic_topology`

对应任务：

- `referring_grounding_bfov`
- `absolute_direction_mc`
- `relative_direction_mc`

这组测：

- 单目标球面定位
- 绝对方向理解
- 全景环的相对角关系

### 2. `viewpoint_conditioned_spatial_updating`

对应任务：

- `camera_rotation_transform_mc`
- `object_conditioned_reorientation_mc`

这组测：

- 当观察方向变化时，模型能否正确更新空间关系

### 3. `observer_centered_3d_layout_understanding`

对应任务：

- `observer_distance_choice`
- `relative_3d_position_mc`

这组测：

- 是否能从 ERP 中恢复 observer-centered 3D 布局

### 4. `erp_representation_understanding`

对应任务：

- `seam_continuity_mc`
- `polar_shape_recovery_mc`

这组测：

- 左右 seam 的 wrap-around 连续性
  - 当前 seam continuity 统一为 5 个固定子题：
    - 跨边界最近邻
    - 跨边界相对方向
    - 去重计数
    - 结构连续性
    - 同一实体判断
  - 每个子题当前只保留 1 个固定 benchmark 模板，不做多模板扩增
  - 其中 `structure_continuity` 只对结构/表面类目标出题，例如墙、桌面/台面、路面/地面、天花、护栏
- 高纬 / 极区畸变理解
- ERP 表示特性本身，而不是一般空间问答

当前正式 scored benchmark 里，rotation consistency 暂时不作为单独 QA 任务发布。
它更适合作为后续的 paired diagnostic protocol，而不是当前版本的 headline task。

当前推荐的补充协议在：

- [ROTATION_PROTOCOL.md](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/ROTATION_PROTOCOL.md)
- [rotation_protocol.py](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/rotation_protocol.py)

这里特别要区分：

- `view_transform`：观察者转了
- `rotation_consistency`：ERP 表示被水平重参数化了

这两者不是一回事。

## 当前 benchmark 与训练集的区别

### 训练集

训练集的目标是：

- 大规模监督
- 覆盖能力尽量广
- 允许一定自动近似和自然语言变体
- 一部分题可经由后处理重包装

### benchmark

benchmark 的目标是：

- 低噪声
- 稳定可评测
- 解释性强
- 尽量避免训练分布泄漏

所以 benchmark 当前特点是：

- benchmark-only 模板
- 闭集 / 固定标签优先
- 更严格过滤
- review queue
- scorer 直接支持 closed-form exact match
- anchor 选择会过滤低区分度、高重复类别，例如：
  - `tree`
  - `window`
  - `leaf`
  - `branch`
  - `foliage`
  - `bush`
  - `shrub`
  - `plant`
- 只有极区 / 高纬畸变相关题会使用“粗 label + 定位”的安全引用
  - 若使用 box，则统一使用 `0-1000` 归一化后的 ERP box
  - 其他任务仍保留 `reground_query / caption_brief` 风格引用，以保留语言-视觉对齐测试价值

一句话说：

- **训练集负责“教会模型”**
- **benchmark 负责“确认模型到底学会了什么”**

## 当前 benchmark 如何打分

正式 scorer 在：

- [evaluate_predictions.py](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/evaluate_predictions.py)

当前 headline metric 是：

- `ability_group_macro_accuracy`

其中 `referring_grounding_bfov` 不是普通多选题，而是直接预测 BFOV，并用
seam-aware spherical BFOV IoU 进行评估；headline 上会把它阈值化为正确率，
同时额外报告平均 IoU 和中心误差。

同时报告：

- `task_macro_score`
- `overall`
- `by_task`
- `by_ability_group`
- `by_diagnostic_slice`

这样更符合 benchmark 的论文叙事：先看四类核心能力，再看具体任务和 slice。

## 当前发布方式

当前正式版 builder 采用 **单一公开 benchmark** 发布方式，所有标准答案公开。

默认导出：

- `benchmark_public.jsonl`
- `benchmark_public_prompts.jsonl`
- `benchmark_public_references.jsonl`

以及辅助文件：

- `candidate_pool.jsonl`
- `review_queue.jsonl`
- `summary.json`

这样更符合你当前论文阶段的目标：

- benchmark 全量公开
- scorer 公开
- 所有人都能直接复现同一套评测

## 主入口

如果你要基于 1k 高质量 ERP 图构建 benchmark：

- [build_benchmark.py](/Users/wcp/code/erp_data_pipeline/benchmark/erp_spatial_benchmark/build_benchmark.py)

最基本命令：

```bash
python3 benchmark/erp_spatial_benchmark/build_benchmark.py \
  --input-root /path/to/erp_benchmark_metadata_root \
  --output-dir /path/to/erp_spatial_benchmark_out \
  --target-public-per-task 250 \
  --seed 20260327
```

## 最终建议

现在最合理的理解方式是：

1. 公开 benchmark 适配层负责外部对比
2. `erp_spatial_benchmark` 负责论文核心验证

其中论文主 benchmark 的故事应该围绕这四件事展开：

- ERP 球面定位与全景拓扑
- 视角变化下的空间更新
- observer-centered 3D 布局理解
- ERP 表示特性理解
