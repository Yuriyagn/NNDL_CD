# EXPERIMENT_PROTOCOL

## 0. 审计结论摘要

本协议面向课程正式实验。当前主实验数据集应固定为 `data/LEVIR_CD_SUBSET_256`。`data/LEVIR_CD_TINYCD_OVERFIT_256` 只能作为 TinyCD 过拟合诊断集，不能混入主实验、主结果表或正式 PPT/报告结论。

当前 `results/tables/metrics_summary.md` 中 Difference + Otsu、FC-EF、FC-Siam-Diff、STANet、SNUNet-CD 可追溯到主数据集或主实验流程；TinyCD 行存在 compact / official-style / overfit 混杂，不能作为正式主实验结果引用。

证据路径：

- 主数据集：`data/LEVIR_CD_SUBSET_256`
- TinyCD 诊断集：`data/LEVIR_CD_TINYCD_OVERFIT_256`
- 当前结果表：`results/tables/metrics_summary.md`
- checkpoint：`results/checkpoints/*/*.pt`
- TinyCD 变体选择逻辑：`models/tinycd.py`
- checkpoint 保存逻辑：`train.py`
- 测试阈值读取逻辑：`test.py`

## 1. 数据集审计

### 1.1 主实验数据集

主实验数据集固定为：

```text
data/LEVIR_CD_SUBSET_256
```

轻量检查命令：

```bash
conda run -n nndl_cd python tools/check_dataset.py --root data/LEVIR_CD_SUBSET_256
```

检查结果：

| Split | 样本数 | 图像尺寸 | 原始标签取值 | 变化像素比例 | 是否纳入主实验 |
|---|---:|---|---|---:|---|
| train | 160 | 256x256 | 0, 255 | 0.0332 | 是 |
| val | 80 | 256x256 | 0, 255 | 0.0351 | 是 |
| test | 80 | 256x256 | 0, 255 | 0.0182 | 是 |

说明：

- 数据布局为 `split/A`, `split/B`, `split/label`，证据见 `data/LEVIR_CD_SUBSET_256/train/A`, `data/LEVIR_CD_SUBSET_256/train/B`, `data/LEVIR_CD_SUBSET_256/train/label`。
- `datasets/levir_cd.py` 将原始 `0/255` 标签通过 `mask > 127` 转换为 `0/1`。
- test split 变化像素比例仅 0.0182，类别不平衡明显。因此正式报告应重点使用 F1、IoU、Precision、Recall，不应只看 OA。

### 1.2 TinyCD 过拟合诊断集

诊断集路径：

```text
data/LEVIR_CD_TINYCD_OVERFIT_256
```

轻量检查命令：

```bash
conda run -n nndl_cd python tools/check_dataset.py --root data/LEVIR_CD_TINYCD_OVERFIT_256
```

检查结果：

| Split | 样本数 | 图像尺寸 | 原始标签取值 | 变化像素比例 | 是否纳入主实验 |
|---|---:|---|---|---:|---|
| train | 20 | 256x256 | 0, 255 | 0.1740 | 否 |
| val | 20 | 256x256 | 0, 255 | 0.1740 | 否 |
| test | 20 | 256x256 | 0, 255 | 0.1740 | 否 |

结论：

- 该数据集是从高变化样本中选出的 TinyCD 过拟合诊断集，证据见 `data/LEVIR_CD_TINYCD_OVERFIT_256/manifest.csv` 和 `configs/tinycd_overfit.yaml`。
- 该数据集的变化像素比例 0.1740 明显高于主数据集 test 的 0.0182，不具备与主实验公平比较的分布一致性。
- 任何来自 `data/LEVIR_CD_TINYCD_OVERFIT_256` 的 checkpoint、曲线、预测图、指标都不得写入正式主实验表。

## 2. Checkpoint 元数据审计

审计对象：

```text
results/checkpoints/*/*.pt
```

checkpoint 内部包含的主要字段为：

```text
model, model_kwargs, epoch, state_dict, val_metrics, best_threshold, config
```

未发现独立的 `best_metric` 字段；当前可使用 `val_metrics.f1` 作为 checkpoint 选择依据。证据见 `train.py`：保存字段为 `val_metrics` 和 `best_threshold`，最佳模型按验证集 F1 选择。

| Checkpoint | model | data_root | epoch | best_metric / val F1 | best_threshold | config 摘要 | 正式主实验可引用 |
|---|---|---|---:|---:|---:|---|---|
| `results/checkpoints/fc_ef/best.pt` | fc_ef | `data/LEVIR_CD_SUBSET_256` | 6 | 0.7554 | 0.70 | base_channels=16, epochs=10, batch_size=8, lr=0.001 | 可引用 |
| `results/checkpoints/fc_ef/last.pt` | fc_ef | `data/LEVIR_CD_SUBSET_256` | 10 | 0.7294 | 0.70 | 同上 | 不建议作为最佳结果 |
| `results/checkpoints/fc_siam_diff/best.pt` | fc_siam_diff | `data/LEVIR_CD_SUBSET_256` | 7 | 0.6656 | 0.70 | base_channels=16, epochs=10, batch_size=8, lr=0.001 | 可引用 |
| `results/checkpoints/fc_siam_diff/last.pt` | fc_siam_diff | `data/LEVIR_CD_SUBSET_256` | 10 | 0.5467 | 0.70 | 同上 | 不建议作为最佳结果 |
| `results/checkpoints/stanet/best.pt` | stanet | `data/LEVIR_CD_SUBSET_256` | 8 | 0.6956 | 0.70 | base_channels=16, epochs=10, batch_size=8, lr=0.001 | 可引用 |
| `results/checkpoints/stanet/last.pt` | stanet | `data/LEVIR_CD_SUBSET_256` | 10 | 0.6462 | 0.70 | 同上 | 不建议作为最佳结果 |
| `results/checkpoints/snunet_cd/best.pt` | snunet_cd | `data/LEVIR_CD_SUBSET_256` | 9 | 0.7200 | 0.70 | base_channels=12, epochs=10, batch_size=4, lr=0.001 | 可引用 |
| `results/checkpoints/snunet_cd/last.pt` | snunet_cd | `data/LEVIR_CD_SUBSET_256` | 10 | 0.6905 | 0.70 | 同上 | 不建议作为最佳结果 |
| `results/checkpoints/tinycd/best.pt` | tinycd | `data/LEVIR_CD_TINYCD_OVERFIT_256` | 57 | 0.5992 | 0.35 | model_kwargs 含 `base_channels=12`，会触发 compact 兼容路径 | 不可引用 |
| `results/checkpoints/tinycd/last.pt` | tinycd | `data/LEVIR_CD_TINYCD_OVERFIT_256` | 100 | 0.5737 | 0.40 | compact 兼容路径，过拟合诊断集 | 不可引用 |
| `results/checkpoints/tinycd_overfit/best.pt` | tinycd | `data/LEVIR_CD_TINYCD_OVERFIT_256` | 77 | 0.9611 | 0.70 | official-style, run_name=tinycd_overfit, epochs=100 | 只能作为诊断结果 |
| `results/checkpoints/tinycd_overfit/last.pt` | tinycd | `data/LEVIR_CD_TINYCD_OVERFIT_256` | 100 | 0.9592 | 0.70 | official-style, run_name=tinycd_overfit, epochs=100 | 只能作为诊断结果 |

结论：

- 当前四个主模型 `fc_ef`, `fc_siam_diff`, `stanet`, `snunet_cd` 的 best checkpoint 绑定 `data/LEVIR_CD_SUBSET_256`。
- 当前 `results/checkpoints/tinycd/` 不是主数据集结果，而是 TinyCD 过拟合诊断集结果。
- 当前 `results/checkpoints/tinycd_overfit/` 明确是 official-style TinyCD 过拟合诊断结果，只能用于说明 TinyCD pipeline/结构能否学习，不得作为主实验公平对比结果。

## 3. `metrics_summary.md` 结果表审计

当前表路径：

```text
results/tables/metrics_summary.md
```

当前内容：

| Model | Params | Precision | Recall | F1 | IoU | OA | 来源一致性判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| Difference + Otsu | - | 0.0165 | 0.3366 | 0.0315 | 0.0160 | 0.6224 | 可作为主数据集 baseline；证据见 `project_context/run_log.md` 和 `test.py --model difference_otsu` 默认主数据集 |
| FC-EF | 483.2K | 0.5954 | 0.8808 | 0.7105 | 0.5510 | 0.9869 | 可追溯到主数据集 checkpoint：`results/checkpoints/fc_ef/best.pt` |
| FC-Siam-Diff | 482.7K | 0.4530 | 0.9141 | 0.6058 | 0.4345 | 0.9783 | 可追溯到主数据集 checkpoint：`results/checkpoints/fc_siam_diff/best.pt` |
| STANet | 800.5K | 0.7283 | 0.8108 | 0.7673 | 0.6225 | 0.9910 | 可追溯到主数据集 checkpoint：`results/checkpoints/stanet/best.pt` |
| SNUNet-CD | 1.29M | 0.6587 | 0.9101 | 0.7642 | 0.6184 | 0.9898 | 可追溯到主数据集 checkpoint：`results/checkpoints/snunet_cd/best.pt` |
| TinyCD | 285.1K | 0.0806 | 0.8767 | 0.1476 | 0.0797 | 0.8153 | 不可作为正式主实验引用；当前 TinyCD checkpoint 与结果行存在混杂 |

TinyCD 问题证据：

- `results/tables/metrics_summary.md` 中 TinyCD 参数量为 285.1K，符合 official-style TinyCD 口径。
- 但当前 `results/checkpoints/tinycd/best.pt` 的 `data_root` 是 `data/LEVIR_CD_TINYCD_OVERFIT_256`，且 `model_kwargs` 为 `{"base_channels": 12}`。
- `models/tinycd.py` 明确：传入 `base_channels` 会把 TinyCD 切换到 `compact` 兼容路径。
- `results/checkpoints/tinycd_overfit/best.pt` 是 official-style，但其 `data_root` 仍是 `data/LEVIR_CD_TINYCD_OVERFIT_256`。
- 因此当前没有可直接引用的 official-style TinyCD 主数据集 checkpoint。

## 4. TinyCD 混杂审计

当前 TinyCD 至少有三种口径：

| 口径 | 证据路径 | 数据集 | 变体 | 用途 | 是否纳入主实验 |
|---|---|---|---|---|---|
| 旧 compact TinyCD | `results/checkpoints/tinycd/best.pt` | `data/LEVIR_CD_TINYCD_OVERFIT_256` | compact, 由 `base_channels=12` 触发 | 历史诊断 | 否 |
| official-style overfit TinyCD | `results/checkpoints/tinycd_overfit/best.pt` | `data/LEVIR_CD_TINYCD_OVERFIT_256` | official, EfficientNet-B4 + MAMB + UpMask | 过拟合诊断 | 否 |
| official-style 主实验 TinyCD | `configs/tinycd_tuned.yaml` | `data/LEVIR_CD_SUBSET_256` | official | 应作为正式 TinyCD 结果 | 尚未完成或未发现 checkpoint |

正式结论：

- TinyCD 当前不能进入正式主结果表。
- 正式实验中应使用 `configs/tinycd_tuned.yaml`，并将结果保存到独立输出目录，例如 `results/formal_debug_v1`，避免覆盖旧结果。

## 5. 正式实验协议

### 5.1 主实验数据集

正式主实验只使用：

```text
data/LEVIR_CD_SUBSET_256
```

禁止使用：

```text
data/LEVIR_CD_TINYCD_OVERFIT_256
```

除非章节明确写作“TinyCD 过拟合诊断实验”。

### 5.2 模型列表与 config

| 编号 | 模型 | config | 是否纳入正式主实验 |
|---|---|---|---|
| M0 | Difference + Otsu | 无需训练，使用 `test.py --model difference_otsu` | 是 |
| M1 | FC-EF | `configs/fc_ef.yaml` | 是 |
| M2 | FC-Siam-Diff | `configs/fc_siam_diff.yaml` | 是 |
| M3 | STANet | `configs/stanet.yaml` | 是 |
| M4 | SNUNet-CD | `configs/snunet_cd.yaml` | 是 |
| M5 | TinyCD official-style | `configs/tinycd_tuned.yaml` | 是，但需重新生成干净结果 |

### 5.3 训练轮数建议

课程正式结果建议：

| 模型 | 建议 epochs | 说明 |
|---|---:|---|
| FC-EF | 30 | 当前 10 epoch 已可用，但正式报告建议 30 epoch |
| FC-Siam-Diff | 30 | 与 FC-EF 保持一致 |
| STANet | 30 | 与其他主模型保持一致 |
| SNUNet-CD | 30 | 计算较重时可保留 10 epoch 调试结果并说明限制 |
| TinyCD official-style | 30 | 使用 `configs/tinycd_tuned.yaml` |

如 GPU 时间不足，最低可交付口径为：保留已完成的 10 epoch 主数据集结果，并只补齐 TinyCD official-style 主数据集结果。

### 5.4 统一训练参数

推荐统一设置：

| 参数 | 建议值 | 证据/说明 |
|---|---|---|
| input size | 256x256 | `data/LEVIR_CD_SUBSET_256` |
| optimizer | AdamW | `train.py`, `configs/*.yaml` |
| loss | BCEWithLogits + Dice | `losses/bce_dice.py` |
| lr | 0.001 | 当前 configs |
| weight_decay | 0.0001 | 当前主实验 configs |
| batch size | 8 优先；显存不足用 4 | 当前 checkpoint 中 FC-EF/FC-Siam-Diff/STANet 为 8，SNUNet-CD 为 4 |
| pos_weight | auto | `train.py` 会按训练集估计正类权重 |
| pos_weight_max | 20；TinyCD tuned 可用 10 | `configs/*.yaml` |
| scheduler | cosine | `train.py` |
| gradient_clip | 1.0 | `configs/*.yaml` |
| seed | 42 | 当前 configs |

### 5.5 阈值策略

训练阶段：

- 在验证集上从候选阈值中选择 F1 最高的阈值。
- 主模型默认候选阈值为 `[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`。
- TinyCD tuned 候选阈值为 `[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]`。

测试阶段：

- `test.py` 会优先读取 checkpoint 中的 `best_threshold`。
- 正式结果表应明确写明“测试使用验证集选择的最佳阈值”。
- 如需公平补充，可另给固定阈值 0.50 表；已有证据路径为 `results/tables/metrics_fixed_threshold_0.50.csv`。

### 5.6 结果保存规范

为避免覆盖旧结果，正式实验不要直接写入根目录 `results/checkpoints/<model>`。建议使用独立输出目录：

```text
results/formal_debug_v1
```

推荐保存结构：

```text
results/formal_debug_v1/
├── checkpoints/
│   ├── fc_ef/
│   ├── fc_siam_diff/
│   ├── stanet/
│   ├── snunet_cd/
│   └── tinycd_tuned/
├── curves/
├── predictions/
├── tables/
└── protocol_notes.md
```

正式报告只引用以下内容：

- `results/formal_debug_v1/tables/metrics_summary.md`
- `results/formal_debug_v1/curves/*.png`
- `results/formal_debug_v1/predictions/*`
- `results/formal_debug_v1/checkpoints/*/best.pt`

旧结果可保留，但引用时必须标注来源。

## 6. 旧结果引用规则

### 6.1 可以保留且可作为调试主实验参考的旧结果

| 结果 | 路径 | 处理建议 |
|---|---|---|
| Difference + Otsu baseline | `results/tables/metrics_summary.md`, `results/predictions/difference_otsu/` | 可作为主数据集 baseline；正式表建议重跑到新输出目录 |
| FC-EF 10 epoch | `results/checkpoints/fc_ef/best.pt`, `results/curves/fc_ef_curve.png` | 可作为调试主实验结果；正式表建议重跑或标注 10 epoch |
| FC-Siam-Diff 10 epoch | `results/checkpoints/fc_siam_diff/best.pt`, `results/curves/fc_siam_diff_curve.png` | 可作为调试主实验结果；正式表建议重跑或标注 10 epoch |
| STANet 10 epoch | `results/checkpoints/stanet/best.pt`, `results/curves/stanet_curve.png` | 可作为调试主实验结果；正式表建议重跑或标注 10 epoch |
| SNUNet-CD 10 epoch | `results/checkpoints/snunet_cd/best.pt`, `results/curves/snunet_cd_curve.png` | 可作为调试主实验结果；正式表建议重跑或标注 10 epoch |
| 固定阈值审计 | `results/tables/metrics_fixed_threshold_0.50.csv` | 可作为补充审计，不替代主结果表 |
| 可靠性审计 | `results/tables/reliability_audit.md` | 可作为实验可信度说明 |

### 6.2 不能作为正式主实验结果引用的旧结果

| 结果 | 路径 | 原因 |
|---|---|---|
| TinyCD compact 旧 checkpoint | `results/checkpoints/tinycd/best.pt` | 绑定 `data/LEVIR_CD_TINYCD_OVERFIT_256`，且 `base_channels=12` 触发 compact 路径 |
| TinyCD compact 曲线 | `results/curves/tinycd_curve.*` | 对应过拟合诊断，不是主数据集 |
| TinyCD official-style overfit checkpoint | `results/checkpoints/tinycd_overfit/best.pt` | 绑定过拟合诊断集，只能说明可学习性 |
| TinyCD overfit 曲线 | `results/curves/tinycd_overfit_curve.*` | 诊断用途，不可进入主实验表 |
| `metrics_summary.md` 的 TinyCD 行 | `results/tables/metrics_summary.md` | 参数口径与当前 checkpoint 元数据不一致，存在混杂 |
| 旧 comparison 图中的 TinyCD 列 | `results/predictions/comparison/`, `results/predictions/comparison_top_change/` | 很可能引用了旧 TinyCD checkpoint，正式展示前应重生成 |

## 7. 建议训练与评估指令

以下指令交给用户在合适环境中运行。它们会写入新目录 `results/formal_debug_v1`，避免覆盖旧结果。

### 7.1 传统 baseline

```bash
conda run -n nndl_cd python test.py \
  --model difference_otsu \
  --data-root data/LEVIR_CD_SUBSET_256 \
  --batch-size 8 \
  --max-visuals 10 \
  --save-masks \
  --output-dir results/formal_debug_v1
```

### 7.2 主模型训练

如果训练时间充足，建议统一 30 epoch：

```bash
conda run -n nndl_cd python train.py --config configs/fc_ef.yaml \
  --epochs 30 --batch-size 8 --run-name fc_ef --output-dir results/formal_debug_v1 --device cuda

conda run -n nndl_cd python train.py --config configs/fc_siam_diff.yaml \
  --epochs 30 --batch-size 8 --run-name fc_siam_diff --output-dir results/formal_debug_v1 --device cuda

conda run -n nndl_cd python train.py --config configs/stanet.yaml \
  --epochs 30 --batch-size 8 --run-name stanet --output-dir results/formal_debug_v1 --device cuda

conda run -n nndl_cd python train.py --config configs/snunet_cd.yaml \
  --epochs 30 --batch-size 4 --run-name snunet_cd --output-dir results/formal_debug_v1 --device cuda

conda run -n nndl_cd python train.py --config configs/tinycd_tuned.yaml \
  --epochs 30 --batch-size 8 --run-name tinycd_tuned --output-dir results/formal_debug_v1 --device cuda
```

如果 CUDA 不可用或训练时间过长，请由用户在 GPU 环境运行。当前 `README.md` 记录本机 PyTorch 曾报告 CUDA 不可用；如当前环境仍不可用，不建议在 CPU 上跑正式 30 epoch。

### 7.3 主模型测试

```bash
conda run -n nndl_cd python test.py \
  --checkpoint results/formal_debug_v1/checkpoints/fc_ef/best.pt \
  --data-root data/LEVIR_CD_SUBSET_256 \
  --batch-size 8 --device cuda --max-visuals 10 --save-masks \
  --output-dir results/formal_debug_v1

conda run -n nndl_cd python test.py \
  --checkpoint results/formal_debug_v1/checkpoints/fc_siam_diff/best.pt \
  --data-root data/LEVIR_CD_SUBSET_256 \
  --batch-size 8 --device cuda --max-visuals 10 --save-masks \
  --output-dir results/formal_debug_v1

conda run -n nndl_cd python test.py \
  --checkpoint results/formal_debug_v1/checkpoints/stanet/best.pt \
  --data-root data/LEVIR_CD_SUBSET_256 \
  --batch-size 8 --device cuda --max-visuals 10 --save-masks \
  --output-dir results/formal_debug_v1

conda run -n nndl_cd python test.py \
  --checkpoint results/formal_debug_v1/checkpoints/snunet_cd/best.pt \
  --data-root data/LEVIR_CD_SUBSET_256 \
  --batch-size 4 --device cuda --max-visuals 10 --save-masks \
  --output-dir results/formal_debug_v1

conda run -n nndl_cd python test.py \
  --checkpoint results/formal_debug_v1/checkpoints/tinycd_tuned/best.pt \
  --data-root data/LEVIR_CD_SUBSET_256 \
  --batch-size 8 --device cuda --max-visuals 10 --save-masks \
  --output-dir results/formal_debug_v1
```

### 7.4 跨模型可视化

```bash
conda run -n nndl_cd python tools/compare_predictions.py \
  --data-root data/LEVIR_CD_SUBSET_256 \
  --split test \
  --include-difference \
  --checkpoints \
  results/formal_debug_v1/checkpoints/fc_ef/best.pt \
  results/formal_debug_v1/checkpoints/fc_siam_diff/best.pt \
  results/formal_debug_v1/checkpoints/stanet/best.pt \
  results/formal_debug_v1/checkpoints/snunet_cd/best.pt \
  results/formal_debug_v1/checkpoints/tinycd_tuned/best.pt \
  --max-samples 10 \
  --selection top-change \
  --device cuda \
  --output-dir results/formal_debug_v1/predictions/comparison_top_change
```

## 8. 正式报告引用口径

正式报告中建议写：

> 本实验以 `data/LEVIR_CD_SUBSET_256` 为主实验数据集，所有正式模型均在相同 train/val/test 划分上训练和测试。TinyCD 过拟合诊断集 `data/LEVIR_CD_TINYCD_OVERFIT_256` 仅用于验证 TinyCD 实现能否在高变化小样本上学习，不参与主实验定量比较。

对于当前旧结果，建议写：

> 现有 10 epoch 结果可作为调试实验参考；由于 TinyCD 历史结果中存在 compact、official-style 和 overfit 数据集混杂，正式表格需重新生成 TinyCD official-style 主数据集结果后再引用。

