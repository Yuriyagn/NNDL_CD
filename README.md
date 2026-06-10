# NNDL Change Detection Model Comparison

Course project comparing remote-sensing change detection methods on a LEVIR-CD subset.

## Models

| # | Model | Type | Year |
|---|-------|------|------|
| 1 | Difference + Otsu | Traditional baseline | — |
| 2 | FC-EF | Fully Convolutional Early Fusion | 2018 |
| 3 | FC-Siam-Diff | Fully Convolutional Siamese Difference | 2018 |
| 4 | STANet | Spatio-Temporal Attention Network | 2020 |
| 5 | SNUNet-CD | Siamese Nested U-Net with ECAM | 2021 |
| 6 | TinyCD | EfficientNet Siamese U-Net with mix-attention | 2022 |

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

Crop the LEVIR-CD 1024×1024 images into 256×256 patches:

```bash
python tools/crop_levir.py --overwrite
python tools/check_dataset.py --root data/LEVIR_CD_SUBSET_256
```

## Usage

**Train a model:**

```bash
python train.py --config configs/fc_siam_diff.yaml
```

**Test a model:**

```bash
python test.py --checkpoint results/checkpoints/fc_siam_diff/best.pt --save-masks
```

**Traditional baseline:**

```bash
python test.py --model difference_otsu
```

**Run all experiments:**

```bash
python tools/run_experiments.py --save-masks
```

**Quick smoke test (CPU):**

```text
--epochs 1 --batch-size 2 --base-channels 8 --max-train-batches 2 --max-val-batches 1
```

## Project Structure

```
NNDL_CD/
├── train.py                  # Training entry
├── test.py                   # Evaluation entry
├── infer.py                  # Single-pair inference
├── experiment_utils.py       # Shared utilities
├── requirements.txt          # Dependencies
├── models/                   # Model implementations
│   ├── fc_ef.py, fc_siam_diff.py, stanet.py
│   ├── snunet_cd.py, tinycd.py, difference.py
│   └── common.py             # Shared building blocks
├── datasets/levir_cd.py      # LEVIR-CD dataset loader
├── losses/bce_dice.py        # BCE + Dice loss
├── metrics/cd_metrics.py     # F1, IoU, OA, Precision, Recall
├── configs/*.yaml            # Per-model hyperparameters
└── tools/                    # Utility scripts
    ├── crop_levir.py, check_dataset.py
    ├── run_experiments.py, compare_predictions.py
    ├── count_flops_params.py, count_params.py
    └── ...
```

## References

- [FC-EF / FC-Siam-Diff (ICIP 2018)](https://arxiv.org/abs/1810.08462)
- [STANet (Remote Sensing 2020)](https://www.mdpi.com/2072-4292/12/10/1662)
- [SNUNet-CD (IEEE GRSL 2021)](https://ieeexplore.ieee.org/document/9355573)
- [TinyCD (2022)](https://arxiv.org/abs/2207.13159)
