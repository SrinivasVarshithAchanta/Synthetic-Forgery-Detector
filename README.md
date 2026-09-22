# Synthetic Forgery Detector

End-to-end **document forgery detection**: a CNN classifier that separates
**tampered** from **genuine** identity documents, trained entirely on
procedurally generated synthetic documents (no real PII) with four
programmatic forgery types.

> Final measured metrics live in [`RESULTS.md`](RESULTS.md).

## Repository layout

```
data/                     # generated dataset (gitignored, rebuild in ~4 min)
scripts/build_dataset.py  # clean mock IDs -> 4 forgery types -> split
src/
  mock_ids.py             # procedural PII-free ID document generator
  forgery_generator.py    # copy-move / splicing / resampling / JPEG forgery
  augmentations.py        # phone-capture aug (train-time only)
  dataset.py              # manifest loading, splits, composition tables
  models.py               # resnet50/18, efficientnet_b0, lightweight CNN
  losses.py               # cross-entropy + focal loss
  metrics.py              # precision/recall/F1/FAR/FRR
  train.py                # AMP training loop + W&B logging
  evaluate.py             # 3-slice evaluation harness (used by evaluate.py)
  sweep_threshold.py      # threshold sweep + PR / FAR curves
  gradcam.py              # Grad-CAM saliency for hard cases
  confusion_analysis.py   # error buckets by type x difficulty + hard cases
  benchmark_latency.py    # per-image latency, AMP vs FP32, resolutions
evaluate.py               # thin CLI wrapper -> src/evaluate.py
tests/verify_pipeline.py  # verification suite (dataset, gen, metrics, model)
notebooks/01_explore_dataset.ipynb
reports/                  # results.json, figures, latency, error buckets
runs/<run>/               # config.json, history.jsonl, best.pt, last.pt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
# optional, for online experiment tracking
wandb login
```

## 1. Build the dataset

```bash
python scripts/build_dataset.py --sources 3000 --seed 42 --out data
```

- 3000 procedurally rendered mock ID documents (no real PII).
- **Source-level** 70/15/15 train/val/test split (no clean document leaks
  across splits) -> `data/splits.json`.
- Each source contributes 1 genuine + 1 tampered image -> **6000 images**.
- Tampered images use the four forgery types round-robined per split;
  difficulty sampled subtle 40% / moderate 35% / aggressive 25%.
- Masks under `data/masks/`, manifest `data/manifest.jsonl`.

## 2. Train

```bash
# baseline: cross-entropy (ResNet-50, ImageNet weights, AMP, W&B)
python -m src.train --model resnet50 --epochs 18 --loss ce \
    --batch-size 96 --workers 8 --run-name baseline-ce

# focal loss + rebalancing toward subtle forgeries
python -m src.train --model resnet50 --epochs 18 --loss focal \
    --focal-gamma 2.0 --balance-sampler --subtle-weight 2.5 \
    --tampered-weight 1.5 --batch-size 96 --workers 8 --run-name focal-subtle

# backbone comparison
python -m src.train --model efficientnet_b0 --epochs 18 --run-name effnet-b0
python -m src.train --model custom_cnn --epochs 25 --lr 1e-3 --run-name custom-cnn
```

All runs log config + metrics to Weights & Biases (offline automatically
when no API key is configured; `wandb sync` uploads later). Checkpoints and
`history.jsonl` are always written to `runs/<run_name>/`.

## 3. Evaluate

```bash
# precision/recall/F1/FAR on clean / noisy / adversarial test slices
python evaluate.py --checkpoint runs/baseline-ce/best.pt --choose-threshold val \
    --json reports/eval_baseline.json

# threshold sweep -> PR curve + FAR-vs-threshold curve
python -m src.sweep_threshold --checkpoint runs/baseline-ce/best.pt

# error buckets (forgery type x difficulty) + hardest cases for review
python -m src.confusion_analysis --checkpoint runs/baseline-ce/best.pt

# Grad-CAM on correctly and incorrectly classified tampered images
python -m src.gradcam --checkpoint runs/baseline-ce/best.pt

# latency: backbones x resolutions x AMP/FP32 on GPU (and CPU)
python -m src.benchmark_latency --cpu --wandb
```

## 4. Verify

```bash
python tests/verify_pipeline.py     # or: pytest -q tests/
```

Rebuild-from-scratch smoke test (fast): rebuilds a 60-source dataset into a
temp dir and exercises the generator, then the trained-model checks run
against `reports/results.json`.

## Results

See [`RESULTS.md`](RESULTS.md) for the final measured numbers: dataset
composition, accuracy/precision/recall/FAR per slice, baseline vs focal-loss
FAR delta on subtle forgeries, and measured inference latency.

## Reproducing

```bash
python scripts/build_dataset.py --sources 3000 --seed 42 --out data
bash scripts/run_all.sh          # trains, evaluates, sweeps, benchmarks
python tests/verify_pipeline.py
```

## Notes

- **No real PII**: documents, names, portraits and MRZ lines are fully
  synthetic. The dataset is designed for research on tamper detection.
- FAR = fraction of *tampered* documents accepted as genuine (1 - recall on
  tampered); FRR = genuine flagged as tampered.
