#!/usr/bin/env bash
# Full experiment reproduction: trains every model, evaluates, sweeps,
# runs the error analysis, Grad-CAM and the latency benchmark.
# Run order matters: the focal delta is computed after both models exist.
set -euo pipefail

echo "== 1/7 baseline (ResNet-50, cross-entropy) =="
python -m src.train --model resnet50 --epochs 18 --loss ce \
    --batch-size 96 --workers 8 --run-name baseline-ce

echo "== 2/7 baseline evaluation (3 slices) =="
python evaluate.py --checkpoint runs/baseline-ce/best.pt --choose-threshold val \
    --json reports/eval_baseline.json

echo "== 3/7 focal loss + rebalancing =="
python -m src.train --model resnet50 --epochs 18 --loss focal \
    --focal-gamma 2.0 --balance-sampler --subtle-weight 2.5 \
    --tampered-weight 1.5 --batch-size 96 --workers 8 --run-name focal-subtle

echo "== 4/7 focal evaluation + FAR delta report =="
python evaluate.py --checkpoint runs/focal-subtle/best.pt --choose-threshold val \
    --json reports/eval_focal.json
python -m src.report --baseline reports/eval_baseline.json \
    --focal reports/eval_focal.json

echo "== 5/7 backbone comparison =="
python -m src.train --model efficientnet_b0 --epochs 18 --run-name effnet-b0
python -m src.train --model custom_cnn --epochs 25 --lr 1e-3 --run-name custom-cnn

echo "== 6/7 analysis =="
python -m src.sweep_threshold --checkpoint runs/baseline-ce/best.pt --wandb
python -m src.confusion_analysis --checkpoint runs/baseline-ce/best.pt
python -m src.gradcam --checkpoint runs/baseline-ce/best.pt
python -m src.gradcam --checkpoint runs/focal-subtle/best.pt

echo "== 7/7 latency =="
python -m src.benchmark_latency --cpu --wandb

echo "== verification =="
python tests/verify_pipeline.py
