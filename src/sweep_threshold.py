"""Threshold sweep + precision-recall / FAR curves.

    python -m src.sweep_threshold --checkpoint runs/baseline-ce/best.pt

Writes reports/threshold_sweep.csv and reports/figures/pr_far_curves.png
(and logs the curves to W&B when available).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augmentations import eval_transform, noisy_eval_transform
from src.dataset import filter_split, load_manifest
from src.evaluate import load_checkpoint, predict_probs
from src.metrics import compute_metrics


def sweep(y_true: np.ndarray, y_prob: np.ndarray,
          thresholds: np.ndarray) -> list[dict]:
    return [compute_metrics(y_true, y_prob, float(t)) for t in thresholds]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    img_size = args.img_size or cfg.get("img_size", 224)

    records = load_manifest(os.path.join(args.data, "manifest.jsonl"))
    test = filter_split(records, "test")

    curves = {}
    for slice_name, tf in [("clean", eval_transform(img_size)),
                           ("noisy", noisy_eval_transform(img_size, seed=123))]:
        prob, true = predict_probs(model, test, args.data, tf, device, amp)
        curves[slice_name] = (true, prob)

    # hardest slice: subtle forgeries only + all genuine
    adv = [r for r in test if r["label"] == 0 or r["difficulty"] == "subtle"]
    prob, true = predict_probs(model, adv, args.data, eval_transform(img_size),
                               device, amp)
    curves["adversarial(subtle)"] = (true, prob)

    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    thr_grid = np.arange(0.01, 1.0, 0.01)

    # ---- CSV ----
    csv_path = os.path.join(args.out_dir, "threshold_sweep.csv")
    with open(csv_path, "w") as f:
        cols = ["slice", "threshold", "precision", "recall", "f1", "far",
                "frr", "accuracy"]
        f.write(",".join(cols) + "\n")
        rows = []
        for sname, (y, p) in curves.items():
            for row in sweep(y, p, thr_grid):
                rows.append([sname] + [row[c] for c in cols[1:]])
                f.write(",".join(f"{v}" if isinstance(v, str) else f"{v:.6f}"
                                 for v in rows[-1]) + "\n")

    # ---- figures ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for sname, (y, p) in curves.items():
        rows = sweep(y, p, thr_grid)
        pre = [r["precision"] for r in rows]
        rec = [r["recall"] for r in rows]
        far = [r["far"] for r in rows]
        axes[0].plot(rec, pre, label=sname)
        axes[1].plot(thr_grid, far, label=sname)
        axes[2].plot(thr_grid, [r["f1"] for r in rows], label=sname)
    axes[0].set_xlabel("recall (tampered)"); axes[0].set_ylabel("precision")
    axes[0].set_title("Precision-Recall"); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("threshold"); axes[1].set_ylabel("FAR")
    axes[1].set_title("False Accept Rate vs threshold"); axes[1].grid(alpha=0.3)
    axes[2].set_xlabel("threshold"); axes[2].set_ylabel("F1")
    axes[2].set_title("F1 vs threshold"); axes[2].grid(alpha=0.3)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "figures", "pr_far_curves.png")
    fig.savefig(fig_path, dpi=140)
    print(f"wrote {csv_path}\nwrote {fig_path}")

    # best operating points per slice (max F1)
    summary = {}
    for sname, (y, p) in curves.items():
        rows = sweep(y, p, thr_grid)
        best = max(rows, key=lambda r: r["f1"])
        summary[sname] = {"best_f1": best["f1"], "threshold": best["threshold"],
                          "far_at_best_f1": best["far"],
                          "far_at_0.5": compute_metrics(y, p, 0.5)["far"]}
        print(f"{sname:20s} best F1={best['f1']:.4f} @ thr={best['threshold']:.2f} "
              f"(FAR={best['far']:.4f}) | FAR@0.5={summary[sname]['far_at_0.5']:.4f}")
    with open(os.path.join(args.out_dir, "threshold_sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if args.wandb:
        from src.wandb_utils import init_wandb
        run = init_wandb({"checkpoint": args.checkpoint},
                         run_name="threshold-sweep", tags=["sweep"])
        if run:
            import wandb
            for sname, (y, p) in curves.items():
                rows = sweep(y, p, thr_grid)
                table = wandb.Table(columns=list(rows[0].keys()),
                                    data=[[r[k] for k in rows[0]] for r in rows])
                run.log({f"sweep/{sname}": table})
            run.finish()


if __name__ == "__main__":
    main()
