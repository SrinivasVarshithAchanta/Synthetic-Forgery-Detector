"""Evaluation harness: three test slices, precision/recall/F1/FAR.

Slices
------
clean        all test images as rendered
noisy        same images through a fixed-seed phone-capture pipeline
adversarial  genuine test images + the hardest tampered subset
             (subtle-difficulty forgeries of the selected hard types)

Usage
-----
    python evaluate.py --checkpoint runs/baseline-ce/best.pt
    python evaluate.py --checkpoint runs/baseline-ce/best.pt --threshold 0.35 \
        --json reports/eval_baseline.json
    python evaluate.py --checkpoint ... --choose-threshold val
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augmentations import eval_transform, noisy_eval_transform
from src.dataset import ForgeryDataset, filter_split, load_manifest
from src.metrics import compute_metrics, format_metrics
from src.models import create_model


def predict_probs(model, records, root, transform, device, amp,
                  batch_size=64, workers=4) -> tuple[np.ndarray, np.ndarray]:
    """Softmax P(tampered) per record (order preserved) + true labels."""
    ds = ForgeryDataset(records, root, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
            probs.append(logits.float().softmax(1)[:, 1].cpu().numpy())
            labels.append(batch["label"].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def load_checkpoint(path: str, device: torch.device):
    """Returns (model, train_config, raw_checkpoint)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    name = config.get("model", "resnet50")
    model = create_model(name, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, ckpt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--choose-threshold", choices=["val"], default=None,
                    help="pick threshold on val maximizing F1, then evaluate")
    ap.add_argument("--img-size", type=int, default=None,
                    help="defaults to the training img size")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hard-types", default="all",
                    help="comma list of forgery types for the adversarial "
                         "slice, or 'all'")
    ap.add_argument("--json", default=None, help="write full report here")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"

    model, train_config, ckpt = load_checkpoint(args.checkpoint, device)
    img_size = args.img_size or train_config.get("img_size", 224)
    print(f"checkpoint={args.checkpoint}  model={train_config.get('model')} "
          f"img_size={img_size} amp={amp}")

    records = load_manifest(os.path.join(args.data, "manifest.jsonl"))
    test = filter_split(records, "test")
    val = filter_split(records, "val")

    threshold = args.threshold
    if args.choose_threshold == "val":
        v_prob, v_true = predict_probs(model, val, args.data,
                                       eval_transform(img_size), device, amp,
                                       args.batch_size, args.workers)
        grid = np.arange(0.01, 0.995, 0.005)
        f1s = [compute_metrics(v_true, v_prob, t)["f1"] for t in grid]
        threshold = float(grid[int(np.argmax(f1s))])
        print(f"chosen threshold (max val F1={max(f1s):.4f}): {threshold:.3f}")

    # adversarial slice membership
    if args.hard_types == "all":
        hard_types = {"copy_move", "splicing", "resampling", "jpeg_recompression"}
    else:
        hard_types = {t.strip() for t in args.hard_types.split(",") if t.strip()}
    adv_records = [r for r in test
                   if r["label"] == 0 or
                   (r["difficulty"] == "subtle" and r["forgery_type"] in hard_types)]

    slices = {
        "clean": (test, eval_transform(img_size)),
        "noisy": (test, noisy_eval_transform(img_size, seed=123)),
        "adversarial": (adv_records, eval_transform(img_size)),
    }

    report = {"checkpoint": args.checkpoint, "model": train_config.get("model"),
              "img_size": img_size, "threshold": threshold,
              "amp": amp, "hard_types": sorted(hard_types), "slices": {}}
    clean_prob = clean_true = None
    for name, (recs, tf) in slices.items():
        prob, true = predict_probs(model, recs, args.data, tf, device, amp,
                                   args.batch_size, args.workers)
        m = compute_metrics(true, prob, threshold)
        report["slices"][name] = m
        if name == "clean":
            clean_prob, clean_true = prob, true
        if not args.quiet:
            print(format_metrics(m, title=f"--- slice: {name} ---"))

    # ---- error distribution: FAR by forgery type x difficulty + subtle FAR ----
    pred = (clean_prob >= threshold).astype(int)
    buckets = []
    if not args.quiet:
        print("--- FAR by forgery type x difficulty (clean slice) ---")
        print(f"  {'forgery_type':20s} {'difficulty':11s} {'n':>4} {'missed':>6} "
              f"{'FAR':>7} {'mean_p':>7}")
    for ft in ["copy_move", "splicing", "resampling", "jpeg_recompression"]:
        for d in ["subtle", "moderate", "aggressive"]:
            idx = [i for i, r in enumerate(test)
                   if r["forgery_type"] == ft and r["difficulty"] == d]
            if not idx:
                continue
            missed = int((pred[idx] == 0).sum())
            b = {"forgery_type": ft, "difficulty": d, "n": len(idx),
                 "missed": missed, "far": missed / len(idx),
                 "mean_p_tampered": float(clean_prob[idx].mean())}
            buckets.append(b)
            if not args.quiet:
                print(f"  {ft:20s} {d:11s} {b['n']:4d} {b['missed']:6d} "
                      f"{b['far']:6.3f} {b['mean_p_tampered']:7.3f}")
    subtle_idx = [i for i, r in enumerate(test)
                  if r["label"] == 1 and r["difficulty"] == "subtle"]
    far_subtle = float((pred[subtle_idx] == 0).mean()) if subtle_idx else float("nan")
    report["far_by_bucket"] = buckets
    report["far_subtle"] = far_subtle
    report["far_aggressive"] = float(
        (pred[[i for i, r in enumerate(test)
               if r["label"] == 1 and r["difficulty"] == "aggressive"]] == 0).mean())
    if not args.quiet:
        print(f"  FAR(subtle tampered)={far_subtle:.4f}  "
              f"FAR(aggressive)={report['far_aggressive']:.4f}")

    overall_acc = report["slices"]["clean"]["accuracy"]
    report["overall_test_accuracy"] = overall_acc
    print(f"\nOVERALL TEST ACCURACY: {overall_acc:.4f} "
          f"({overall_acc * 100:.2f}%)  @ threshold {threshold:.2f}")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
