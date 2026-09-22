"""Confusion analysis: bucket errors by forgery type x difficulty.

    python -m src.confusion_analysis --checkpoint runs/baseline-ce/best.pt

Prints an error-rate table per forgery type / difficulty, FRR on genuine
images, and exports the hardest failure cases to reports/hard_cases/ for
manual review.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augmentations import eval_transform
from src.dataset import filter_split, load_manifest
from src.evaluate import load_checkpoint, predict_probs
from src.metrics import compute_metrics

TYPES = ["copy_move", "splicing", "resampling", "jpeg_recompression"]
DIFFS = ["subtle", "moderate", "aggressive"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=8, help="hard cases to export")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    device = __import__("torch").device(
        "cuda" if __import__("torch").cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    img_size = args.img_size or cfg.get("img_size", 224)
    amp = device.type == "cuda"

    records = load_manifest(os.path.join(args.data, "manifest.jsonl"))
    test = filter_split(records, "test")
    prob, true = predict_probs(model, test, args.data, eval_transform(img_size),
                               device, amp)
    pred = (prob >= args.threshold).astype(int)
    t = args.threshold

    # ------------------------------------------------ error buckets
    print(f"\n=== error buckets @ threshold {t:.2f} ===")
    print(f"{'forgery_type':20s} {'difficulty':11s} {'n':>4} {'missed':>6} "
          f"{'error%':>7} {'mean p(tampered)':>16}")
    rows = []
    for ft in TYPES:
        for d in DIFFS:
            idx = [i for i, r in enumerate(test)
                   if r["forgery_type"] == ft and r["difficulty"] == d]
            if not idx:
                continue
            missed = [i for i in idx if pred[i] == 0]
            rate = len(missed) / len(idx)
            mp = float(np.mean(prob[idx])) if idx else float("nan")
            rows.append({"forgery_type": ft, "difficulty": d, "n": len(idx),
                         "missed": len(missed), "error_rate": rate,
                         "mean_p_tampered": mp})
            print(f"{ft:20s} {d:11s} {len(idx):4d} {len(missed):6d} "
                  f"{rate*100:6.1f}% {mp:16.3f}")

    genuine_idx = [i for i, r in enumerate(test) if r["label"] == 0]
    fp = [i for i in genuine_idx if pred[i] == 1]
    frr = len(fp) / max(len(genuine_idx), 1)
    rows.append({"forgery_type": "genuine", "difficulty": "-", "n": len(genuine_idx),
                 "missed": len(fp), "error_rate": frr,
                 "mean_p_tampered": float(np.mean(prob[genuine_idx]))})
    print(f"{'genuine (FRR)':20s} {'-':11s} {len(genuine_idx):4d} {len(fp):6d} "
          f"{frr*100:6.1f}% {np.mean(prob[genuine_idx]):16.3f}")

    overall = compute_metrics(true, prob, t)
    print(f"\noverall: acc={overall['accuracy']:.4f} FAR={overall['far']:.4f} "
          f"FRR={overall['frr']:.4f}")

    # ------------------------------------------------ hardest cases
    missed_idx = [i for i in range(len(test))
                  if test[i]["label"] == 1 and pred[i] == 0]
    missed_idx.sort(key=lambda i: prob[i])          # most confident misses
    fp_sorted = sorted(fp, key=lambda i: prob[i], reverse=True)

    hard_dir = os.path.join(args.out, "hard_cases")
    os.makedirs(hard_dir, exist_ok=True)
    for f in os.listdir(hard_dir):
        os.remove(os.path.join(hard_dir, f))

    case_rows = []
    for rank, i in enumerate(missed_idx[:args.top]):
        rec = test[i]
        dst = os.path.join(hard_dir,
                           f"miss_{rank:02d}_{rec['forgery_type']}_"
                           f"{rec['difficulty']}.jpg")
        shutil.copyfile(os.path.join(args.data, rec["image"]), dst)
        case_rows.append({"rank": rank, "kind": "missed_forger",
                          "p_tampered": float(prob[i]),
                          "forgery_type": rec["forgery_type"],
                          "difficulty": rec["difficulty"],
                          "image": rec["image"], "copy": dst})
    for rank, i in enumerate(fp_sorted[:args.top]):
        rec = test[i]
        dst = os.path.join(hard_dir, f"falsealarm_{rank:02d}.jpg")
        shutil.copyfile(os.path.join(args.data, rec["image"]), dst)
        case_rows.append({"rank": rank, "kind": "false_alarm",
                          "p_tampered": float(prob[i]),
                          "forgery_type": None, "difficulty": None,
                          "image": rec["image"], "copy": dst})

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "error_buckets.json"), "w") as f:
        json.dump({"threshold": t, "buckets": rows,
                   "overall": overall,
                   "hardest_cases": case_rows}, f, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'error_buckets.json')}")
    print(f"exported {len(case_rows)} hard cases -> {hard_dir}/")
    for c in case_rows[:10]:
        print(f"  {c['kind']:15s} p={c['p_tampered']:.3f} "
              f"{c['forgery_type']}/{c['difficulty']}")


if __name__ == "__main__":
    main()
