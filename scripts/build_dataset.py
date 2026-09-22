#!/usr/bin/env python
"""Build the full synthetic forgery dataset.

Pipeline
--------
1. Render N clean mock identity documents (procedural, PII-free).
2. Split *sources* (not images) 70/15/15 train/val/test with a fixed seed,
   so no clean source leaks across splits.
3. For every source emit one genuine image and one tampered image
   (forgery type round-robined, difficulty sampled 40/35/25).
4. Write images, masks, `manifest.jsonl`, `splits.json`, `stats.json`.

Usage
-----
    python scripts/build_dataset.py --sources 3000 --seed 42 --out data
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zlib

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mock_ids import generate_mock_id, WIDTH, HEIGHT  # noqa: F401
from src.forgery_generator import (ForgeryGenerator, ForgeryConfig,
                                   FORGERY_TYPES, LABEL_GENUINE, LABEL_TAMPERED)
from src.dataset import composition_table

JPEG_Q = 93
DIFF_PROBS = [("subtle", 0.40), ("moderate", 0.35), ("aggressive", 0.25)]


def sample_difficulty(rng: np.random.Generator) -> str:
    p = rng.random()
    acc = 0.0
    for name, w in DIFF_PROBS:
        acc += w
        if p <= acc:
            return name
    return DIFF_PROBS[-1][0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=int, default=3000,
                    help="number of clean source documents")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--jpeg-quality", type=int, default=JPEG_Q)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.out
    clean_dir = os.path.join(out, "clean")
    os.makedirs(clean_dir, exist_ok=True)

    # ---------------------------------------------------------------- 1. clean
    print(f"[1/4] rendering {args.sources} clean documents ...")
    source_ids: list[str] = []
    t0 = time.time()
    for i in tqdm(range(args.sources), ncols=90):
        sid = f"sid{i:05d}"
        mock = generate_mock_id(rng)
        path = os.path.join(clean_dir, f"{sid}.jpg")
        cv2.imwrite(path, mock.image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        source_ids.append(sid)
    print(f"      done in {time.time() - t0:.1f}s "
          f"({args.sources / max(time.time() - t0, 1e-9):.1f} docs/s)")

    # ----------------------------------------------------------------- 2. split
    order = rng.permutation(args.sources)
    n_test = int(round(args.sources * args.test_frac))
    n_val = int(round(args.sources * args.val_frac))
    test_ids = [source_ids[i] for i in order[:n_test]]
    val_ids = [source_ids[i] for i in order[n_test:n_test + n_val]]
    train_ids = [source_ids[i] for i in order[n_test + n_val:]]
    splits = {"train": sorted(train_ids), "val": sorted(val_ids),
              "test": sorted(test_ids)}
    with open(os.path.join(out, "splits.json"), "w") as f:
        json.dump({"seed": args.seed, "unit": "clean source document",
                   "fractions": {"train": 1 - args.val_frac - args.test_frac,
                                 "val": args.val_frac,
                                 "test": args.test_frac},
                   "splits": splits}, f, indent=2)
    print(f"[2/4] source-level split -> train {len(train_ids)} / "
          f"val {len(val_ids)} / test {len(test_ids)}")

    # ------------------------------------------------------------- 3. forgeries
    gen = ForgeryGenerator(ForgeryConfig(seed=args.seed,
                                         jpeg_baseline_quality=args.jpeg_quality))
    records: list[dict] = []
    manifest_path = os.path.join(out, "manifest.jsonl")
    print("[3/4] generating genuine + tampered images ...")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for split in ["train", "val", "test"]:
            ids = splits[split]
            img_dir = os.path.join(out, "images", split)
            mask_dir = os.path.join(out, "masks", split)
            for sub in ("genuine", "tampered"):
                os.makedirs(os.path.join(img_dir, sub), exist_ok=True)
                os.makedirs(os.path.join(mask_dir, sub), exist_ok=True)

            # shuffle type assignment so each split gets even type counts
            type_order = rng.permutation(len(FORGERY_TYPES))
            for j, sid in enumerate(tqdm(ids, ncols=90, desc=f"  {split}")):
                # genuine: identical to the clean render
                rel_g = f"images/{split}/genuine/{sid}.jpg"
                shutil.copyfile(os.path.join(clean_dir, f"{sid}.jpg"),
                                os.path.join(out, rel_g))
                rec = {"image": rel_g, "mask": None,
                       "label": LABEL_GENUINE, "forgery_type": None,
                       "difficulty": None, "source_id": sid, "split": split}
                records.append(rec)
                mf.write(json.dumps(rec) + "\n")

                # tampered
                ftype = FORGERY_TYPES[type_order[j % len(FORGERY_TYPES)]]
                diff = sample_difficulty(rng)
                clean = cv2.imread(os.path.join(clean_dir, f"{sid}.jpg"))
                # donor: a different source *within the same split* (no leakage)
                donor_ids = [d for d in ids if d != sid]
                donor_sid = donor_ids[int(rng.integers(0, len(donor_ids)))]
                donor = cv2.imread(os.path.join(clean_dir, f"{donor_sid}.jpg"))
                # stable per-record seed (PYTHONHASHSEED-independent)
                seed = zlib.crc32(f"{sid}|{ftype}|{diff}".encode()) % (1 << 32)
                res = gen.generate(clean, forgery_type=ftype, difficulty=diff,
                                   donor=donor, rois=None, seed=seed)
                name = f"{sid}__{ftype}__{diff}"
                rel_i = f"images/{split}/tampered/{name}.jpg"
                rel_m = f"masks/{split}/tampered/{name}.png"
                cv2.imwrite(os.path.join(out, rel_i), res.image,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                cv2.imwrite(os.path.join(out, rel_m), res.mask)
                rec = {"image": rel_i, "mask": rel_m,
                       "label": LABEL_TAMPERED, "forgery_type": ftype,
                       "difficulty": diff, "source_id": sid, "split": split,
                       "meta": res.meta}
                records.append(rec)
                mf.write(json.dumps(rec) + "\n")

    # ---------------------------------------------------------------- 4. stats
    table = composition_table(records)
    print("[4/4] composition:\n" + table)
    stats = {
        "n_images": len(records),
        "n_sources": args.sources,
        "seed": args.seed,
        "jpeg_quality": args.jpeg_quality,
        "image_size": [WIDTH, HEIGHT],
        "composition": table,
    }
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    readme = f"""# Dataset

Generated by `scripts/build_dataset.py --sources {args.sources} --seed {args.seed}`.

- {args.sources} procedurally rendered clean mock ID documents (**no real PII**),
  split **at source level** 70/15/15 (train/val/test) so no clean document
  appears in more than one split (`splits.json`, seed {args.seed}).
- Each source contributes **1 genuine + 1 tampered** image → {len(records)} images total.
- Tampered images: 4 forgery types (copy-move, splicing, resampling,
  JPEG recompression) round-robined per split; difficulty sampled
  subtle 40% / moderate 35% / aggressive 25%.
- Binary labels: 0 = genuine, 1 = tampered. Per-image forgery masks under `masks/`.
- All images encoded at JPEG quality {args.jpeg_quality}.

```
{table}
```
"""
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"\nwrote {manifest_path} ({len(records)} records) -> {out}/")


if __name__ == "__main__":
    main()
