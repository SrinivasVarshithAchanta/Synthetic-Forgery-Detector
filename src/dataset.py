"""Dataset / manifest utilities."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_manifest(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_split(records: list[dict], split: str) -> list[dict]:
    return [r for r in records if r["split"] == split]


def class_counts(records: list[dict]) -> dict:
    out = {"genuine": 0, "tampered": 0}
    for r in records:
        out["genuine" if r["label"] == 0 else "tampered"] += 1
    return out


def composition_table(records: list[dict]) -> str:
    """Pretty composition table by split / class / forgery type / difficulty."""
    splits = sorted({r["split"] for r in records})
    types = ["copy_move", "splicing", "resampling", "jpeg_recompression"]
    diffs = ["subtle", "moderate", "aggressive"]
    header = f"{'split':6} {'genuine':>8} {'tampered':>9} | forgery types"
    header += "  " + "  ".join(t[:9].ljust(9) for t in types)
    lines = [header]
    for sp in splits:
        rs = [r for r in records if r["split"] == sp]
        g = sum(1 for r in rs if r["label"] == 0)
        t = sum(1 for r in rs if r["label"] == 1)
        by_t = [sum(1 for r in rs if r["forgery_type"] == ty) for ty in types]
        row = f"{sp:6} {g:8d} {t:9d} | "
        row += "  ".join(str(v).ljust(9) for v in by_t)
        lines.append(row)
    lines.append("")
    lines.append("difficulty (tampered):")
    for sp in splits:
        rs = [r for r in records if r["split"] == sp and r["label"] == 1]
        by_d = [sum(1 for r in rs if r["difficulty"] == d) for d in diffs]
        lines.append(f"  {sp:6} " + "  ".join(f"{d}={v}" for d, v in zip(diffs, by_d)))
    return "\n".join(lines)


class ForgeryDataset(Dataset):
    """Serves (image tensor, label) pairs from manifest records.

    `transform` is an albumentations pipeline. `records` order defines the
    index order, so metadata stays aligned for evaluation / error analysis.
    """

    def __init__(self, records: list[dict], root: str, transform):
        self.records = records
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> list[int]:
        return [r["label"] for r in self.records]

    def image_path(self, idx: int) -> str:
        return os.path.join(self.root, self.records[idx]["image"])

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = cv2.imread(os.path.join(self.root, rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(rec["image"])
        out = self.transform(image=img)["image"]
        return {
            "image": out,
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "idx": idx,
        }


def weighted_sampler_weights(records: list[dict],
                             subtle_weight: float = 1.0,
                             tampered_weight: float = 1.0) -> np.ndarray:
    """Per-sample sampler weights with boosts for tampered / subtle edits."""
    w = np.ones(len(records), dtype=np.float64)
    for i, r in enumerate(records):
        if r["label"] == 1:
            w[i] *= tampered_weight
            if r["difficulty"] == "subtle":
                w[i] *= subtle_weight
    return w
