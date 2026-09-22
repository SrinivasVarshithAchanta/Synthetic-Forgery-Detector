#!/usr/bin/env python
"""End-to-end verification tests for the forgery detection pipeline.

Run with either:

    python tests/verify_pipeline.py
    pytest -q tests/verify_pipeline.py

Each check prints PASS/FAIL with the measured numbers - the process exits
non-zero if any check fails.  Tests marked [requires checkpoint] are
skipped when no trained model / report exists yet.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results = []


def record(name: str, status: str, detail: str = "") -> None:
    _results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


# ----------------------------------------------------------------- dataset
def test_dataset_manifest():
    name = "dataset: manifest exists, balanced, splits disjoint"
    path = os.path.join(DATA, "manifest.jsonl")
    if not os.path.exists(path):
        record(name, SKIP, "data/ not built (run scripts/build_dataset.py)")
        return
    recs = [json.loads(l) for l in open(path)]
    n = len(recs)
    n_pos = sum(r["label"] for r in recs)
    splits = {}
    for r in recs:
        splits.setdefault(r["split"], set()).add(r["source_id"])
    disjoint = (splits["train"].isdisjoint(splits["val"])
                and splits["train"].isdisjoint(splits["test"])
                and splits["val"].isdisjoint(splits["test"]))
    ok = n >= 2000 and n_pos * 2 == n and disjoint
    record(name, PASS if ok else FAIL,
           f"n={n}, tampered={n_pos}, source-disjoint={disjoint}")
    return recs


def test_images_load():
    name = "dataset: sample images + masks load at expected size"
    path = os.path.join(DATA, "manifest.jsonl")
    if not os.path.exists(path):
        record(name, SKIP, "data/ not built")
        return
    recs = [json.loads(l) for l in open(path)]
    rng = np.random.default_rng(0)
    picks = [recs[i] for i in rng.choice(len(recs), 30, replace=False)]
    bad = []
    for r in picks:
        img = cv2.imread(os.path.join(DATA, r["image"]))
        if img is None or img.shape[:2] != (480, 768):
            bad.append(r["image"])
        if r["label"] == 1:
            m = cv2.imread(os.path.join(DATA, r["mask"]), 0)
            frac = m.mean() / 255.0 if m is not None else 0
            if m is None or not (0.01 <= frac <= 0.45):
                bad.append(f"{r['mask']} (area={frac:.3f})")
    record(name, PASS if not bad else FAIL,
           f"30 sampled, bad={bad[:3]}")
    return recs


# --------------------------------------------------------------- forgery gen
def test_forgery_generator():
    from src.forgery_generator import (ForgeryGenerator, ForgeryConfig,
                                       FORGERY_TYPES, DIFFICULTIES)
    from src.mock_ids import generate_mock_id

    name = "forgery generator: 4 types x 3 difficulties produce valid output"
    rng = np.random.default_rng(0)
    gen = ForgeryGenerator(ForgeryConfig(seed=1))
    clean = generate_mock_id(rng)
    donor = generate_mock_id(rng)
    problems = []
    for ft in FORGERY_TYPES:
        for d in DIFFICULTIES:
            r = gen.generate(clean.image, forgery_type=ft, difficulty=d,
                             donor=donor.image, rois=clean.rois, seed=7)
            area = r.mask.mean() / 255.0
            diff = np.abs(r.image.astype(int) - clean.image.astype(int)).mean()
            if r.mask.shape != clean.image.shape[:2]:
                problems.append(f"{ft}/{d}: bad mask shape")
            if not (0.005 <= area <= 0.45):
                problems.append(f"{ft}/{d}: area {area:.3f} out of range")
            if diff < 0.2:
                problems.append(f"{ft}/{d}: image unchanged (Δ={diff:.3f})")
            if r.forgery_type != ft or r.difficulty != d:
                problems.append(f"{ft}/{d}: label mismatch")
    record(name, PASS if not problems else FAIL,
           "; ".join(problems) if problems else "12/12 combos OK")
    return not problems


# ------------------------------------------------------------------- metrics
def test_metrics_far_formula():
    from src.metrics import compute_metrics

    name = "metrics: FAR = tampered accepted as genuine"
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.6, 0.4, 0.1, 0.9, 0.7, 0.2, 0.1])
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    # tampered: 2 of 4 accepted (>=0.5) -> 2 missed -> FAR 0.5
    ok = abs(m["far"] - 0.5) < 1e-9 and m["fn"] == 2 and abs(m["frr"] - 0.5) < 1e-9
    record(name, PASS if ok else FAIL,
           f"FAR={m['far']} (expected 0.5), FN={m['fn']}, FRR={m['frr']}")
    return ok


def test_focal_equals_ce_at_gamma0():
    import torch
    from src.losses import FocalLoss, build_criterion

    name = "loss: focal(gamma=0) == cross-entropy"
    torch.manual_seed(0)
    logits = torch.randn(64, 2)
    y = torch.randint(0, 2, (64,))
    ce = build_criterion("ce")(logits, y)
    fl = FocalLoss(gamma=0.0)(logits, y)
    ok = torch.allclose(ce, fl, atol=1e-5)
    record(name, PASS if ok else FAIL, f"ce={ce:.6f} focal0={fl:.6f}")
    return bool(ok)


# ------------------------------------------------------------------- gradcam
def test_gradcam_output():
    import torch
    from src.gradcam import GradCAM
    from src.models import create_model

    name = "gradcam: cam shape, range, non-constant"
    model = create_model("resnet18", pretrained=False).eval()
    engine = GradCAM(model)
    x = torch.randn(1, 3, 224, 224)
    cam, logits = engine(x, class_idx=1)
    engine.remove()
    ok = (cam.shape == (1, 1, 224, 224)
          and 0.0 <= float(cam.min()) and float(cam.max()) <= 1.0
          and float(cam.std()) > 0)
    record(name, PASS if ok else FAIL,
           f"shape={tuple(cam.shape)} range=[{float(cam.min()):.2f},"
           f"{float(cam.max()):.2f}] std={float(cam.std()):.4f}")
    return ok


# -------------------------------------------------- trained-model sanity checks
def _find_checkpoint() -> str | None:
    for tag in ("baseline-ce", "focal-subtle"):
        p = os.path.join(ROOT, "runs", tag, "best.pt")
        if os.path.exists(p):
            return p
    runs = os.path.join(ROOT, "runs")
    if os.path.isdir(runs):
        for d in sorted(os.listdir(runs)):
            p = os.path.join(runs, d, "best.pt")
            if os.path.exists(p):
                return p
    return None


def test_checkpoint_beats_chance():
    name = "model: best checkpoint beats random on held-out test"
    ck = _find_checkpoint()
    if ck is None:
        record(name, SKIP, "no checkpoint (run training first)")
        return None
    import torch
    from src.augmentations import eval_transform
    from src.dataset import filter_split, load_manifest
    from src.evaluate import load_checkpoint, predict_probs
    from src.metrics import compute_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(ck, device)
    recs = load_manifest(os.path.join(DATA, "manifest.jsonl"))
    test = filter_split(recs, "test")
    prob, true = predict_probs(model, test, DATA, eval_transform(224),
                               device, device.type == "cuda")
    m = compute_metrics(true, prob, 0.5)
    ok = m["accuracy"] >= 0.80
    record(name, PASS if ok else FAIL,
           f"acc={m['accuracy']:.4f} FAR={m['far']:.4f} n={m['n']} ckpt={ck}")
    return m


def test_reports_match_recomputed_accuracy():
    name = "reports: RESULTS accuracy matches a fresh evaluation run"
    res_path = os.path.join(ROOT, "reports", "results.json")
    if not os.path.exists(res_path):
        record(name, SKIP, "reports/results.json not built yet")
        return
    from src.augmentations import eval_transform
    from src.dataset import filter_split, load_manifest
    from src.evaluate import load_checkpoint, predict_probs
    from src.metrics import compute_metrics
    import torch

    saved = json.load(open(res_path))
    ck = saved.get("checkpoint")
    if not ck or not os.path.exists(ck):
        record(name, SKIP, f"checkpoint from report missing: {ck}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(ck, device)
    recs = load_manifest(os.path.join(DATA, "manifest.jsonl"))
    test = filter_split(recs, "test")
    prob, true = predict_probs(model, test, DATA, eval_transform(224),
                               device, device.type == "cuda")
    thr = saved["slices"]["clean"].get("threshold", 0.5)
    m = compute_metrics(true, prob, thr)
    delta = abs(m["accuracy"] - saved["slices"]["clean"]["accuracy"])
    ok = delta < 1e-6
    record(name, PASS if ok else FAIL,
           f"recomputed={m['accuracy']:.4f} vs reported="
           f"{saved['slices']['clean']['accuracy']:.4f}")
    return ok


def test_latency_report_under_60ms():
    name = "latency: at least one setting measured below 60 ms median"
    path = os.path.join(ROOT, "reports", "latency.json")
    if not os.path.exists(path):
        record(name, SKIP, "reports/latency.json not built yet")
        return
    rows = json.load(open(path))
    gpu = [r for r in rows if r["device"].startswith("cuda")]
    best = min(gpu, key=lambda r: r["median_ms"]) if gpu else None
    ok = bool(best and best["median_ms"] < 60)
    record(name, PASS if ok else FAIL,
           (f"best={best['tag']} {best['size']}px {best['precision']} "
            f"median={best['median_ms']:.1f}ms") if best else "no GPU rows")
    return ok


ALL_TESTS = [
    test_dataset_manifest,
    test_images_load,
    test_forgery_generator,
    test_metrics_far_formula,
    test_focal_equals_ce_at_gamma0,
    test_gradcam_output,
    test_checkpoint_beats_chance,
    test_reports_match_recomputed_accuracy,
    test_latency_report_under_60ms,
]


def main() -> int:
    if len(sys.argv) > 1:  # allow selecting one test
        wanted = sys.argv[1:]
        tests = [t for t in ALL_TESTS if t.__name__ in wanted]
    else:
        tests = ALL_TESTS
    for t in tests:
        try:
            t()
        except Exception as e:
            traceback.print_exc()
            record(t.__name__, FAIL, f"exception: {e}")
    n_pass = sum(1 for _, s, _ in _results if s == PASS)
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    n_skip = sum(1 for _, s, _ in _results if s == SKIP)
    print(f"\n== {n_pass} passed, {n_fail} failed, {n_skip} skipped ==")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
