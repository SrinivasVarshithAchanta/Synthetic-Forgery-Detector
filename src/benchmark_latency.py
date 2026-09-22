"""Inference latency benchmark (GPU + CPU), AMP vs FP32, across settings.

    python -m src.benchmark_latency --checkpoints runs/baseline-ce/best.pt

Benchmarks every trained checkpoint we can find under runs/ plus the
custom CNN at several input resolutions. Batch=1, warmup then timed
repeats using CUDA events; reports median / p95 per image in ms and logs
the table to W&B.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augmentations import eval_transform
from src.models import MODEL_NAMES, create_model


def bench(model: torch.nn.Module, size: int, device: torch.device,
          amp: bool, n: int = 100, warmup: int = 25) -> dict:
    model = model.to(device).eval()
    x = torch.randn(1, 3, size, size, device=device)
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            with torch.amp.autocast("cuda", enabled=amp and use_cuda):
                model(x)
        if use_cuda:
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(n):
                with torch.amp.autocast("cuda", enabled=amp):
                    model(x)
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end) / n
        else:
            t0 = time.perf_counter()
            for _ in range(n):
                with torch.amp.autocast("cuda", enabled=False):
                    model(x)
            total_ms = (time.perf_counter() - t0) * 1000 / n

    # per-image latency distribution (CPU timing loop, works for both devices)
    lat = []
    with torch.no_grad():
        for _ in range(n):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=amp and use_cuda):
                model(x)
            if use_cuda:
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    return {"median_ms": float(np.median(lat)), "mean_ms": float(lat.mean()),
            "p95_ms": float(np.percentile(lat, 95)),
            "batched_avg_ms": float(total_ms), "n": n}


def load_or_init(name: str, checkpoint: str | None) -> torch.nn.Module:
    if checkpoint and os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location="cpu")
        cfg = ckpt.get("config", {})
        mname = cfg.get("model", name)
        model = create_model(mname, pretrained=False)
        model.load_state_dict(ckpt["model"])
        return model, cfg.get("img_size", 224)
    return create_model(name, pretrained=False), 224


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="runs", help="directory of training runs")
    ap.add_argument("--checkpoints", nargs="*", default=[],
                    help="explicit checkpoints to benchmark")
    ap.add_argument("--sizes", default="160,224,294",
                    help="input resolutions to sweep")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cpu", action="store_true", help="also benchmark CPU")
    ap.add_argument("--out", default="reports/latency.json")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")

    # discover: explicit ckpts, then runs/*/best.pt mapping to their model
    jobs: list[tuple[str, str | None, str]] = []  # (model, ckpt, tag)
    for ck in args.checkpoints:
        cfg = torch.load(ck, map_location="cpu").get("config", {})
        jobs.append((cfg.get("model", "resnet50"), ck,
                     os.path.basename(os.path.dirname(os.path.abspath(ck)))))
    if not jobs:
        for best in sorted(glob.glob(os.path.join(args.runs, "*", "best.pt"))):
            cfg = torch.load(best, map_location="cpu").get("config", {})
            tag = os.path.basename(os.path.dirname(best))
            jobs.append((cfg.get("model", "resnet50"), best, tag))
    for m in MODEL_NAMES:  # every backbone, even untrained (latency-neutral)
        if not any(j[0] == m for j in jobs):
            jobs.append((m, None, f"{m}-untrained"))

    sizes = [int(s) for s in args.sizes.split(",") if s]
    rows = []
    for model_name, ckpt, tag in jobs:
        for size in sizes:
            for amp in ([False, True] if device.type == "cuda" else [False]):
                model, _ = load_or_init(model_name, ckpt)
                r = bench(model, size, device, amp=amp, n=args.n)
                row = {"tag": tag, "model": model_name, "size": size,
                       "precision": "amp" if amp else "fp32",
                       "device": str(device), **r,
                       "under_60ms": r["median_ms"] < 60}
                rows.append(row)
                print(f"{tag:26s} {size:4d}px {row['precision']:4s} "
                      f"median={r['median_ms']:6.1f}ms p95={r['p95_ms']:6.1f}ms "
                      f"{'OK<60ms' if row['under_60ms'] else 'SLOW'}")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if args.cpu and device.type == "cuda":
        cpu = torch.device("cpu")
        for model_name, ckpt, tag in jobs:
            for size in sizes:
                model, _ = load_or_init(model_name, ckpt)
                r = bench(model, size, cpu, amp=False, n=max(args.n // 5, 20))
                row = {"tag": tag, "model": model_name, "size": size,
                       "precision": "fp32", "device": "cpu", **r,
                       "under_60ms": r["median_ms"] < 60}
                rows.append(row)
                print(f"{tag:26s} {size:4d}px CPU  median={r['median_ms']:6.1f}ms "
                      f"p95={r['p95_ms']:6.1f}ms")
                del model

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {args.out}")

    # markdown table for the report
    md = ["| tag | model | res | precision | device | median ms | p95 ms | <60ms |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['tag']} | {r['model']} | {r['size']} | {r['precision']} "
                  f"| {r['device']} | {r['median_ms']:.1f} | {r['p95_ms']:.1f} "
                  f"| {'yes' if r['under_60ms'] else 'no'} |")
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {md_path}")

    if args.wandb:
        from src.wandb_utils import init_wandb
        run = init_wandb({"n": args.n}, run_name="latency-bench",
                         tags=["latency"])
        if run:
            import wandb
            run.log({"latency": wandb.Table(
                columns=list(rows[0].keys()),
                data=[[r[k] for k in rows[0]] for r in rows])})
            run.finish()


if __name__ == "__main__":
    main()
