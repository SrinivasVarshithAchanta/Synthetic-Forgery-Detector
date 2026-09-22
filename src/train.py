"""Training entry point: AMP fine-tuning + W&B logging.

Examples
--------
# baseline: cross-entropy, plain class balance
python -m src.train --model resnet50 --epochs 12 --loss ce --run-name baseline-ce

# focal loss + oversampling of subtle forgeries
python -m src.train --model resnet50 --epochs 12 --loss focal --focal-gamma 2.0 \
    --balance-sampler --subtle-weight 3.0 --run-name focal-subtle

Checkpoints + config + history land in `runs/<run_name>/`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.augmentations import train_transform, eval_transform
from src.dataset import (ForgeryDataset, filter_split, load_manifest,
                         weighted_sampler_weights)
from src.losses import build_criterion
from src.metrics import compute_metrics
from src.models import MODEL_NAMES, count_parameters, create_model
from src.wandb_utils import init_wandb


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="resnet50", choices=MODEL_NAMES)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: 2e-4 pretrained backbones, 1e-3 custom")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=float, default=None,
                    help="per-class alpha for tampered class (weight on class 1)")
    ap.add_argument("--class-weight", choices=["none", "inverse"], default="none",
                    help="inverse-frequency loss weighting")
    ap.add_argument("--balance-sampler", action="store_true",
                    help="use a weighted random sampler")
    ap.add_argument("--subtle-weight", type=float, default=1.0,
                    help="sampler boost for subtle tampered images")
    ap.add_argument("--tampered-weight", type=float, default=1.0,
                    help="sampler boost for all tampered images")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--wandb", action="store_true", default=True)
    ap.add_argument("--no-wandb", action="store_false", dest="wandb")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


@torch.no_grad()
def evaluate_loader(model, loader, device, amp: bool, threshold: float = 0.5,
                    max_batches: int | None = None) -> tuple[dict, dict]:
    """Returns (metrics_dict, {'prob':..,'label':..,'idx':..})."""
    model.eval()
    probs, labels, idxs = [], [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"]
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
        probs.append(logits.float().softmax(1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
        idxs.append(batch["idx"].numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels)
    idx_arr = np.concatenate(idxs)
    return (compute_metrics(y_true, y_prob, threshold),
            {"prob": y_prob, "label": y_true, "idx": idx_arr})


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"
    lr = args.lr if args.lr is not None else (1e-3 if args.model == "custom_cnn"
                                              else 2e-4)
    run_name = args.run_name or (
        f"{args.model}-{args.loss}-e{args.epochs}"
        f"{'-focal-g' + str(args.focal_gamma) if args.loss == 'focal' else ''}"
        f"{'-sampler' if args.balance_sampler else ''}-{int(time.time()) % 100000}"
    )
    run_dir = os.path.join(args.out, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config = {**vars(args), "resolved_lr": lr, "run_name": run_name,
              "device": str(device), "amp": amp}
    run = init_wandb(config, run_name=run_name,
                     notes=f"training {args.model} with {args.loss} loss",
                     tags=[args.model, args.loss], enabled=args.wandb)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    # ------------------------------------------------------------- data
    records = load_manifest(os.path.join(args.data, "manifest.jsonl"))
    train_recs = filter_split(records, "train")
    val_recs = filter_split(records, "val")
    print(f"train={len(train_recs)} val={len(val_recs)}")

    train_ds = ForgeryDataset(train_recs, args.data, train_transform(args.img_size))
    val_ds = ForgeryDataset(val_recs, args.data, eval_transform(args.img_size))

    if args.balance_sampler:
        w = weighted_sampler_weights(train_recs, subtle_weight=args.subtle_weight,
                                     tampered_weight=args.tampered_weight)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.workers,
                                  pin_memory=True, persistent_workers=args.workers > 0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True,
                                  persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ------------------------------------------------------------- model
    model = create_model(args.model, pretrained=not args.no_pretrained).to(device)
    print(f"{args.model}: {count_parameters(model)/1e6:.2f}M params")

    # class weights (inverse frequency) for the loss
    class_w = None
    if args.class_weight == "inverse":
        n1 = sum(r["label"] for r in train_recs)
        n0 = len(train_recs) - n1
        total = len(train_recs)
        class_w = torch.tensor([total / (2 * max(n0, 1)), total / (2 * max(n1, 1))])
        print(f"inverse-frequency class weights: {class_w.tolist()}")
    alpha = args.focal_alpha
    if alpha is not None:
        alpha = [1.0 - alpha, alpha]  # [genuine, tampered]
    criterion = build_criterion(args.loss, class_weights=class_w,
                                gamma=args.focal_gamma, alpha=alpha,
                                label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(enabled=amp)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"resumed weights from {args.resume}")

    # subtle tampered validation subset (for per-epoch hard-subset FAR)
    subtle_idx = [i for i, r in enumerate(val_recs)
                  if r["label"] == 1 and r["difficulty"] == "subtle"]

    best = {"val_acc": -1.0, "epoch": -1}
    history_path = os.path.join(run_dir, "history.jsonl")
    with open(history_path, "w") as hist:
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            running, nb = 0.0, 0
            for batch in train_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.detach())
                nb += 1
            scheduler.step()

            val_m, val_raw = evaluate_loader(model, val_loader, device, amp)
            subtle_far = float("nan")
            if subtle_idx:
                pos = {i: j for j, i in enumerate(val_raw["idx"].tolist())}
                sel = [pos[i] for i in subtle_idx if i in pos]
                if sel:
                    sp = val_raw["prob"][sel]
                    sl = val_raw["label"][sel]
                    subtle_far = compute_metrics(sl, sp)["far"]

            lr_now = optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch, "train_loss": running / max(nb, 1),
                   "lr": lr_now, "val_loss": None,
                   "val_acc": val_m["accuracy"], "val_f1": val_m["f1"],
                   "val_far": val_m["far"], "val_frr": val_m["frr"],
                   "val_far_subtle": subtle_far,
                   "val_auroc": val_m["auroc"],
                   "seconds": time.time() - t0}
            hist.write(json.dumps(row) + "\n")
            hist.flush()
            print(f"ep {epoch:02d}/{args.epochs} loss={row['train_loss']:.4f} "
                  f"acc={val_m['accuracy']:.4f} f1={val_m['f1']:.4f} "
                  f"FAR={val_m['far']:.4f} FAR_subtle={subtle_far:.4f} "
                  f"({row['seconds']:.0f}s)")
            if run:
                run.log({**row, "epoch": epoch})

            if val_m["accuracy"] > best["val_acc"]:
                best = {"val_acc": val_m["accuracy"], "epoch": epoch}
                torch.save({"model": model.state_dict(), "config": config,
                            "epoch": epoch, "val_metrics": val_m},
                           os.path.join(run_dir, "best.pt"))

    torch.save({"model": model.state_dict(), "config": config,
                "epoch": args.epochs},
               os.path.join(run_dir, "last.pt"))
    summary = {"best_val_acc": best["val_acc"], "best_epoch": best["epoch"],
               "run_name": run_name,
               "checkpoint": os.path.join(run_dir, "best.pt")}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"best val acc {best['val_acc']:.4f} @ epoch {best['epoch']} "
          f"-> {run_dir}/best.pt")
    if run:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
