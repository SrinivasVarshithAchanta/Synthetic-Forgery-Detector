"""Grad-CAM saliency for the forgery detector.

    python -m src.gradcam --checkpoint runs/baseline-ce/best.pt --n 6

Renders original | Grad-CAM heatmap | overlay for a mix of correctly and
incorrectly classified *tampered* test images, saving individual PNGs plus a
summary grid under reports/figures/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augmentations import IMAGENET_MEAN, IMAGENET_STD, eval_transform
from src.dataset import filter_split, load_manifest
from src.evaluate import load_checkpoint, predict_probs


class GradCAM:
    """Standard Grad-CAM (Selvaraju et al., 2017) on `model._gradcam_target`."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layer = model._gradcam_target
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._fh = self.layer.register_forward_hook(self._fwd)
        self._bh = self.layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self._acts = out.detach()

    def _bwd(self, module, gin, gout):
        self._grads = gout[0].detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cam [B,1,H,W] normalised 0..1, logits)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1)[0].item())
        probs = logits.softmax(1)
        probs[:, class_idx].sum().backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        mn = cam.amin(dim=(2, 3), keepdim=True)
        mx = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - mn) / (mx - mn + 1e-8)
        return cam, logits


def denormalize(x: torch.Tensor) -> np.ndarray:
    img = x.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)


def render_case(img: np.ndarray, cam: np.ndarray, title: str):
    """cam: HxW in 0..1. Returns (fig, overlay_array)."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip(img * 0.55 + heat * 0.45, 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    for ax, im, t in zip(axes, [img, heat, overlay],
                         ["input", "Grad-CAM", "overlay"]):
        ax.imshow(im)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig, overlay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--n-correct", type=int, default=4,
                    help="correctly classified tampered examples")
    ap.add_argument("--n-incorrect", type=int, default=4,
                    help="misclassified tampered examples")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="reports/figures")
    ap.add_argument("--tag", default=None,
                    help="output filename prefix (defaults to checkpoint name)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    img_size = args.img_size or cfg.get("img_size", 224)
    amp = device.type == "cuda"

    records = load_manifest(os.path.join(args.data, "manifest.jsonl"))
    test = filter_split(records, "test")
    tampered_idx = [i for i, r in enumerate(test) if r["label"] == 1]

    prob, true = predict_probs(model, test, args.data, eval_transform(img_size),
                               device, amp)
    correct = [i for i in tampered_idx if (prob[i] >= args.threshold) ==
               (true[i] == 1)]
    correct_set = set(correct)
    wrong = [i for i in tampered_idx if i not in correct_set]
    # hardest first: correct = lowest confidence, wrong = most confident misses
    correct.sort(key=lambda i: prob[i])
    wrong.sort(key=lambda i: prob[i], reverse=True)
    picked = [("correct", i) for i in correct[:args.n_correct]] + \
             [("missed", i) for i in wrong[:args.n_incorrect]]
    if not picked:
        print("no test cases to render")
        return

    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or os.path.basename(os.path.dirname(
        os.path.abspath(args.checkpoint)))
    cam_engine = GradCAM(model)
    tf = eval_transform(img_size)

    rows = []
    overlays = []
    for kind, i in picked:
        rec = test[i]
        raw = cv2.imread(os.path.join(args.data, rec["image"]))
        tensor = torch.unsqueeze(tf(image=raw)["image"], 0).to(device)
        with torch.amp.autocast("cuda", enabled=amp):
            cams, logits = cam_engine(tensor, class_idx=1)
        cam = cams[0, 0].cpu().numpy()
        p = float(logits.softmax(1)[0, 1].item())
        img = denormalize(tensor[0])
        title = (f"{kind} | p(tampered)={p:.3f} | "
                 f"{rec['forgery_type']} / {rec['difficulty']}")
        fig, overlay = render_case(img, cam, title)
        case_path = os.path.join(args.out, f"gradcam_{tag}_{kind}_{i:03d}.png")
        fig.savefig(case_path, dpi=140)
        overlays.append((img, overlay))
        rows.append({"idx": i, "kind": kind, "p_tampered": p,
                     "forgery_type": rec["forgery_type"],
                     "difficulty": rec["difficulty"],
                     "image": rec["image"], "figure": case_path})
        print(f"saved {case_path} ({title})")
        plt.close(fig)

    # summary grid: input | overlay | annotation, one row per case
    if rows:
        n = len(rows)
        grid = plt.figure(figsize=(11.0, 2.6 * n))
        for k, ((img, overlay), r) in enumerate(zip(overlays, rows)):
            ax1 = grid.add_subplot(n, 3, k * 3 + 1)
            ax1.imshow(img); ax1.axis("off")
            ax1.set_title(f"{r['kind']}: {r['forgery_type']}/{r['difficulty']}",
                          fontsize=8)
            ax2 = grid.add_subplot(n, 3, k * 3 + 2)
            ax2.imshow(overlay); ax2.axis("off"); ax2.set_title("Grad-CAM", fontsize=8)
            ax3 = grid.add_subplot(n, 3, k * 3 + 3)
            ax3.axis("off")
            ax3.text(0.02, 0.5,
                     f"p(tampered)={r['p_tampered']:.3f}\n"
                     f"{r['image'].split('/')[-1]}",
                     fontsize=8, va="center", family="monospace")
        grid.tight_layout()
        gpath = os.path.join(args.out, f"gradcam_grid_{tag}.png")
        grid.savefig(gpath, dpi=130)
        plt.close(grid)
        print(f"saved {gpath}")

    with open(os.path.join(args.out, f"gradcam_cases_{tag}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    cam_engine.remove()


if __name__ == "__main__":
    main()
