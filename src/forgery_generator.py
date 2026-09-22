"""Configurable synthetic forgery generator.

Takes a clean document image and produces:
    - tampered image (uint8 BGR)
    - binary mask of the edited region (255 = tampered pixels)
    - forgery type label + difficulty level + parameter metadata

Four forgery types:
    copy_move           duplicate a region within the same document
    splicing            paste a region from a different document
    resampling          locally down/up-scale a region (resampling artifacts)
    jpeg_recompression  region with a mismatched JPEG quantisation grid

Difficulty levels (subtle | moderate | aggressive) control the edited
area, the blend feathering and how extreme the artefacts are.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field, asdict

FORGERY_TYPES = [
    "copy_move",
    "splicing",
    "resampling",
    "jpeg_recompression",
]
DIFFICULTIES = ["subtle", "moderate", "aggressive"]
LABEL_GENUINE, LABEL_TAMPERED = 0, 1

# edited-area fraction of the image per difficulty
_AREA = {
    "subtle": (0.03, 0.07),
    "moderate": (0.08, 0.15),
    "aggressive": (0.16, 0.30),
}
_FEATHER = {"subtle": 9.0, "moderate": 5.0, "aggressive": 2.0}
_JPEG_Q = {"subtle": (68, 85), "moderate": (45, 67), "aggressive": (22, 44)}
_RESAMPLE = {"subtle": (0.85, 0.95), "moderate": (0.65, 0.85), "aggressive": (0.45, 0.65)}
_ROTATION = {"subtle": (0, 8), "moderate": (5, 18), "aggressive": (12, 35)}


@dataclass
class ForgeryConfig:
    seed: int | None = None
    prefer_rois: bool = True          # place edits over semantic regions
    roi_bias: float = 0.7             # P(anchored on an ROI)
    jpeg_baseline_quality: int = 93   # quality of the clean/genuine encoding
    extra: dict = field(default_factory=dict)


@dataclass
class ForgeryResult:
    image: np.ndarray          # tampered BGR image
    mask: np.ndarray           # uint8 HxW, 255 = edited region
    forgery_type: str
    difficulty: str
    meta: dict = field(default_factory=dict)


class ForgeryGenerator:
    """Applies one of four forgery transforms to a clean image."""

    def __init__(self, config: ForgeryConfig | None = None):
        self.cfg = config or ForgeryConfig()

    # ------------------------------------------------------------------ utils
    def _rng(self, seed: int | None = None) -> np.random.Generator:
        if seed is not None:
            return np.random.default_rng(seed)
        if self.cfg.seed is not None:
            return np.random.default_rng(
                np.random.SeedSequence([self.cfg.seed, int(np.random.randint(1 << 31))])
            )
        return np.random.default_rng()

    @staticmethod
    def _pick_region(h: int, w: int, difficulty: str,
                     rois: dict | None, rng: np.random.Generator,
                     roi_bias: float = 0.7) -> tuple[int, int, int, int]:
        """Return (x, y, rw, rh) fully inside the image."""
        lo, hi = _AREA[difficulty]
        frac = rng.uniform(lo, hi)
        aspect = rng.uniform(0.6, 1.7)
        rw = int(np.sqrt(frac * h * w * aspect))
        rh = int((frac * h * w) / max(rw, 1))
        rw, rh = min(rw, w - 8), min(rh, h - 8)
        rw, rh = max(rw, 24), max(rh, 24)

        use_roi = rois and rng.random() < roi_bias
        if use_roi:
            keys = [k for k, v in rois.items() if v[2] > 30 and v[3] > 30]
            if keys:
                k = keys[int(rng.integers(0, len(keys)))]
                rx, ry, rrw, rrh = rois[k]
                # sub-rectangle inside the ROI
                x = int(rng.integers(rx, max(rx + 1, rx + rrw - rw + 1)))
                y = int(rng.integers(ry, max(ry + 1, ry + rrh - rh + 1)))
                x = int(np.clip(x, 0, w - rw))
                y = int(np.clip(y, 0, h - rh))
                return x, y, rw, rh
        x = int(rng.integers(0, w - rw))
        y = int(rng.integers(0, h - rh))
        return x, y, rw, rh

    @staticmethod
    def _feather_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return mask.astype(np.float64)
        k = int(max(3, sigma * 3)) | 1
        return cv2.GaussianBlur(mask.astype(np.float64), (k, k), sigma)

    def _paste(self, img: np.ndarray, patch: np.ndarray,
               x: int, y: int, feather: float,
               rotate: float = 0.0, scale: float = 1.0) -> np.ndarray:
        """Blend `patch` into `img` at (x, y); returns the binary mask."""
        h, w = patch.shape[:2]
        if abs(rotate) > 1e-3 or abs(scale - 1.0) > 1e-3:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate, scale)
            patch = cv2.warpAffine(patch, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
        alpha = np.full((h, w), 255.0)
        alpha = self._feather_mask(alpha, feather)
        # bounding box of the (possibly rotated) patch content
        src = img[y:y + h, x:x + w].astype(np.float64)
        a = (alpha / 255.0)[..., None]
        blended = src * (1 - a) + patch.astype(np.float64) * a
        img[y:y + h, x:x + w] = np.clip(blended, 0, 255).astype(np.uint8)
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[y:y + h, x:x + w] = (alpha > 8).astype(np.uint8) * 255
        return mask

    # ------------------------------------------------------------- forgeries
    def copy_move(self, img: np.ndarray, rois: dict | None,
                  difficulty: str, rng: np.random.Generator):
        h, w = img.shape[:2]
        x, y, rw, rh = self._pick_region(h, w, difficulty, rois, rng,
                                          self.cfg.roi_bias)
        patch = img[y:y + rh, x:x + rw].copy()
        # paste elsewhere
        lo, hi = _ROTATION[difficulty]
        rotate = float(rng.uniform(lo, hi) * rng.choice([-1, 1]))
        scale = float(rng.uniform(0.85, 1.18)) if difficulty != "subtle" \
            else float(rng.uniform(0.95, 1.06))
        for _ in range(20):
            x2 = int(rng.integers(0, w - rw))
            y2 = int(rng.integers(0, h - rh))
            if abs(x2 - x) + abs(y2 - y) > 0.6 * (w + h) * 0.3:
                break
        feather = _FEATHER[difficulty] * float(rng.uniform(0.8, 1.3))
        mask = self._paste(img, patch, x2, y2, feather, rotate, scale)
        return img, mask, {"src_box": [x, y, rw, rh], "dst_box": [x2, y2, rw, rh],
                           "rotation": rotate, "scale": scale}

    def splicing(self, img: np.ndarray, donor: np.ndarray,
                 rois: dict | None, difficulty: str,
                 rng: np.random.Generator):
        h, w = img.shape[:2]
        x, y, rw, rh = self._pick_region(h, w, difficulty, rois, rng,
                                          self.cfg.roi_bias)
        # donor region (its own semantic ROI when possible)
        d_h, d_w = donor.shape[:2]
        dx, dy, drw, drh = self._pick_region(d_h, d_w, difficulty, None, rng, 0.0)
        patch = donor[dy:dy + drh, dx:dx + drw]
        if patch.shape[:2] != (rh, rw):
            patch = cv2.resize(patch, (rw, rh), interpolation=cv2.INTER_LINEAR)
        feather = _FEATHER[difficulty] * float(rng.uniform(0.8, 1.3))
        mask = self._paste(img, patch.copy(), x, y, feather)
        return img, mask, {"dst_box": [x, y, rw, rh], "donor_box": [dx, dy, drw, drh]}

    def resampling(self, img: np.ndarray, rois: dict | None,
                   difficulty: str, rng: np.random.Generator):
        h, w = img.shape[:2]
        x, y, rw, rh = self._pick_region(h, w, difficulty, rois, rng,
                                          self.cfg.roi_bias)
        lo, hi = _RESAMPLE[difficulty]
        f = float(rng.uniform(lo, hi))
        region = img[y:y + rh, x:x + rw]
        if rng.random() < 0.5:
            # down-up: anti-aliasing blur + interpolated grid
            small = cv2.resize(region, (max(2, int(rw * f)), max(2, int(rh * f))),
                               interpolation=cv2.INTER_AREA)
            out = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
        else:
            # up-down: interpolation overshoot then decimation
            big = cv2.resize(region, (int(rw / f), int(rh / f)),
                             interpolation=cv2.INTER_CUBIC)
            out = cv2.resize(big, (rw, rh), interpolation=cv2.INTER_NEAREST
                             if rng.random() < 0.3 else cv2.INTER_AREA)
        feather = _FEATHER[difficulty] * 0.7
        alpha = self._feather_mask(np.full((rh, rw), 255.0), feather) / 255.0
        blended = region.astype(np.float64) * (1 - alpha[..., None]) \
            + out.astype(np.float64) * alpha[..., None]
        img[y:y + rh, x:x + rw] = np.clip(blended, 0, 255).astype(np.uint8)
        mask = np.zeros((h, w), np.uint8)
        mask[y:y + rh, x:x + rw] = (alpha > 0.03).astype(np.uint8) * 255
        return img, mask, {"dst_box": [x, y, rw, rh], "factor": f}

    @staticmethod
    def _jpeg_roundtrip(arr: np.ndarray, q: int) -> np.ndarray:
        ok, enc = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        if not ok:
            return arr
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def jpeg_recompression(self, img: np.ndarray, rois: dict | None,
                           difficulty: str, rng: np.random.Generator):
        """Region whose JPEG quantisation grid mismatches its surroundings."""
        h, w = img.shape[:2]
        q_lo, q_hi = _JPEG_Q[difficulty]
        q = int(rng.integers(q_lo, q_hi + 1))
        x, y, rw, rh = self._pick_region(h, w, difficulty, rois, rng,
                                          self.cfg.roi_bias)
        mode = "A" if rng.random() < 0.5 else "B"
        if mode == "A":
            # Mode A: whole image recompressed, original-quality region pasted back
            base = self._jpeg_roundtrip(img, q)
            patch = img[y:y + rh, x:x + rw].copy()
        else:
            # Mode B: clean base, region independently recompressed (grid offset)
            base = img.copy()
            region = img[y:y + rh, x:x + rw]
            patch = self._jpeg_roundtrip(region, q)
            if patch.shape != region.shape:
                patch = cv2.resize(patch, (rw, rh))
        feather = 1.5
        out = base.copy()
        mask = self._paste(out, patch, x, y, feather)
        return out, mask, {"dst_box": [x, y, rw, rh], "quality": q, "mode": mode}

    # ------------------------------------------------------------------ API
    def generate(self, clean: np.ndarray, forgery_type: str | None = None,
                 difficulty: str | None = None, donor: np.ndarray | None = None,
                 rois: dict | None = None, seed: int | None = None) -> ForgeryResult:
        """Tamper `clean` and return image + mask + labels.

        Parameters mirror the module docstring; `donor` is required only for
        `splicing` (falls back to a flipped copy of `clean` if not given).
        """
        rng = self._rng(seed)
        if forgery_type is None:
            forgery_type = FORGERY_TYPES[int(rng.integers(0, len(FORGERY_TYPES)))]
        if forgery_type not in FORGERY_TYPES:
            raise ValueError(f"unknown forgery type: {forgery_type}")
        if difficulty is None:
            p = rng.random()
            difficulty = "subtle" if p < 0.4 else ("moderate" if p < 0.75 else "aggressive")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty: {difficulty}")

        img = clean.copy()
        if forgery_type == "copy_move":
            out, mask, meta = self.copy_move(img, rois, difficulty, rng)
        elif forgery_type == "splicing":
            if donor is None:
                donor = cv2.flip(clean, 1)
            out, mask, meta = self.splicing(img, donor, rois, difficulty, rng)
        elif forgery_type == "resampling":
            out, mask, meta = self.resampling(img, rois, difficulty, rng)
        else:
            out, mask, meta = self.jpeg_recompression(img, rois, difficulty, rng)

        meta.update({"type": forgery_type, "difficulty": difficulty,
                     "area_frac": float(mask.mean() / 255.0)})
        return ForgeryResult(image=out, mask=mask, forgery_type=forgery_type,
                             difficulty=difficulty, meta=meta)

    @staticmethod
    def config_dict(cfg: ForgeryConfig) -> dict:
        return asdict(cfg)


if __name__ == "__main__":
    from src.mock_ids import generate_mock_id
    rng = np.random.default_rng(0)
    gen = ForgeryGenerator(ForgeryConfig(seed=0))
    clean = generate_mock_id(rng)
    donor = generate_mock_id(rng)
    for i, t in enumerate(FORGERY_TYPES):
        r = gen.generate(clean.image, forgery_type=t, difficulty="moderate",
                         donor=donor.image, rois=clean.rois, seed=i)
        cv2.imwrite(f"scratch/forg_{t}.jpg", r.image,
                    [cv2.IMWRITE_JPEG_QUALITY, 93])
        cv2.imwrite(f"scratch/mask_{t}.png", r.mask)
        print(t, r.difficulty, r.meta)
