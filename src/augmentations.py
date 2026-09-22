"""Phone-capture simulation augmentations.

Stochastic pipeline applied **only during training** (plus one deterministic
variant used to build the `noisy` evaluation slice). Simulates handheld
capture: motion/defocus blur, specular glare, small rotation, perspective
warp, sensor noise, mild colour/exposure shifts.
"""

from __future__ import annotations

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def capture_pipeline(seed: int | None = None) -> A.Compose:
    """Handheld capture artefacts; each op fires stochastically."""
    return A.Compose(
        [
            A.Rotate(limit=4, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.Perspective(scale=(0.02, 0.06), keep_size=True,
                          border_mode=cv2.BORDER_REFLECT_101, p=0.35),
            A.MotionBlur(blur_limit=7, p=0.4),
            A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.3), p=0.3),
            A.GaussNoise(std_range=(0.02, 0.07), p=0.4),
            A.ISONoise(intensity=(0.1, 0.3), p=0.25),
            # specular glare sweeping in from a corner
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=180,
                             num_flare_circles_range=(4, 8), p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.12,
                                       contrast_limit=0.12, p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15,
                                 val_shift_limit=10, p=0.3),
            A.Downscale(scale_range=(0.75, 0.95), p=0.25),
        ],
        seed=seed,
    )


def train_transform(img_size: int = 224) -> A.Compose:
    """Resize first (cheaper) -> capture aug (train-time only) -> normalise.

    Running the capture artefacts at training resolution rather than the
    native 768x480 keeps the pipeline fast enough to stay off the critical
    path of data loading.
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            *capture_pipeline().transforms,
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def eval_transform(img_size: int = 224) -> A.Compose:
    """Deterministic resize + normalise (clean slice)."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def noisy_eval_transform(img_size: int, seed: int) -> A.Compose:
    """Capture aug with a fixed seed -> reproducible `noisy` test slice."""
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            *capture_pipeline(seed=seed).transforms,
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        seed=seed,
    )


def adversarial_eval_transform(img_size: int, seed: int) -> A.Compose:
    """Heavier capture degradation for the hardest slice (fixed seed)."""
    pipe = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Rotate(limit=6, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.Perspective(scale=(0.03, 0.08), keep_size=True,
                          border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.MotionBlur(blur_limit=9, p=0.5),
            A.Defocus(radius=(3, 6), alias_blur=(0.1, 0.4), p=0.4),
            A.GaussNoise(std_range=(0.03, 0.09), p=0.5),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.6), src_radius=220,
                             num_flare_circles_range=(4, 8), p=0.35),
            A.RandomBrightnessContrast(brightness_limit=0.15,
                                       contrast_limit=0.15, p=0.6),
            A.Downscale(scale_range=(0.6, 0.9), p=0.35),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        seed=seed,
    )
    return pipe
