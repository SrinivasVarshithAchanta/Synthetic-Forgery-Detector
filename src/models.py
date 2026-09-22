"""Model zoo: pretrained backbones + a lightweight custom CNN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class DepthwiseSeparable(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightweightCNN(nn.Module):
    """Small depthwise-separable CNN (~1.4M params) for latency comparison."""

    def __init__(self, num_classes: int = 2, width: int = 32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, w, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            DepthwiseSeparable(w, w * 2, stride=1),      # /2
            DepthwiseSeparable(w * 2, w * 4, stride=2),  # /4
            DepthwiseSeparable(w * 4, w * 4, stride=1),
            DepthwiseSeparable(w * 4, w * 8, stride=2),  # /8
            DepthwiseSeparable(w * 8, w * 8, stride=1),
            DepthwiseSeparable(w * 8, w * 16, stride=2),  # /16
            DepthwiseSeparable(w * 16, w * 16, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(w * 16, num_classes))
        self.last_conv = self.features[-1]

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


MODEL_NAMES = ["resnet50", "resnet18", "efficientnet_b0", "custom_cnn"]


def create_model(name: str, num_classes: int = 2,
                 pretrained: bool = True) -> nn.Module:
    """Build a classifier; the Grad-CAM target layer is stashed on the model."""
    if name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model._gradcam_target = model.layer4
    elif name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model._gradcam_target = model.layer4
    elif name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model._gradcam_target = model.features[-1]
    elif name == "custom_cnn":
        model = LightweightCNN(num_classes=num_classes)
        model._gradcam_target = model.last_conv
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for n in MODEL_NAMES:
        m = create_model(n, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        y = m(x)
        print(f"{n:18s} params={count_parameters(m)/1e6:.2f}M out={tuple(y.shape)}")
