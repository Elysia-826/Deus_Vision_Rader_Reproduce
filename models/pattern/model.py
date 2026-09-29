"""图案分类骨干。默认仍是 V3-Small；EfficientNet-B0 给 96 输入对照。"""

from __future__ import annotations

from enum import StrEnum
from typing import assert_never

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
)

from models.pattern.classes import CLASS_NAMES


class PatternBackbone(StrEnum):
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    EFFICIENTNET_B0 = "efficientnet_b0"


class MobileNetV3SmallClassifier(nn.Module):  # noqa: MUTABLE_OK — nn.Module 必须可变
    """torchvision V3-Small + 替换 classifier[-1]。"""

    def __init__(self, num_classes: int = len(CLASS_NAMES), pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


class EfficientNetB0Classifier(nn.Module):  # noqa: MUTABLE_OK — nn.Module 必须可变
    """B0 默认 224，本仓用 96。classifier[1] 是最后一层 Linear。"""

    def __init__(self, num_classes: int = len(CLASS_NAMES), pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


def build_classifier(
    backbone: PatternBackbone,
    *,
    num_classes: int = len(CLASS_NAMES),
    pretrained: bool = True,
) -> nn.Module:
    match backbone:
        case PatternBackbone.MOBILENET_V3_SMALL:
            return MobileNetV3SmallClassifier(num_classes=num_classes, pretrained=pretrained)
        case PatternBackbone.EFFICIENTNET_B0:
            return EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained)
        case unreachable:
            assert_never(unreachable)
