"""MobileNetV3-Small：只换最后一层线性头，6 类图案。

港科大用 V2 @ 64。V3-Small 同样是 64 输入、ImageNet 预训练，头更轻。
不要换整网结构：训练脚本和 TensorRT 导出都按这一层 Linear 的 out_features=6 来。
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from models.pattern.classes import CLASS_NAMES


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
