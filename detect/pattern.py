"""装甲 ROI → EfficientNet-B0 图案。颜色仍由装甲 YOLO 的 red/blue/dead 提供。

港科大第三级是 MobileNet-V2 @ 64，带颜色的 14 类文件夹。本仓把颜色丢掉，
只认 6 类兵种 @ 96，和 models.pattern.classes.CLASS_NAMES 对齐。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from numpy import uint8
from numpy.typing import NDArray
from PIL import Image

from detect.types import Detection, LinkedRobot, PatternPred
from models.pattern.classes import CLASS_NAMES

ImageU8 = NDArray[uint8]
PatternTransform = Callable[[Image.Image], torch.Tensor]


class PatternNet(Protocol):
    def __call__(self, batch: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class PatternStage:
    """一次推理用的分类器。net 可以是 torchvision 也可以是 TensorRT 包装。"""

    net: PatternNet
    transform: PatternTransform
    device: str
    names: tuple[str, ...] = CLASS_NAMES


def attach_patterns(
    frame: ImageU8,
    robots: tuple[LinkedRobot, ...],
    stage: PatternStage,
) -> tuple[LinkedRobot, ...]:
    """按全图装甲框裁图、组 batch、写回 PatternPred。空裁剪保持 pattern=None。"""
    tensors: list[torch.Tensor] = []
    keys: list[tuple[int, int]] = []
    for robot_i, robot in enumerate(robots):
        for armor_i, armor in enumerate(robot.armors):
            box = armor.box
            if box.width <= 0 or box.height <= 0:
                continue
            crop = frame[box.y1 : box.y2, box.x1 : box.x2]
            if crop.size == 0:
                continue
            # OpenCV 帧是 BGR，训练/分类器是 RGB；::-1 比再 import cv2 轻。
            rgb = crop[:, :, ::-1].copy()
            tensors.append(stage.transform(Image.fromarray(rgb)))
            keys.append((robot_i, armor_i))
    preds: dict[tuple[int, int], PatternPred] = {}
    if tensors:
        device = torch.device(stage.device)
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            probs = torch.softmax(stage.net(batch), dim=1)
            class_ids = probs.argmax(dim=1).tolist()
        for key, class_id, row in zip(keys, class_ids, probs, strict=True):
            name = stage.names[class_id] if class_id < len(stage.names) else str(class_id)
            preds[key] = PatternPred(name=name, conf=float(row[class_id]))
    return tuple(
        LinkedRobot(
            car=robot.car,
            armors=tuple(
                Detection(
                    label=armor.label,
                    conf=armor.conf,
                    box=armor.box,
                    track_id=armor.track_id,
                    pattern=preds.get((robot_i, armor_i)),
                )
                for armor_i, armor in enumerate(robot.armors)
            ),
        )
        for robot_i, robot in enumerate(robots)
    )
