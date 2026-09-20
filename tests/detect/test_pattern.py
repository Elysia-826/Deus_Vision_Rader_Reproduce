"""Given armor crops, When pattern net votes a class, Then Detection.pattern is that class name."""

from __future__ import annotations

import numpy as np
import torch

from detect.pattern import PatternStage, attach_patterns
from detect.types import BBox, Detection, LinkedRobot, PatternPred
from models.pattern.classes import CLASS_NAMES
from models.pattern.transforms import val_transform


class _ConstPatternNet:
    """Every crop → class index 4 (S). Logits only, no softmax."""

    def __init__(self, class_index: int) -> None:
        self._class_index = class_index

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(batch.shape[0], len(CLASS_NAMES))
        logits[:, self._class_index] = 8.0
        return logits


def test_attach_patterns_writes_sentry_name_on_red_armor() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[16:26, 25:35] = 40
    car = Detection(label="robot", conf=0.9, box=BBox(20, 10, 80, 60))
    armor = Detection(label="red", conf=0.7, box=BBox(25, 16, 35, 26))
    robots = (LinkedRobot(car=car, armors=(armor,)),)
    stage = PatternStage(net=_ConstPatternNet(4), transform=val_transform(64), device="cpu")

    out = attach_patterns(frame, robots, stage)

    pred = out[0].armors[0].pattern
    assert pred == PatternPred(name="S", conf=pred.conf)
    assert pred.conf > 0.9
    assert out[0].armors[0].label == "red"


def test_attach_patterns_leaves_empty_crop_without_pattern() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    car = Detection(label="robot", conf=0.9, box=BBox(20, 10, 80, 60))
    armor = Detection(label="blue", conf=0.5, box=BBox(0, 0, 0, 0))
    robots = (LinkedRobot(car=car, armors=(armor,)),)
    stage = PatternStage(net=_ConstPatternNet(0), transform=val_transform(64), device="cpu")

    out = attach_patterns(frame, robots, stage)

    assert out[0].armors[0].pattern is None
