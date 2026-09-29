from __future__ import annotations

import random

import numpy as np

from dataset.sim_pattern_crops import jitter_plate


def test_jitter_plate_returns_smaller_dark_crop() -> None:
    plate = np.full((64, 64, 3), 200, dtype=np.uint8)
    crop = jitter_plate(plate, random.Random(0))
    assert crop.ndim == 3
    assert crop.shape[0] >= 28
    assert crop.shape[0] <= 92
    assert float(crop.mean()) < 180
