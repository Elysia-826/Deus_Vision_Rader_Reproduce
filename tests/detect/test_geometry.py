from __future__ import annotations

import numpy as np
import pytest

from detect.errors import EmptyCropError
from detect.geometry import clamp_box, crop_roi, remap_box
from detect.types import BBox


def test_clamp_box_keeps_in_bounds() -> None:
    box = BBox(x1=-10, y1=10, x2=110, y2=90)
    clamped = clamp_box(box, width=100, height=80)
    assert clamped == BBox(x1=0, y1=10, x2=100, y2=80)


def test_clamp_box_returns_none_when_fully_outside() -> None:
    box = BBox(x1=200, y1=200, x2=240, y2=240)
    assert clamp_box(box, width=100, height=80) is None


def test_clamp_box_returns_none_when_zero_area() -> None:
    box = BBox(x1=10, y1=10, x2=10, y2=20)
    assert clamp_box(box, width=100, height=80) is None


def test_crop_roi_returns_exact_window() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[10:20, 5:15] = 7
    crop = crop_roi(image, BBox(x1=5, y1=10, x2=15, y2=20))
    assert crop.shape == (10, 10, 3)
    assert int(crop[0, 0, 0]) == 7


def test_crop_roi_rejects_empty_box() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    with pytest.raises(EmptyCropError):
        crop_roi(image, BBox(x1=5, y1=10, x2=5, y2=20))


def test_remap_box_adds_car_origin() -> None:
    car = BBox(x1=100, y1=50, x2=300, y2=200)
    armor_in_crop = BBox(x1=10, y1=20, x2=40, y2=60)
    assert remap_box(armor_in_crop, car) == BBox(x1=110, y1=70, x2=140, y2=110)
