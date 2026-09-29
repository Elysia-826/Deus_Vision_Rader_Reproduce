from pathlib import Path

import pytest

from dataset.convert_rm27_armor import UnknownRm27Label, armor_class, pattern_folder, to_yolo, xyxy_from_points
from models.pattern.classes import ArmorFolder


def test_pattern_folder_maps_six_known_labels() -> None:
    assert pattern_folder("blue1") is ArmorFolder.B1
    assert pattern_folder("red3") is ArmorFolder.R3
    assert pattern_folder("bluesb") is ArmorFolder.BS
    assert pattern_folder("redsb") is ArmorFolder.RS


def test_armor_class_is_color_only() -> None:
    assert armor_class("red1") == 1
    assert armor_class("blue3") == 2
    assert armor_class("redsb") == 1


def test_unknown_label_raises() -> None:
    with pytest.raises(UnknownRm27Label):
        pattern_folder("blue2")


def test_to_yolo_center_box() -> None:
    cx, cy, bw, bh = to_yolo(25, 25, 75, 75, 100, 100)
    assert abs(cx - 0.5) < 1e-6
    assert abs(cy - 0.5) < 1e-6
    assert abs(bw - 0.5) < 1e-6
    assert abs(bh - 0.5) < 1e-6


def test_xyxy_from_labelme_quad() -> None:
    x1, y1, x2, y2 = xyxy_from_points([[10.0, 20.0], [40.0, 20.0], [40.0, 50.0], [10.0, 50.0]])
    assert (x1, y1, x2, y2) == (10.0, 20.0, 40.0, 50.0)
