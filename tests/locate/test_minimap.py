from __future__ import annotations

import json
from pathlib import Path

import pytest

from locate.field import FIELD_X_MAX, FIELD_Y_MAX
from locate.minimap import minimap_from_corners, minimap_from_path
from locate.types import FieldXY, Pixel

_REPO = Path(__file__).resolve().parents[2]


def axis_aligned() -> tuple[Pixel, Pixel, Pixel, Pixel]:
    return (
        Pixel(u=0.0, v=0.0),
        Pixel(u=800.0, v=0.0),
        Pixel(u=800.0, v=400.0),
        Pixel(u=0.0, v=400.0),
    )


def test_field_center_maps_to_image_center() -> None:
    calib = minimap_from_corners(axis_aligned(), width=800, height=400)
    pixel = calib.field_to_pixel(FieldXY(0.0, 0.0))
    assert pixel.u == pytest.approx(400.0)
    assert pixel.v == pytest.approx(200.0)


def test_field_top_right_maps_to_image_top_right() -> None:
    calib = minimap_from_corners(axis_aligned(), width=800, height=400)
    pixel = calib.field_to_pixel(FieldXY(FIELD_X_MAX, FIELD_Y_MAX))
    assert pixel.u == pytest.approx(800.0)
    assert pixel.v == pytest.approx(0.0)


def test_minimap_from_path(tmp_path: Path) -> None:
    path = tmp_path / "minimap.json"
    path.write_text(
        json.dumps(
            {
                "image_size": [800, 400],
                "image_corners": [[0, 0], [800, 0], [800, 400], [0, 400]],
            }
        ),
        encoding="utf-8",
    )
    calib = minimap_from_path(path)
    pixel = calib.field_to_pixel(FieldXY(0.0, 0.0))
    assert pixel.u == pytest.approx(400.0)
    assert pixel.v == pytest.approx(200.0)


def test_repo_minimap_json_maps_center() -> None:
    calib = minimap_from_path(_REPO / "locate" / "calib" / "minimap.json")
    pixel = calib.field_to_pixel(FieldXY(0.0, 0.0))
    assert pixel.u == pytest.approx(834.0)
    assert pixel.v == pytest.approx(477.5)
