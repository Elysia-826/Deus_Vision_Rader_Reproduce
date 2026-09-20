from __future__ import annotations

import json
from pathlib import Path

import pytest

from locate.field import FIELD_X_MAX, FIELD_X_MIN, FIELD_Y_MAX, FIELD_Y_MIN
from locate.homography import homography_from_pairs, homography_from_path
from locate.landmarks import landmarks_from_path
from locate.types import FieldXY, Pixel

_REPO = Path(__file__).resolve().parents[2]


def _axis_pairs() -> tuple[tuple[Pixel, ...], tuple[FieldXY, ...], tuple[str, ...]]:
    pixels = (
        Pixel(0.0, 0.0),
        Pixel(280.0, 0.0),
        Pixel(280.0, 150.0),
        Pixel(0.0, 150.0),
    )
    fields = (
        FieldXY(FIELD_X_MIN, FIELD_Y_MAX),
        FieldXY(FIELD_X_MAX, FIELD_Y_MAX),
        FieldXY(FIELD_X_MAX, FIELD_Y_MIN),
        FieldXY(FIELD_X_MIN, FIELD_Y_MIN),
    )
    ids = ("tl", "tr", "br", "bl")
    return pixels, fields, ids


def test_image_center_maps_to_field_center() -> None:
    pixels, fields, ids = _axis_pairs()
    mapped = homography_from_pairs(pixels, fields, ids)
    hit = mapped.pixel_to_field(Pixel(140.0, 75.0))
    assert hit.x == pytest.approx(0.0, abs=1e-2)
    assert hit.y == pytest.approx(0.0, abs=1e-2)


def test_image_top_right_maps_to_field_top_right() -> None:
    pixels, fields, ids = _axis_pairs()
    mapped = homography_from_pairs(pixels, fields, ids)
    hit = mapped.pixel_to_field(Pixel(280.0, 0.0))
    assert hit.x == pytest.approx(FIELD_X_MAX, abs=1e-2)
    assert hit.y == pytest.approx(FIELD_Y_MAX, abs=1e-2)


def test_homography_from_path_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "match.json"
    path.write_text(
        json.dumps(
            {
                "pairs": [
                    {"id": "tl", "pixel": [0, 0], "field": [FIELD_X_MIN, FIELD_Y_MAX]},
                    {"id": "tr", "pixel": [280, 0], "field": [FIELD_X_MAX, FIELD_Y_MAX]},
                    {"id": "br", "pixel": [280, 150], "field": [FIELD_X_MAX, FIELD_Y_MIN]},
                    {"id": "bl", "pixel": [0, 150], "field": [FIELD_X_MIN, FIELD_Y_MIN]},
                ]
            }
        ),
        encoding="utf-8",
    )
    mapped = homography_from_path(path)
    hit = mapped.pixel_to_field(Pixel(140.0, 75.0))
    assert hit.x == pytest.approx(0.0, abs=1e-2)
    assert hit.y == pytest.approx(0.0, abs=1e-2)


def test_repo_landmarks_json_has_four_click_points() -> None:
    table, order = landmarks_from_path(_REPO / "locate" / "calib" / "landmarks.json")
    assert len(order) >= 4
    ids = {item.id for item in table}
    assert set(order) <= ids
    energy = next(item for item in table if item.id == "energy")
    assert energy.xy.x == pytest.approx(0.0)
    assert energy.xy.y == pytest.approx(0.0)
