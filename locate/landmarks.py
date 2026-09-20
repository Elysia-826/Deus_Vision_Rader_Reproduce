"""RMUC 场上不动点。坐标是地面落点，米，原点在场地中心。

默认值按 RMUC2026 28×15 m 俯视图的对称布局，蓝侧 +x（蓝雷达约 x=13 m）。
换年规则图时只改 locate/calib/landmarks.json，不要改点击工具。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from locate.errors import HomographyError
from locate.types import FieldXY

DEFAULT_CLICK_ORDER: tuple[str, ...] = (
    "outpost_blue",
    "outpost_red",
    "energy",
    "base_blue",
)


@dataclass(frozen=True, slots=True)
class Landmark:
    id: str
    name_zh: str
    xy: FieldXY


class _LandmarkFileItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name_zh: str
    x: float
    y: float


class _LandmarkFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    landmarks: tuple[_LandmarkFileItem, ...]
    click_order: tuple[str, ...] = DEFAULT_CLICK_ORDER


def landmarks_from_path(path: Path) -> tuple[tuple[Landmark, ...], tuple[str, ...]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HomographyError(path=path, reason="cannot read landmarks") from exc
    except json.JSONDecodeError as exc:
        raise HomographyError(path=path, reason=f"invalid json: {exc.msg}") from exc
    try:
        parsed = _LandmarkFile.model_validate(payload)
    except ValidationError as exc:
        raise HomographyError(path=path, reason=str(exc)) from exc
    table = tuple(Landmark(id=item.id, name_zh=item.name_zh, xy=FieldXY(item.x, item.y)) for item in parsed.landmarks)
    ids = {item.id for item in table}
    for landmark_id in parsed.click_order:
        if landmark_id not in ids:
            raise HomographyError(path=path, reason=f"click_order unknown id {landmark_id}")
    if len(parsed.click_order) < 4:
        raise HomographyError(path=path, reason="click_order needs at least 4 landmarks")
    return table, parsed.click_order


def landmark_by_id(table: tuple[Landmark, ...], landmark_id: str) -> Landmark:
    for item in table:
        if item.id == landmark_id:
            return item
    raise HomographyError(path=None, reason=f"unknown landmark {landmark_id}")
