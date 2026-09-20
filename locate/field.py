"""RMUC 官方俯视图标注的场地外框：28 m × 15 m，原点在中心。"""

from __future__ import annotations

from typing import Final

from locate.types import FieldXY

FIELD_LENGTH_M: Final[float] = 28.0
FIELD_WIDTH_M: Final[float] = 15.0
FIELD_X_MIN: Final[float] = -14.0
FIELD_X_MAX: Final[float] = 14.0
FIELD_Y_MIN: Final[float] = -7.5
FIELD_Y_MAX: Final[float] = 7.5


def field_corners() -> tuple[FieldXY, FieldXY, FieldXY, FieldXY]:
    """俯视四角，顺序左上、右上、右下、左下（y 上为正）。"""
    return (
        FieldXY(x=FIELD_X_MIN, y=FIELD_Y_MAX),
        FieldXY(x=FIELD_X_MAX, y=FIELD_Y_MAX),
        FieldXY(x=FIELD_X_MAX, y=FIELD_Y_MIN),
        FieldXY(x=FIELD_X_MIN, y=FIELD_Y_MIN),
    )
