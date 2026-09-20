"""场地米制 ↔ 官方俯视图像素。四角单应，图四周留白所以不能整图线性缩放。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

from locate.errors import MinimapCalibError
from locate.field import field_corners
from locate.types import FieldXY, Pixel

FloatMat = NDArray[np.float64]


class _MinimapFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_size: tuple[int, int]
    image_corners: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass(frozen=True, slots=True)
class MinimapCalib:
    """俯视图四角（左上、右上、右下、左下）对应场地四角。"""

    width: int
    height: int
    homography: FloatMat

    def field_to_pixel(self, point: FieldXY) -> Pixel:
        import cv2

        src = np.array([[[point.x, point.y]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.homography)
        u, v = dst.reshape(2)
        return Pixel(u=float(u), v=float(v))


def minimap_from_corners(
    image_corners: tuple[Pixel, Pixel, Pixel, Pixel],
    *,
    width: int,
    height: int,
) -> MinimapCalib:
    """image_corners 顺序必须是左上、右上、右下、左下。"""
    import cv2

    if width <= 0 or height <= 0:
        raise MinimapCalibError(path=None, reason=f"bad minimap size {width}x{height}")
    field = field_corners()
    src = np.array([[p.x, p.y] for p in field], dtype=np.float32)
    dst = np.array([[p.u, p.v] for p in image_corners], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    homography.flags.writeable = False
    return MinimapCalib(width=width, height=height, homography=homography)


def minimap_from_path(path: Path) -> MinimapCalib:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MinimapCalibError(path=path, reason="cannot read file") from exc
    except json.JSONDecodeError as exc:
        raise MinimapCalibError(path=path, reason=f"invalid json: {exc.msg}") from exc
    try:
        parsed = _MinimapFile.model_validate(payload)
    except ValidationError as exc:
        raise MinimapCalibError(path=path, reason=str(exc)) from exc
    corners = tuple(Pixel(u=u, v=v) for u, v in parsed.image_corners)
    if len(corners) != 4:
        raise MinimapCalibError(path=path, reason="need 4 image_corners")
    return minimap_from_corners(
        (corners[0], corners[1], corners[2], corners[3]),
        width=parsed.image_size[0],
        height=parsed.image_size[1],
    )
