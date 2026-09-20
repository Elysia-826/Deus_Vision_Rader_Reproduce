"""定位用的值对象。场地是 Blender 约定：XY 平面、Z 朝上、原点在场地中心。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pixel:
    """图像坐标，u 向右，v 向下，单位像素，可以是亚像素。"""

    u: float
    v: float


@dataclass(frozen=True, slots=True)
class FieldXY:
    """场地水平坐标，米。x 沿长边，y 沿短边。"""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class WorldXYZ:
    """场地三维坐标，米。z 是高度。"""

    x: float
    y: float
    z: float
