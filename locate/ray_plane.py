"""像素射线打 Z=0 地面。车辆框底边中点当触地点。"""

from __future__ import annotations

import numpy as np

from detect.types import BBox
from locate.camera import CameraPose, FloatMat
from locate.types import FieldXY, Pixel, WorldXYZ

_PARALLEL_EPS = 1e-9


def foot_pixel(box: BBox) -> Pixel:
    """车辆框底边中点。x2/y2 是不含端，底边落在最后一行像素中心。"""
    return Pixel(u=(box.x1 + box.x2) / 2.0, v=float(box.y2) - 0.5)


def _ray_world(pose: CameraPose, pixel: Pixel) -> tuple[FloatMat, FloatMat]:
    """去畸变后：光心 C，世界系方向 d（未单位化）。"""
    import cv2

    pts = np.array([[[pixel.u, pixel.v]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, pose.K, pose.dist, P=pose.K)
    u, v = undistorted.reshape(2)
    pixel_h = np.array([u, v, 1.0], dtype=np.float64)
    cam_dir = np.linalg.inv(pose.K) @ pixel_h
    world_dir = pose.R.T @ cam_dir
    return pose.center, world_dir


def pixel_to_ground(pose: CameraPose, pixel: Pixel, *, z: float = 0.0) -> FieldXY | None:
    """像素 → 地面 (x, y)。平行地面、打到相机后、打到无穷远则 None。"""
    origin, direction = _ray_world(pose, pixel)
    dz = float(direction[2])
    if abs(dz) < _PARALLEL_EPS:
        return None
    scale = (z - float(origin[2])) / dz
    if scale <= 0.0:
        return None
    hit = origin + scale * direction
    return FieldXY(x=float(hit[0]), y=float(hit[1]))


def world_to_pixel(pose: CameraPose, point: WorldXYZ) -> Pixel:
    return pose.project(point)
