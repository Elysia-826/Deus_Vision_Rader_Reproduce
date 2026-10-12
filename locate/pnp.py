"""真机 PnP 标定：≥4 个地标（像素 × 世界 XYZ）→ CameraPose。

产物与仿真 Camera_Radar_Blue.json 同构：LOCATE_MODE=ray 直接可用，
比四点单应多一个维度——斜坡、高地不再是平面假设。

地标要求（决定解的质量）：
  1. 至少 4 个，建议 6 个：P3P 只要 4，多出的做残差校验；
  2. 不能全共线——现有 landmarks.json 的四个点全在 y=0 中线上，PnP 会病态；
  3. 带高度差更好（点建筑顶/塔尖比点地面底座条件数更优）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from locate.camera import CameraPose
from locate.errors import PnpCalibrationError
from locate.types import Pixel, WorldXYZ

FloatMat = NDArray[np.float64]

_MIN_POINTS = 4


@dataclass(frozen=True, slots=True)
class PnpResult:
    """解 + 重投影残差。残差大说明点击或地标坐标有错，不要拿去用。"""

    pose: CameraPose
    rms_px: float
    max_px: float


def solve_pose(
    *,
    name: str,
    width: int,
    height: int,
    K: FloatMat,
    dist: FloatMat,
    pairs: Sequence[tuple[WorldXYZ, Pixel]],
) -> PnpResult:
    """EPNP 初始化 + LM 精化。世界系与 CameraPose 一致：场地 XY、Z 上。"""
    import cv2  # noqa: PLC0415 —— 和 CameraPose.project 同样的惰性策略

    if len(pairs) < _MIN_POINTS:
        raise PnpCalibrationError(f"PnP 需要至少 {_MIN_POINTS} 个点，收到 {len(pairs)}")
    objpoints = np.array([[p.x, p.y, p.z] for p, _ in pairs], dtype=np.float64)
    imgpoints = np.array([[q.u, q.v] for _, q in pairs], dtype=np.float64)
    _reject_collinear(objpoints)
    ok, rvec, tvec = cv2.solvePnP(objpoints, imgpoints, K, dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise PnpCalibrationError("solvePnP(EPNP) 未收敛")
    rvec, tvec = cv2.solvePnPRefineLM(objpoints, imgpoints, K, dist, rvec, tvec)
    rotation, _ = cv2.Rodrigues(rvec)
    pose = CameraPose.create(
        name=name, width=width, height=height, K=K, R=rotation, t=tvec.reshape(3), dist=dist
    )
    rms, worst = _residual_px(pose, pairs)
    return PnpResult(pose=pose, rms_px=rms, max_px=worst)


def residual_px(pose: CameraPose, pairs: Sequence[tuple[WorldXYZ, Pixel]]) -> tuple[float, float]:
    """公开版残差：calibrate_pnp 存盘前再校一次用。"""
    return _residual_px(pose, pairs)


def _residual_px(pose: CameraPose, pairs: Sequence[tuple[WorldXYZ, Pixel]]) -> tuple[float, float]:
    errors = [
        float(np.hypot(projected.u - pixel.u, projected.v - pixel.v))
        for point, pixel in pairs
        for projected in (pose.project(point),)
    ]
    return float(np.sqrt(np.mean(np.square(errors)))), float(np.max(errors))


def _reject_collinear(objpoints: FloatMat) -> None:
    """所有点挤在一条直线上时 PnP 解不唯一，直接拒绝。"""
    centered = objpoints - objpoints.mean(axis=0)
    _, singular, _ = np.linalg.svd(centered)
    if float(singular[1]) < 1e-6 * max(float(singular[0]), 1.0):
        raise PnpCalibrationError("地标全共线（检查 landmarks.json：别全放在 y=0 中线上）")


def dump_camera_json(pose: CameraPose, path: Path) -> None:
    """存成 camera_from_path 认识的格式，run_radar LOCATE_MODE=ray 直接读。"""
    payload = {
        "name": pose.name,
        "image_size": [pose.width, pose.height],
        "K": [[float(v) for v in row] for row in pose.K],
        "R": [[float(v) for v in row] for row in pose.R],
        "t": [float(v) for v in pose.t.reshape(3)],
        "dist": [float(v) for v in pose.dist.reshape(-1)],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
