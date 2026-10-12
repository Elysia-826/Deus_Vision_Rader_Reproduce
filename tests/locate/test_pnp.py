from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from locate.camera import camera_from_path
from locate.errors import PnpCalibrationError
from locate.pnp import dump_camera_json, solve_pose
from locate.types import Pixel, WorldXYZ

_REPO = Path(__file__).resolve().parents[2]
_CAMERA = _REPO / "locate" / "calib" / "Camera_Radar_Blue.json"

# 仿真场景固定建筑（Blender 真值坐标，非共线、带高度差，都在蓝雷达画面内）。
_LANDMARKS = (
    WorldXYZ(x=0.0, y=0.0, z=0.68),
    WorldXYZ(x=0.99, y=1.04, z=1.82),
    WorldXYZ(x=-0.99, y=-0.95, z=1.83),
    WorldXYZ(x=0.47, y=0.0, z=0.89),
    WorldXYZ(x=-13.42, y=0.0, z=2.5),
    WorldXYZ(x=-13.77, y=0.9, z=2.8),
)


def _gt() -> object:
    return camera_from_path(_CAMERA)


def _pairs(gt, noise_px: float = 0.0) -> list[tuple[WorldXYZ, Pixel]]:
    rng = np.random.default_rng(3)
    return [
        (
            point,
            Pixel(
                u=pixel.u + float(rng.normal(0.0, noise_px)),
                v=pixel.v + float(rng.normal(0.0, noise_px)),
            ),
        )
        for point in _LANDMARKS
        for pixel in (gt.project(point),)
    ]


def _solve(gt, pairs) -> object:
    return solve_pose(
        name="recovered", width=gt.width, height=gt.height, K=gt.K, dist=gt.dist, pairs=pairs
    )


def test_solve_pose_recovers_sim_camera_exactly() -> None:
    gt = _gt()
    result = _solve(gt, _pairs(gt))
    assert result.rms_px < 1e-6
    np.testing.assert_allclose(result.pose.R, gt.R, atol=1e-8)
    np.testing.assert_allclose(result.pose.t, gt.t, atol=1e-8)
    np.testing.assert_allclose(result.pose.center, gt.center, atol=1e-8)


def test_solve_pose_survives_sloppy_clicks() -> None:
    """10px 均匀噪声的点击也应解出可用外参：残差在噪声量级内，光心偏差有界。"""
    gt = _gt()
    result = _solve(gt, _pairs(gt, noise_px=10.0))
    assert result.rms_px < 25.0  # 6 点 σ=10px：残差本身就是噪声量级
    center_error = float(np.linalg.norm(result.pose.center - gt.center))
    assert center_error < 0.5  # 0.8m 准确线还有余量


def test_solve_pose_rejects_too_few_points() -> None:
    gt = _gt()
    with pytest.raises(PnpCalibrationError):
        _solve(gt, _pairs(gt)[:3])


def test_solve_pose_rejects_collinear_landmarks() -> None:
    """全在 y=0 中线上的地标集合必须直接报错（PnP 解不唯一）。"""
    gt = _gt()
    collinear = tuple(WorldXYZ(x=float(x), y=0.0, z=0.0) for x in (-12.5, -5.5, 0.0, 5.5, 12.5))
    pairs = [(point, gt.project(point)) for point in collinear]
    with pytest.raises(PnpCalibrationError):
        _solve(gt, pairs)


def test_dump_camera_json_roundtrips(tmp_path: Path) -> None:
    gt = _gt()
    result = _solve(gt, _pairs(gt))
    out = tmp_path / "camera.json"
    dump_camera_json(result.pose, out)
    loaded = camera_from_path(out)
    assert loaded.name == "recovered"
    np.testing.assert_allclose(loaded.R, gt.R, atol=1e-8)
    np.testing.assert_allclose(loaded.t, gt.t, atol=1e-8)
    pixel = loaded.project(_LANDMARKS[0])
    expected = gt.project(_LANDMARKS[0])
    assert pixel.u == pytest.approx(expected.u)
    assert pixel.v == pytest.approx(expected.v)
