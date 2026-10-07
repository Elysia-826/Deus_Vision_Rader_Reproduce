from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from locate.camera import CameraPose, blender_intrinsics, blender_world_to_opencv, camera_from_path
from locate.errors import CameraJsonError
from locate.types import WorldXYZ


def looking_down() -> CameraPose:
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    tvec = np.array([[0.0], [0.0], [10.0]])
    k = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    return CameraPose.create("test", 1920, 1080, k, rotation, tvec)


def test_center_is_minus_r_transpose_t() -> None:
    pose = looking_down()
    np.testing.assert_allclose(pose.center, [0.0, 0.0, 10.0])


def test_project_ground_origin_hits_principal_point() -> None:
    pixel = looking_down().project(WorldXYZ(0.0, 0.0, 0.0))
    assert pixel.u == pytest.approx(960.0)
    assert pixel.v == pytest.approx(540.0)


def test_project_offset_ground_point() -> None:
    pixel = looking_down().project(WorldXYZ(2.0, 1.0, 0.0))
    assert pixel.u == pytest.approx(1160.0)
    assert pixel.v == pytest.approx(440.0)


def test_blender_identity_camera_matches_looking_down() -> None:
    rotation, tvec = blender_world_to_opencv(np.eye(3), np.array([0.0, 0.0, 10.0]))
    np.testing.assert_allclose(rotation, [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    np.testing.assert_allclose(tvec.reshape(3), [0.0, 0.0, 10.0])


def test_shipped_radar_k_is_mv_ch120_8mm() -> None:
    """MV-CH120-60UC，3.45 μm，4096×3000，1.1 英寸靶面 8 mm。主点在中心，未测畸变。"""
    pixel_um = 3.45
    width, height = 4096, 3000
    expected = blender_intrinsics(
        lens_mm=8.0,
        sensor_width_mm=width * pixel_um / 1000.0,
        sensor_height_mm=height * pixel_um / 1000.0,
        sensor_fit="HORIZONTAL",
        pixel_width=width,
        pixel_height=height,
    )
    root = Path(__file__).resolve().parents[2] / "locate" / "calib"
    for name in ("Camera_Radar_Blue.json", "Camera_Radar_Red.json"):
        pose = camera_from_path(root / name)
        assert (pose.width, pose.height) == (width, height)
        np.testing.assert_allclose(pose.K, expected)
        assert float(pose.K[0, 0]) == pytest.approx(8.0 / 0.00345)


def test_blender_intrinsics_horizontal_36mm() -> None:
    k = blender_intrinsics(
        lens_mm=50.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        sensor_fit="HORIZONTAL",
        pixel_width=1920,
        pixel_height=1080,
    )
    assert k[0, 0] == pytest.approx(50.0 * 1920 / 36.0)
    assert k[1, 1] == pytest.approx(k[0, 0])
    assert k[0, 2] == pytest.approx(960.0)
    assert k[1, 2] == pytest.approx(540.0)


def test_camera_from_path_reads_column_t(tmp_path: Path) -> None:
    pose = looking_down()
    payload = {
        "name": "Camera_Radar_Blue",
        "image_size": [1920, 1080],
        "K": pose.K.tolist(),
        "R": pose.R.tolist(),
        "t": pose.t.tolist(),
        "dist": [0, 0, 0, 0, 0],
    }
    path = tmp_path / "cam.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = camera_from_path(path)
    np.testing.assert_allclose(loaded.K, pose.K)
    np.testing.assert_allclose(loaded.t, pose.t)
    assert loaded.name == "Camera_Radar_Blue"


def test_camera_from_path_rejects_missing_k(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"name": "x", "image_size": [1, 1], "R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,1]}', encoding="utf-8")
    with pytest.raises(CameraJsonError):
        camera_from_path(path)
