from __future__ import annotations

import numpy as np
import pytest

from detect.types import BBox
from locate.camera import CameraPose
from locate.ray_plane import foot_pixel, pixel_to_ground, world_to_pixel
from locate.types import Pixel, WorldXYZ


def looking_down() -> CameraPose:
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    tvec = np.array([[0.0], [0.0], [10.0]])
    k = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    return CameraPose.create("test", 1920, 1080, k, rotation, tvec)


def test_foot_pixel_uses_bottom_row_center() -> None:
    pixel = foot_pixel(BBox(x1=10, y1=20, x2=30, y2=40))
    assert pixel.u == pytest.approx(20.0)
    assert pixel.v == pytest.approx(39.5)


def test_pixel_to_ground_inverts_project() -> None:
    pose = looking_down()
    world = WorldXYZ(2.0, 1.0, 0.0)
    pixel = world_to_pixel(pose, world)
    hit = pixel_to_ground(pose, pixel)
    assert hit is not None
    assert hit.x == pytest.approx(2.0, abs=1e-6)
    assert hit.y == pytest.approx(1.0, abs=1e-6)


def test_pixel_to_ground_rejects_upward_ray() -> None:
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    tvec = np.array([[0.0], [1.0], [10.0]])
    k = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    pose = CameraPose.create("up", 1920, 1080, k, rotation, tvec)
    assert pixel_to_ground(pose, Pixel(u=960.0, v=0.0)) is None


def test_pixel_to_ground_rejects_parallel_ray() -> None:
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    tvec = np.array([[0.0], [-10.0], [0.0]])
    k = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    pose = CameraPose.create("horizon", 1920, 1080, k, rotation, tvec)
    assert pixel_to_ground(pose, Pixel(u=960.0, v=540.0)) is None
