from __future__ import annotations

import numpy as np

from locate.camera import CameraPose
from locate.ray_mesh import FieldMesh, _bin_faces
from locate.types import Pixel, WorldXYZ


def _mesh() -> FieldMesh:
    vertices = np.array(
        [[-1.0, -1.0, 0.4], [1.0, -1.0, 0.4], [1.0, 1.0, 0.4], [-1.0, 1.0, 0.4]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return FieldMesh(vertices=vertices, faces=faces, cell_faces=_bin_faces(vertices, faces))


def _pose() -> CameraPose:
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    tvec = np.array([[0.0], [0.0], [5.0]])
    k = np.array([[1000.0, 0.0, 100.0], [0.0, 1000.0, 100.0], [0.0, 0.0, 1.0]])
    return CameraPose.create("down", 200, 200, k, rotation, tvec)


def test_center_pixel_hits_raised_quad() -> None:
    hit = _mesh().pixel_to_field(_pose(), Pixel(u=100.0, v=100.0))
    assert hit is not None
    assert abs(hit.x) < 1e-3
    assert abs(hit.y) < 1e-3


def test_projected_point_roundtrips() -> None:
    pose = _pose()
    pixel = pose.project(WorldXYZ(0.2, -0.1, 0.4))
    hit = _mesh().pixel_to_field(pose, pixel)
    assert hit is not None
    assert abs(hit.x - 0.2) < 1e-2
    assert abs(hit.y + 0.1) < 1e-2


def test_upward_ray_misses() -> None:
    assert _mesh().cast(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])) is None
