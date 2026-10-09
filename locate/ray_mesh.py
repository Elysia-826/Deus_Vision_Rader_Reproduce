"""像素射线打场地三角网格。港科大同一条底边像素，交点取最近的正 t。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from locate.camera import CameraPose
from locate.ray_plane import _ray_world
from locate.types import FieldXY, Pixel

_CELL_M = 0.5
_X_MIN = -16.0
_Y_MIN = -9.0
_NX = 64
_NY = 36
_T_MIN = 1e-3
_T_MAX = 40.0


@dataclass(frozen=True, slots=True)
class FieldMesh:
    """XY 格子里的三角面。射线只测穿过的格子，避免每次扫 100 万面。"""

    vertices: np.ndarray
    faces: np.ndarray
    cell_faces: tuple[np.ndarray, ...]

    @classmethod
    def from_npz(cls, path: Path) -> FieldMesh:
        data = np.load(path)
        vertices = np.asarray(data["vertices"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int32)
        return cls(vertices=vertices, faces=faces, cell_faces=_bin_faces(vertices, faces))

    def pixel_to_field(self, pose: CameraPose, pixel: Pixel) -> FieldXY | None:
        origin, direction = _ray_world(pose, pixel)
        hit = self.cast(origin, direction)
        if hit is None:
            return None
        return FieldXY(x=float(hit[0]), y=float(hit[1]))

    def cast(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
        direction = np.asarray(direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            return None
        direction = direction / norm
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        best_t = _T_MAX
        best: np.ndarray | None = None
        for t_enter, cell in _cells_along(origin, direction):
            if t_enter > best_t:
                break
            face_ids = self.cell_faces[cell]
            if len(face_ids) == 0:
                continue
            tri = self.vertices[self.faces[face_ids]]
            t = _closest_t(origin, direction, tri)
            if t is not None and t < best_t:
                best_t = t
                best = origin + t * direction
        return best


def _bin_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, ...]:
    tri = vertices[faces]
    mins = tri.min(axis=1)
    maxs = tri.max(axis=1)
    ix0 = np.clip(((mins[:, 0] - _X_MIN) / _CELL_M).astype(np.int32), 0, _NX - 1)
    ix1 = np.clip(((maxs[:, 0] - _X_MIN) / _CELL_M).astype(np.int32), 0, _NX - 1)
    iy0 = np.clip(((mins[:, 1] - _Y_MIN) / _CELL_M).astype(np.int32), 0, _NY - 1)
    iy1 = np.clip(((maxs[:, 1] - _Y_MIN) / _CELL_M).astype(np.int32), 0, _NY - 1)
    single = (ix0 == ix1) & (iy0 == iy1)
    keys = ix0[single] + iy0[single] * _NX
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    single_ids = np.flatnonzero(single)[order]
    buckets: list[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(_NX * _NY)]
    if len(sorted_keys):
        cuts = np.flatnonzero(np.diff(sorted_keys)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [len(sorted_keys)]))
        for start, end in zip(starts, ends, strict=True):
            buckets[int(sorted_keys[start])] = single_ids[start:end].astype(np.int32, copy=False)
    extra: list[list[int]] = [[] for _ in range(_NX * _NY)]
    for index in np.flatnonzero(~single):
        for ix in range(int(ix0[index]), int(ix1[index]) + 1):
            for iy in range(int(iy0[index]), int(iy1[index]) + 1):
                extra[ix + iy * _NX].append(int(index))
    merged: list[np.ndarray] = []
    for bucket, more in zip(buckets, extra, strict=True):
        if not more:
            merged.append(bucket)
            continue
        merged.append(np.concatenate((bucket, np.asarray(more, dtype=np.int32))))
    return tuple(merged)


def _cell_key(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ix = np.clip(((x - _X_MIN) / _CELL_M).astype(np.int32), 0, _NX - 1)
    iy = np.clip(((y - _Y_MIN) / _CELL_M).astype(np.int32), 0, _NY - 1)
    return ix + iy * _NX


def _cells_along(origin: np.ndarray, direction: np.ndarray) -> list[tuple[float, int]]:
    seen: list[tuple[float, int]] = []
    used: set[int] = set()
    step = _CELL_M * 0.5
    t = _T_MIN
    while t < _T_MAX:
        point = origin + t * direction
        key = int(_cell_key(np.array([point[0]]), np.array([point[1]]))[0])
        if key not in used:
            used.add(key)
            seen.append((t, key))
        t += step
    return seen


def _closest_t(origin: np.ndarray, direction: np.ndarray, tri: np.ndarray) -> float | None:
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    ok = np.abs(det) > 1e-8
    if not np.any(ok):
        return None
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]
    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, edge1)
    v = np.einsum("ij,j->i", qvec, direction) * inv
    dist = np.einsum("ij,ij->i", edge2, qvec) * inv
    hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (dist > _T_MIN) & (dist < _T_MAX)
    if not np.any(hit):
        return None
    return float(dist[hit].min())
