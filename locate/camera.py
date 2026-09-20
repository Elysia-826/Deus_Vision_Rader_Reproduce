"""OpenCV 相机位姿：X_cam = R X_world + t。世界系是场地 XY、Z 上。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from locate.errors import CameraJsonError
from locate.types import Pixel, WorldXYZ

FloatMat = NDArray[np.float64]

_BCAM_TO_CV: FloatMat = np.diag(np.array([1.0, -1.0, -1.0], dtype=np.float64))


class _CameraFile(BaseModel):
    """磁盘 JSON 的唯一入口。内部 CameraPose 不再碰 dict。"""

    model_config = ConfigDict(frozen=True)

    name: str
    image_size: tuple[int, int]
    K: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    R: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    t: tuple[float, float, float]
    dist: tuple[float, ...] = Field(default=(0.0, 0.0, 0.0, 0.0, 0.0))

    @field_validator("t", mode="before")
    @classmethod
    def _flatten_t(cls, value: list[float] | list[list[float]] | tuple[float, ...]) -> tuple[float, float, float]:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != 3:
            raise ValueError("t must have 3 entries")  # noqa: GENERIC_ERR_OK — pydantic before-validator
        return (float(array[0]), float(array[1]), float(array[2]))

    @field_validator("dist", mode="before")
    @classmethod
    def _dist_tuple(cls, value: list[float] | tuple[float, ...]) -> tuple[float, ...]:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        return tuple(float(item) for item in array)


def _as_3x3(name: str, raw: Sequence[Sequence[float]]) -> FloatMat:
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise CameraJsonError(path=Path("."), reason=f"{name} must be 3x3, got {matrix.shape}")
    out = matrix.copy()
    out.flags.writeable = False
    return out


def _as_tvec(raw: Sequence[float] | Sequence[Sequence[float]]) -> FloatMat:
    vector = np.asarray(raw, dtype=np.float64).reshape(-1)
    if vector.size != 3:
        raise CameraJsonError(path=Path("."), reason=f"t must have 3 entries, got {vector.size}")
    out = vector.reshape(3, 1).copy()
    out.flags.writeable = False
    return out


def _as_dist(raw: Sequence[float] | None) -> FloatMat:
    if raw is None:
        out = np.zeros(5, dtype=np.float64)
        out.flags.writeable = False
        return out
    vector = np.asarray(raw, dtype=np.float64).reshape(-1)
    if vector.size not in {4, 5, 8, 12, 14}:
        raise CameraJsonError(path=Path("."), reason=f"dist length {vector.size} is not an OpenCV model")
    out = vector.copy()
    out.flags.writeable = False
    return out


@dataclass(frozen=True, slots=True)
class CameraPose:
    """已解析的 OpenCV 外参 + 内参。K/R/t 在工厂里拷成只读。"""

    name: str
    width: int
    height: int
    K: FloatMat
    R: FloatMat
    t: FloatMat
    dist: FloatMat

    @classmethod
    def create(
        cls,
        name: str,
        width: int,
        height: int,
        K: Sequence[Sequence[float]] | FloatMat,
        R: Sequence[Sequence[float]] | FloatMat,
        t: Sequence[float] | Sequence[Sequence[float]] | FloatMat,
        dist: Sequence[float] | FloatMat | None = None,
    ) -> CameraPose:
        if width <= 0 or height <= 0:
            raise CameraJsonError(path=Path("."), reason=f"bad image size {width}x{height}")
        return cls(
            name=name,
            width=width,
            height=height,
            K=_as_3x3("K", K),
            R=_as_3x3("R", R),
            t=_as_tvec(t),
            dist=_as_dist(None if dist is None else list(np.asarray(dist, dtype=np.float64).reshape(-1))),
        )

    @property
    def center(self) -> FloatMat:
        """光心 C = −Rᵀ t，世界坐标。"""
        return (-self.R.T @ self.t).reshape(3)

    def project(self, point: WorldXYZ) -> Pixel:
        """世界点投到像素。和 ray_plane.pixel_to_ground 互为逆。"""
        import cv2

        xyz = np.array([[point.x, point.y, point.z]], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(self.R)
        projected, _ = cv2.projectPoints(xyz, rvec, self.t, self.K, self.dist)
        u, v = projected.reshape(2)
        return Pixel(u=float(u), v=float(v))


def blender_world_to_opencv(rotation_c2w: Sequence[Sequence[float]] | FloatMat, camera_world: Sequence[float]) -> tuple[FloatMat, FloatMat]:
    """Blender 相机 matrix_world → OpenCV (R, t)。

    Blender 相机看向局部 −Z、Y 上；OpenCV 看向 +Z、Y 下。
    """
    r_c2w = np.asarray(rotation_c2w, dtype=np.float64).reshape(3, 3)
    center = np.asarray(camera_world, dtype=np.float64).reshape(3)
    rotation = _BCAM_TO_CV @ r_c2w.T
    tvec = -rotation @ center
    return rotation, tvec.reshape(3, 1)


def blender_intrinsics(
    *,
    lens_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    sensor_fit: str,
    pixel_width: int,
    pixel_height: int,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> FloatMat:
    """由镜头毫米数和 sensor_fit 算 K。像素按正方形处理。"""
    if lens_mm <= 0 or sensor_width_mm <= 0 or sensor_height_mm <= 0:
        raise CameraJsonError(path=Path("."), reason="lens/sensor must be positive")
    if pixel_width <= 0 or pixel_height <= 0:
        raise CameraJsonError(path=Path("."), reason="pixel size must be positive")
    fit = sensor_fit.upper()
    use_horizontal = fit != "VERTICAL"
    if fit == "AUTO":
        use_horizontal = pixel_width >= pixel_height
    if use_horizontal:
        fx = lens_mm * pixel_width / sensor_width_mm
        fy = fx
        cx = pixel_width / 2.0 + shift_x * pixel_width
        cy = pixel_height / 2.0 - shift_y * pixel_width
    else:
        fy = lens_mm * pixel_height / sensor_height_mm
        fx = fy
        cx = pixel_width / 2.0 + shift_x * pixel_height
        cy = pixel_height / 2.0 - shift_y * pixel_height
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    k.flags.writeable = False
    return k


def camera_from_path(path: Path) -> CameraPose:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CameraJsonError(path=path, reason="cannot read file") from exc
    except json.JSONDecodeError as exc:
        raise CameraJsonError(path=path, reason=f"invalid json: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise CameraJsonError(path=path, reason="root must be an object")
    if "dist" not in payload and "dist_coeffs" in payload:
        payload = {**payload, "dist": payload["dist_coeffs"]}
    try:
        parsed = _CameraFile.model_validate(payload)
    except ValidationError as exc:
        raise CameraJsonError(path=path, reason=str(exc)) from exc
    return CameraPose.create(
        name=parsed.name,
        width=parsed.image_size[0],
        height=parsed.image_size[1],
        K=parsed.K,
        R=parsed.R,
        t=parsed.t,
        dist=parsed.dist,
    )
