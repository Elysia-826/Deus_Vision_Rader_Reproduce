"""检测链路边界错误。空裁剪、缺权重、读图失败都是调用方给的路径/框问题，不是模型内部状态。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from detect.types import BBox


@dataclass(frozen=True, slots=True)
class EmptyCropError(Exception):
    box: BBox

    def __str__(self) -> str:
        return f"empty car crop: {self.box}"


@dataclass(frozen=True, slots=True)
class ModelFileNotFoundError(Exception):
    candidates: tuple[Path, ...]

    def __str__(self) -> str:
        listed = ", ".join(str(path) for path in self.candidates)
        return f"no model file among: {listed}"


@dataclass(frozen=True, slots=True)
class ImageReadError(Exception):
    path: Path

    def __str__(self) -> str:
        return f"cannot read image or video: {self.path}"
