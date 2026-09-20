"""定位边界错误。JSON 坏、矩阵形状不对、标定点数不够，都是调用方给的文件问题。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CameraJsonError(Exception):
    path: Path
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class SourceNotReadyError(Exception):
    source: str

    def __str__(self) -> str:
        return f"source {self.source} is not wired yet"


@dataclass(frozen=True, slots=True)
class MinimapCalibError(Exception):
    path: Path | None
    reason: str

    def __str__(self) -> str:
        if self.path is None:
            return self.reason
        return f"{self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class HomographyError(Exception):
    path: Path | None
    reason: str

    def __str__(self) -> str:
        if self.path is None:
            return self.reason
        return f"{self.path}: {self.reason}"
