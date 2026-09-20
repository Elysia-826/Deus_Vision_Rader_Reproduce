"""图案分类数据边界错误。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PatternDatasetError(Exception):
    """--data 不是可训练的图案分类根目录。"""

    path: Path
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"
