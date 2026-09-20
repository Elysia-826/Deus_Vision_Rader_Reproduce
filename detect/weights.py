"""按优先级挑第一个存在的权重。engine 优先于 pt：同卡不要混 PyTorch cuDNN 和 TensorRT。"""

from __future__ import annotations

from pathlib import Path

from detect.errors import ModelFileNotFoundError


def first_existing(candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        resolved = path.expanduser()
        if resolved.is_file():
            return resolved
    raise ModelFileNotFoundError(candidates=candidates)
