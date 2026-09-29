"""把原图垫成 engine 的正方形输入，再把框还原回原图。

现有 armor/car engine 的 output0 是 letterbox 画布坐标，不是原图像素。
和 Ultralytics 对齐的垫法：等比缩放、居中、灰边 114，left/top 用 round(pad - 0.1)。
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy import float32, uint8
from numpy.typing import NDArray

ImageU8 = NDArray[uint8]
BoxesF = NDArray[float32]


@dataclass(frozen=True, slots=True)
class Letterbox:
    """一次垫图的几何。ratio 是原图到画布的缩放，left/top 是灰边。"""

    ratio: float
    left: int
    top: int
    resized_w: int
    resized_h: int


def letterbox_params(width: int, height: int, size: int) -> Letterbox:
    """只算几何，不碰像素。测试用这一层锁定和 Ultralytics 的对齐。"""
    ratio = min(size / height, size / width)
    resized_w = int(round(width * ratio))
    resized_h = int(round(height * ratio))
    pad_x = (size - resized_w) / 2
    pad_y = (size - resized_h) / 2
    return Letterbox(
        ratio=ratio,
        left=int(round(pad_x - 0.1)),
        top=int(round(pad_y - 0.1)),
        resized_w=resized_w,
        resized_h=resized_h,
    )


def undo_xyxy(boxes: BoxesF, pad: Letterbox) -> BoxesF:
    """画布 xyxy → 原图 xyxy。调用方再按 conf / max_det 截断。"""
    restored = boxes.copy()
    restored[:, 0] = (restored[:, 0] - pad.left) / pad.ratio
    restored[:, 2] = (restored[:, 2] - pad.left) / pad.ratio
    restored[:, 1] = (restored[:, 1] - pad.top) / pad.ratio
    restored[:, 3] = (restored[:, 3] - pad.top) / pad.ratio
    return restored
