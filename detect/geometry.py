"""车辆框裁剪，以及装甲板框从 ROI 回投全图。

装甲 YOLO 看到的是 192×192 裁剪里的局部坐标。报点和画框必须加回车辆框左上角，
否则全图上装甲会贴在 (0,0) 附近。clamp 是因为 YOLO 框会略超出图像，numpy 切片会空。
"""

from __future__ import annotations

from numpy import uint8
from numpy.typing import NDArray

from detect.errors import EmptyCropError
from detect.types import BBox

ImageU8 = NDArray[uint8]


def clamp_box(box: BBox, width: int, height: int) -> BBox | None:
    """把框裁进图像。完全在外或面积为 0 则没有可用 ROI。"""
    x1 = max(box.x1, 0)
    y1 = max(box.y1, 0)
    x2 = min(box.x2, width)
    y2 = min(box.y2, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


def crop_roi(image: ImageU8, box: BBox) -> ImageU8:
    """按已夹紧的框切片。调用方必须先 clamp，否则空框直接炸。"""
    if box.width <= 0 or box.height <= 0:
        raise EmptyCropError(box=box)
    return image[box.y1 : box.y2, box.x1 : box.x2]


def remap_box(local: BBox, origin: BBox) -> BBox:
    """ROI 内 xyxy 加上车辆左上角，得到全图像素坐标。"""
    return BBox(
        x1=local.x1 + origin.x1,
        y1=local.y1 + origin.y1,
        x2=local.x2 + origin.x1,
        y2=local.y2 + origin.y1,
    )
