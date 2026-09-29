"""端到端 YOLO 的 (N, 6) 输出：xyxy、conf、cls，已经在 engine 里做过 NMS。"""

from __future__ import annotations

from dataclasses import dataclass

from numpy import argsort, float32
from numpy.typing import NDArray

BoxesF = NDArray[float32]


@dataclass(frozen=True, slots=True)
class E2EDets:
    """原图像素上的检出。空检出用长度为 0 的数组，不用 None。"""

    xyxy: BoxesF
    conf: NDArray[float32]
    cls: NDArray[float32]


def select_e2e(rows: BoxesF, conf: float, max_det: int) -> E2EDets:
    """丢掉低分，再按分数取前 max_det。rows 的第 4 列是 conf，第 5 列是类别。"""
    if rows.size == 0:
        empty = rows.reshape(0, 4)
        return E2EDets(xyxy=empty, conf=rows.reshape(0), cls=rows.reshape(0))
    kept = rows[rows[:, 4] >= conf]
    if kept.shape[0] > max_det:
        order = argsort(-kept[:, 4])[:max_det]
        kept = kept[order]
    return E2EDets(xyxy=kept[:, :4], conf=kept[:, 4], cls=kept[:, 5])
