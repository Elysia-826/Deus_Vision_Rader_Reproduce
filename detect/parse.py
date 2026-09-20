"""Ultralytics 检出 → Detection。只在这一层碰原始 tensor。

xyxy/conf/cls/id 从 GPU tensor 拉到 Python 列表之后，pipeline 不再碰 boxes。
track id 只在 model.track 时存在；predict 的 boxes.id 为 None，保持 Detection.track_id=None。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from detect.types import BBox, Detection


def detections_from_rows(
    xyxy_rows: Sequence[Sequence[float]],
    confs: Sequence[float],
    class_ids: Sequence[int],
    names: Mapping[int, str],
    track_ids: Sequence[int | None] | None = None,
) -> tuple[Detection, ...]:
    """三列等长。names 缺 id 时用数字字符串，避免推理中途 KeyError。"""
    ids: Sequence[int | None] = track_ids if track_ids is not None else (None,) * len(confs)
    parsed: list[Detection] = []
    for xyxy, conf, class_id, track_id in zip(xyxy_rows, confs, class_ids, ids, strict=True):
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        label = names[class_id] if class_id in names else str(class_id)
        parsed.append(
            Detection(
                label=label,
                conf=float(conf),
                box=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                track_id=track_id,
            )
        )
    return tuple(parsed)
