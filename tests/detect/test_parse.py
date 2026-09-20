from __future__ import annotations

from detect.parse import detections_from_rows
from detect.types import BBox


def test_detections_from_rows_uses_class_name() -> None:
    dets = detections_from_rows(
        xyxy_rows=[[1.9, 2.1, 10.0, 20.0]],
        confs=[0.8],
        class_ids=[2],
        names={0: "dead", 1: "red", 2: "blue"},
    )
    assert len(dets) == 1
    assert dets[0].label == "blue"
    assert dets[0].conf == 0.8
    assert dets[0].box == BBox(x1=1, y1=2, x2=10, y2=20)


def test_detections_from_rows_falls_back_to_id() -> None:
    dets = detections_from_rows([[0.0, 0.0, 4.0, 4.0]], [0.1], [9], {0: "robot"})
    assert dets[0].label == "9"
