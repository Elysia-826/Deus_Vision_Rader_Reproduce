from __future__ import annotations

import numpy as np

from detect.e2e import select_e2e
from detect.letterbox import letterbox_params, undo_xyxy


def test_undo_xyxy_matches_armor_engine_probe() -> None:
    # 62x77 装甲图垫到 192 后，engine 原始框与 Ultralytics xyxy 差在 0.05 像素内。
    pad = letterbox_params(width=62, height=77, size=192)
    raw = np.array([[117.01, 126.33, 159.48, 172.04]], dtype=np.float32)
    undone = undo_xyxy(raw, pad)
    assert abs(float(undone[0, 0]) - 39.706) < 0.05
    assert abs(float(undone[0, 1]) - 50.662) < 0.05
    assert abs(float(undone[0, 2]) - 56.739) < 0.05
    assert abs(float(undone[0, 3]) - 68.996) < 0.05


def test_select_e2e_drops_low_conf_and_caps_count() -> None:
    rows = np.array(
        [
            [0, 0, 10, 10, 0.9, 1],
            [1, 1, 8, 8, 0.2, 0],
            [2, 2, 9, 9, 0.5, 2],
        ],
        dtype=np.float32,
    )
    picked = select_e2e(rows, conf=0.3, max_det=1)
    assert picked.conf.shape == (1,)
    assert abs(float(picked.conf[0]) - 0.9) < 1e-6
    assert int(picked.cls[0]) == 1
