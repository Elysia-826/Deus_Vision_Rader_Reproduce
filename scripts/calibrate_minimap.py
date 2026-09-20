# ─── How to run ───
#     python scripts/calibrate_minimap.py
# 左上、右上、右下、左下各点一次场地外框，回车写入 locate/calib/minimap.json
# ──────────────────

"""点官方俯视图四角，写出小地图标定。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from detect.errors import ImageReadError
from locate.field import FIELD_LENGTH_M, FIELD_WIDTH_M

IMAGE = _ROOT / "scene" / "materials" / "rmuc_fig4_1_topdown.png"
OUT = _ROOT / "locate" / "calib" / "minimap.json"
LABELS = ("TL", "TR", "BR", "BL")


def main() -> None:
    image = cv2.imread(str(IMAGE))
    if image is None:
        raise ImageReadError(IMAGE)
    clicks: list[tuple[float, float]] = []
    canvas = image.copy()

    def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN or len(clicks) >= 4:
            return
        clicks.append((float(x), float(y)))
        cv2.circle(canvas, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(canvas, LABELS[len(clicks) - 1], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("minimap calib", canvas)

    cv2.imshow("minimap calib", canvas)
    cv2.setMouseCallback("minimap calib", on_mouse)
    print(f"click {', '.join(LABELS)} on the {FIELD_LENGTH_M:.0f}x{FIELD_WIDTH_M:.0f}m field border, then press any key")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(clicks) != 4:
        raise SystemExit(f"need 4 clicks, got {len(clicks)}")
    height, width = image.shape[:2]
    payload = {"image_size": [width, height], "image_corners": [[u, v] for u, v in clicks]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
