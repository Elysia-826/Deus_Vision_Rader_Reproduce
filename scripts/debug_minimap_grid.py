# ─── How to run ───
#     python scripts/debug_minimap_grid.py
# ──────────────────

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from locate.field import FIELD_X_MAX, FIELD_X_MIN, FIELD_Y_MAX, FIELD_Y_MIN, field_corners
from locate.minimap import minimap_from_path
from locate.types import FieldXY

OUT = _ROOT / "scene" / "locate_out" / "minimap_grid.png"


def main() -> None:
    board = cv2.imread(str(_ROOT / "scene" / "materials" / "rmuc_fig4_1_topdown.png"))
    if board is None:
        raise SystemExit("missing topdown png")
    calib = minimap_from_path(_ROOT / "locate" / "calib" / "minimap.json")
    for a, b in (
        (FieldXY(FIELD_X_MIN, 0.0), FieldXY(FIELD_X_MAX, 0.0)),
        (FieldXY(0.0, FIELD_Y_MIN), FieldXY(0.0, FIELD_Y_MAX)),
    ):
        pa, pb = calib.field_to_pixel(a), calib.field_to_pixel(b)
        cv2.line(board, (int(pa.u), int(pa.v)), (int(pb.u), int(pb.v)), (0, 255, 255), 2)
    for i, corner in enumerate(field_corners()):
        p = calib.field_to_pixel(corner)
        cv2.circle(board, (int(p.u), int(p.v)), 8, (0, 0, 255), -1)
        cv2.putText(board, str(i), (int(p.u) + 6, int(p.v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    origin = calib.field_to_pixel(FieldXY(0.0, 0.0))
    cv2.circle(board, (int(origin.u), int(origin.v)), 10, (255, 0, 255), 2)
    cv2.putText(board, "origin", (int(origin.u) + 8, int(origin.v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    plus_x = calib.field_to_pixel(FieldXY(8.0, 0.0))
    cv2.putText(board, "+X", (int(plus_x.u), int(plus_x.v)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    plus_y = calib.field_to_pixel(FieldXY(0.0, 5.0))
    cv2.putText(board, "+Y", (int(plus_y.u), int(plus_y.v)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    gt = json.loads((_ROOT / "scene" / "locate_out" / "gt.json").read_text(encoding="utf-8"))
    for robot in gt["robots"]:
        p = calib.field_to_pixel(FieldXY(float(robot["x"]), float(robot["y"])))
        cv2.circle(board, (int(p.u), int(p.v)), 7, (255, 255, 0), -1)
        cv2.putText(
            board,
            f"{robot['x']:.0f},{robot['y']:.0f}",
            (int(p.u) + 6, int(p.v) + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
        )
    cv2.imwrite(str(OUT), board)
    print("saved", OUT)


if __name__ == "__main__":
    main()
