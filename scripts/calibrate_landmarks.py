# ─── How to run ───
#     python scripts/calibrate_landmarks.py
#     python scripts/calibrate_landmarks.py path\to\frame.jpg
# 按提示点地面落点：蓝前哨 → 红前哨 → 能量机关 → 蓝基地。z 撤销，回车保存。
# ──────────────────

"""赛场四点标定：相机画面地标 → locate/calib/match_homography.json。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from detect.errors import ImageReadError
from locate.homography import dump_homography_pairs, homography_from_pairs
from locate.landmarks import landmark_by_id, landmarks_from_path
from locate.types import Pixel

LANDMARKS = _ROOT / "locate" / "calib" / "landmarks.json"
OUT = _ROOT / "locate" / "calib" / "match_homography.json"
CANDIDATES = (
    _ROOT / "scene" / "locate_out" / "live_frame.jpg",
    _ROOT / "scene" / "locate_out" / "locate_frame.png",
)


def _image_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for path in CANDIDATES:
        if path.is_file():
            return path
    raise ImageReadError(CANDIDATES[0])


def main() -> None:
    table, order = landmarks_from_path(LANDMARKS)
    image_path = _image_path()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ImageReadError(image_path)
    clicks: list[Pixel] = []
    canvas = image.copy()

    def redraw() -> None:
        nonlocal canvas
        canvas = image.copy()
        for index, pixel in enumerate(clicks):
            pt = (int(round(pixel.u)), int(round(pixel.v)))
            cv2.circle(canvas, pt, 7, (0, 255, 255), -1)
            label = landmark_by_id(table, order[index]).name_zh
            cv2.putText(canvas, label, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        remain = order[len(clicks)] if len(clicks) < len(order) else "回车保存"
        hint = f"{len(clicks)}/{len(order)}  next={remain}  z=undo  enter=save"
        cv2.putText(canvas, hint, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("landmark calib", canvas)

    def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN or len(clicks) >= len(order):
            return
        clicks.append(Pixel(u=float(x), v=float(y)))
        redraw()

    cv2.imshow("landmark calib", canvas)
    cv2.setMouseCallback("landmark calib", on_mouse)
    redraw()
    print("click ground footprints:", ", ".join(landmark_by_id(table, item).name_zh for item in order))
    print("image", image_path)
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in {ord("z"), ord("Z")} and clicks:
            clicks.pop()
            redraw()
        elif key in {13, 10}:
            break
        elif key in {ord("q"), 27}:
            cv2.destroyAllWindows()
            raise SystemExit("cancelled")
    cv2.destroyAllWindows()
    if len(clicks) < 4:
        raise SystemExit(f"need at least 4 clicks, got {len(clicks)}")
    used = order[: len(clicks)]
    fields = tuple(landmark_by_id(table, item).xy for item in used)
    mapped = homography_from_pairs(tuple(clicks), fields, used)
    center = mapped.pixel_to_field(Pixel(image.shape[1] / 2.0, image.shape[0] / 2.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dump_homography_pairs(used, tuple(clicks), fields), indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(f"image center -> field ({center.x:.2f}, {center.y:.2f}) m")


if __name__ == "__main__":
    main()
