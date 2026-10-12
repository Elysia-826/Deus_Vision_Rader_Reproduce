# ─── How to run ───
#     python scripts/calibrate_pnp.py frame.jpg --intrinsics locate/calib/Camera_Radar_Blue.json
#     python scripts/calibrate_pnp.py frame.jpg --intrinsics ... --out locate/calib/Camera_Field.json
# 按提示点地标（顺序读 landmarks.json 的 click_order）。z 撤销，回车求解存盘。
# ──────────────────

"""真机 PnP 标定：地标点击 → CameraPose JSON（LOCATE_MODE=ray 直接可用）。

和 calibrate_landmarks.py 的区别：单应只解平面，PnP 解完整 R/t，
斜坡/高地下不再有平面假设。要求 landmarks.json 的 click_order 至少 4 个、
不全共线；带 z 高度的地标（塔尖/建筑顶）条件数更好。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from detect.errors import ImageReadError
from locate.camera import camera_from_path
from locate.landmarks import landmark_by_id, landmarks_from_path
from locate.pnp import dump_camera_json, solve_pose
from locate.types import Pixel, WorldXYZ

LANDMARKS = _ROOT / "locate" / "calib" / "landmarks.json"
MAX_RESIDUAL_PX = 12.0  # 点击很糙也该 <12px；超了就是点错或坐标表错


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="标定帧（上场抓一帧高清）")
    parser.add_argument("--intrinsics", type=Path, required=True, help="内参模板 json（K/dist/分辨率）")
    parser.add_argument("--landmarks", type=Path, default=LANDMARKS)
    parser.add_argument("--out", type=Path, default=_ROOT / "locate" / "calib" / "Camera_Field.json")
    args = parser.parse_args()

    table, order = landmarks_from_path(args.landmarks)
    template = camera_from_path(args.intrinsics)
    image = cv2.imread(str(args.image))
    if image is None:
        raise ImageReadError(args.image)
    if (image.shape[1], image.shape[0]) != (template.width, template.height):
        raise SystemExit(
            f"图像 {image.shape[1]}x{image.shape[0]} 和内参 {template.width}x{template.height} 不一致"
        )

    clicks: list[Pixel] = []

    def redraw(hint: str) -> None:
        canvas = image.copy()
        for index, pixel in enumerate(clicks):
            pt = (int(round(pixel.u)), int(round(pixel.v)))
            cv2.circle(canvas, pt, 7, (0, 255, 255), -1)
            cv2.putText(
                canvas, landmark_by_id(table, order[index]).name_zh,
                (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
        cv2.putText(canvas, hint, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("pnp calib", canvas)

    def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN or len(clicks) >= len(order):
            return
        clicks.append(Pixel(u=float(x), v=float(y)))
        redraw(_hint(clicks, order, table))

    cv2.imshow("pnp calib", image)
    cv2.setMouseCallback("pnp calib", on_mouse)
    redraw(_hint(clicks, order, table))
    print("click landmarks:", ", ".join(landmark_by_id(table, item).name_zh for item in order))
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in {ord("z"), ord("Z")} and clicks:
            clicks.pop()
            redraw(_hint(clicks, order, table))
        elif key in {13, 10} and len(clicks) >= 4:
            break
    cv2.destroyAllWindows()

    pairs: list[tuple[WorldXYZ, Pixel]] = []
    for landmark_id, pixel in zip(order, clicks, strict=False):
        landmark = landmark_by_id(table, landmark_id)
        pairs.append((WorldXYZ(x=landmark.xy.x, y=landmark.xy.y, z=landmark.z), pixel))
    result = solve_pose(
        name=f"{template.name}_pnp",
        width=template.width,
        height=template.height,
        K=template.K,
        dist=template.dist,
        pairs=pairs,
    )
    print(f"residual  rms {result.rms_px:.2f}px  max {result.max_px:.2f}px")
    if result.max_px > MAX_RESIDUAL_PX:
        raise SystemExit("残差过大：有点点错/坐标表与实物不符，先别用这个外参")
    dump_camera_json(result.pose, args.out)
    print("saved", args.out, "- run_radar: LOCATE_MODE=ray + poses 指向该文件即可")


def _hint(clicks: list[Pixel], order: tuple[str, ...], table) -> str:
    if len(clicks) < len(order):
        next_name = landmark_by_id(table, order[len(clicks)]).name_zh
    else:
        next_name = "回车求解"
    return f"{len(clicks)}/{len(order)}  next={next_name}  z=undo  enter=solve"


if __name__ == "__main__":
    main()
