# ─── How to run ───
#     python scripts/calibrate_pnp.py frame.jpg --intrinsics locate/calib/Camera_Radar_Blue.json
#     python scripts/calibrate_pnp.py frame.jpg --intrinsics ... --out locate/calib/Camera_Field.json
# 按提示点地标（顺序读 landmarks.json 的 click_order）。每次点击弹出 4 倍放大镜，
# WASD 像素级微调（Shift=8px 步进），Enter/n 确认，z/ESC 取消本次点击；
# 主窗口 z 撤销上一个点，回车求解存盘。
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

# 放大镜微调（借鉴 T-DT 标定交互）：点击后取周围 ±ZOOM_HALF px 放大 ZOOM_SCALE 倍，
# 键盘步进把"手抖 3-5px"压到亚像素级操作。放大用最近邻，像素边界清晰。
ZOOM_HALF = 64
ZOOM_SCALE = 4
_FAST_STEP = 8  # Shift+WASD 的步进


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

    def zoom_pick(start_x: int, start_y: int) -> Pixel | None:
        """放大镜微调一次点击。返回确认后的像素；取消返回 None。

        WASD/wasd 步进（Shift=8px），Enter/n/space 确认，z/ESC 取消本次点击。
        ROI 窗口向图像边缘方向平移而非截断准心，保证准心始终可见。
        """
        height, width = image.shape[:2]
        x, y = start_x, start_y
        name = "pnp zoom"
        while True:
            # ROI 贴边时整体平移，准心保持在窗内（不在边缘被裁掉）
            roi_x = min(max(x - ZOOM_HALF, 0), max(width - 2 * ZOOM_HALF, 0))
            roi_y = min(max(y - ZOOM_HALF, 0), max(height - 2 * ZOOM_HALF, 0))
            roi_w = min(2 * ZOOM_HALF, width - roi_x)
            roi_h = min(2 * ZOOM_HALF, height - roi_y)
            zoom = cv2.resize(
                image[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w],
                (roi_w * ZOOM_SCALE, roi_h * ZOOM_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            cx = (x - roi_x) * ZOOM_SCALE
            cy = (y - roi_y) * ZOOM_SCALE
            bar = 30
            cv2.line(zoom, (cx, cy - bar), (cx, cy + bar), (0, 0, 255), 1)
            cv2.line(zoom, (cx - bar, cy), (cx + bar, cy), (0, 0, 255), 1)
            cv2.putText(
                zoom, f"x={x} y={y}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.putText(
                zoom, "WASD move(Shift=8px)  Enter/n=ok  z/ESC=cancel",
                (8, zoom.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
            )
            cv2.imshow(name, zoom)
            key = cv2.waitKey(20) & 0xFF
            if key in {ord("w"), ord("W")}:
                y -= _FAST_STEP if key == ord("W") else 1
            elif key in {ord("s"), ord("S")}:
                y += _FAST_STEP if key == ord("S") else 1
            elif key in {ord("a"), ord("A")}:
                x -= _FAST_STEP if key == ord("A") else 1
            elif key in {ord("d"), ord("D")}:
                x += _FAST_STEP if key == ord("D") else 1
            elif key in {13, 10, ord("n"), ord("N"), ord(" ")}:
                cv2.destroyWindow(name)
                return Pixel(u=float(x), v=float(y))
            elif key in {27, ord("z"), ord("Z")}:
                cv2.destroyWindow(name)
                return None
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)

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
        picked = zoom_pick(x, y)
        if picked is None:
            return
        clicks.append(picked)
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
