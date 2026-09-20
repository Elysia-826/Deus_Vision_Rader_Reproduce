# ─── How to run ───
# 先用 Blender dump 相机，再叠检测和小地图：
#
#     & "D:\Program Files\blender\blender.exe" scene/RMUC2026_V2.0.0_radar.blend --background --python scene/armor_assets/dump_radar_cameras.py -- --render
#     python scripts/visualize_minimap.py
# ──────────────────

"""检测框底边 → 平面射线 → 官方俯视图。不改 detect 包。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy import uint8
from numpy.typing import NDArray

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

import visualize_two_stage as vis2  # noqa: E402
from detect.types import LinkedRobot  # noqa: E402
from locate.camera import camera_from_path  # noqa: E402
from locate.minimap import minimap_from_path  # noqa: E402
from locate.ray_plane import foot_pixel, pixel_to_ground, world_to_pixel  # noqa: E402
from locate.types import FieldXY, Pixel, WorldXYZ  # noqa: E402

CAMERA_JSON = _ROOT / "locate" / "calib" / "Camera_Radar_Blue.json"
MINIMAP_JSON = _ROOT / "locate" / "calib" / "minimap.json"
MINIMAP_IMAGE = _ROOT / "scene" / "materials" / "rmuc_fig4_1_topdown.png"
SOURCE_PATH = _ROOT / "scene" / "locate_out" / "locate_frame.png"
GT_JSON = _ROOT / "scene" / "locate_out" / "gt.json"
OUT_PATH = _ROOT / "scene" / "locate_out" / "minimap_overlay.png"
SHOW_WINDOW = True

ImageU8 = NDArray[uint8]


def _tag(robot: LinkedRobot) -> str | None:
    """小地图车号：图案分类 1/2/3/4/S/Q + 装甲颜色。没有图案就不标，不用跟踪 ID。"""
    best_conf = -1.0
    best: str | None = None
    for armor in robot.armors:
        if armor.pattern is None:
            continue
        if armor.pattern.conf <= best_conf:
            continue
        match armor.label:
            case "red":
                prefix = "R"
            case "blue":
                prefix = "B"
            case "dead":
                prefix = "G"
            case _:
                prefix = ""
        best_conf = armor.pattern.conf
        best = f"{prefix}{armor.pattern.name}"
    return best


def _bgr(robot: LinkedRobot) -> tuple[int, int, int]:
    for armor in robot.armors:
        match armor.label:
            case "red":
                return (0, 0, 255)
            case "blue":
                return (255, 0, 0)
            case "dead":
                return (180, 180, 180)
            case _:
                return (0, 255, 255)
    return (0, 255, 0)


def draw_foot(frame: ImageU8, robot: LinkedRobot, tag: str | None) -> Pixel:
    pixel = foot_pixel(robot.car.box)
    pt = (int(round(pixel.u)), int(round(pixel.v)))
    cv2.circle(frame, pt, 6, _bgr(robot), -1)
    if tag is not None:
        cv2.putText(frame, tag, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _bgr(robot), 2)
    return pixel


def stack_horizontal(left: ImageU8, right: ImageU8) -> ImageU8:
    height = max(left.shape[0], right.shape[0])

    def _fit(image: ImageU8) -> ImageU8:
        if image.shape[0] == height:
            return image
        scale = height / image.shape[0]
        return cv2.resize(image, (int(image.shape[1] * scale), height))

    return np.hstack((_fit(left), _fit(right)))


def load_gt(path: Path) -> tuple[FieldXY, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    robots = raw["robots"]
    return tuple(FieldXY(x=float(item["x"]), y=float(item["y"])) for item in robots)


def nearest_error(hit: FieldXY, truth: tuple[FieldXY, ...]) -> float:
    return min(((hit.x - gt.x) ** 2 + (hit.y - gt.y) ** 2) ** 0.5 for gt in truth)


def main() -> None:
    pose = camera_from_path(CAMERA_JSON)
    calib = minimap_from_path(MINIMAP_JSON)
    frame = cv2.imread(str(SOURCE_PATH))
    if frame is None:
        raise vis2.ImageReadError(SOURCE_PATH)
    minimap = cv2.imread(str(MINIMAP_IMAGE))
    if minimap is None:
        raise vis2.ImageReadError(MINIMAP_IMAGE)

    net, transform, device = vis2._load_classifier()
    detector = vis2._load_detector(
        vis2.PatternStage(net=net, transform=transform, device=str(device))
    )
    vis2.warmup(detector, frame)
    result = detector.infer(frame)
    canvas = vis2.draw_result(frame, result, 0.0)
    board = minimap.copy()
    truth = load_gt(GT_JSON) if GT_JSON.is_file() else ()
    for gt in truth:
        cam_px = world_to_pixel(pose, WorldXYZ(gt.x, gt.y, 0.0))
        cv2.circle(canvas, (int(round(cam_px.u)), int(round(cam_px.v))), 6, (0, 255, 255), 2)
        px = calib.field_to_pixel(gt)
        cv2.circle(board, (int(px.u), int(px.v)), 8, (0, 255, 255), 2)
    for robot in result.robots:
        tag = _tag(robot)
        pixel = draw_foot(canvas, robot, tag)
        hit = pixel_to_ground(pose, pixel)
        if hit is None:
            print(f"{tag} ray missed ground  pixel=({pixel.u:.1f},{pixel.v:.1f})")
            continue
        mapped = calib.field_to_pixel(hit)
        color = _bgr(robot)
        cv2.circle(board, (int(mapped.u), int(mapped.v)), 7, color, -1)
        if tag is not None:
            cv2.putText(board, tag, (int(mapped.u) + 8, int(mapped.v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        err = f"  err={nearest_error(hit, truth):.2f}m" if truth else ""
        print(f"{tag}  uv=({pixel.u:.1f},{pixel.v:.1f})  xy=({hit.x:.2f},{hit.y:.2f}){err}")
    overlay = stack_horizontal(canvas, board)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PATH), overlay)
    print(f"saved {OUT_PATH}")
    if SHOW_WINDOW:
        cv2.imshow("minimap", vis2._scale(overlay))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
