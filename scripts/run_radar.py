# ─── How to run ───
# 终端 1（仿真出图）：
#     & "D:\Program Files\blender\blender.exe" scene/RMUC2026_V2.0.0_radar.blend --python scene/armor_assets/radar_sim_loop.py -- --motion patrol
# 终端 2（检测 + 小地图）：
#     python scripts/run_radar.py
# SOURCE=video 时不需要 Blender。q 退出。
# ──────────────────

"""主循环：图像源 → 三级检测 → 平面射线 → 相机画面内嵌小地图。"""

from __future__ import annotations

import json
import sys
import time
from enum import StrEnum
from typing import assert_never
from pathlib import Path

import cv2
from numpy import frombuffer, uint8, zeros
from numpy.typing import NDArray

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPTS))

import visualize_minimap as vis_map  # noqa: E402
import visualize_two_stage as vis2  # noqa: E402
from locate.camera import camera_from_path  # noqa: E402
from locate.ray_mesh import FieldMesh  # noqa: E402
from locate.errors import SourceNotReadyError  # noqa: E402
from locate.homography import HomographyMap, homography_from_path  # noqa: E402
from locate.minimap import minimap_from_path  # noqa: E402

from locate.types import FieldXY  # noqa: E402
from track.cascade import CascadeMatchTracker  # noqa: E402
from track.errors import TruthFieldError, UnmappedRobotName  # noqa: E402
from track.log import append_frame  # noqa: E402
from track.observe import observations_from_result  # noqa: E402
from track.truth import parse_sim_robots  # noqa: E402
from track.types import PublishedTrack, SimRobot, Team  # noqa: E402

# ========== 开关 ==========
SOURCE = "blender"  # blender | video | hik
LOCATE_MODE = "ray"  # ray | homography
VIDEO_PATH = str(_ROOT / "scripts" / "RM_TestVideo.mp4")
CAMERA_JSON = _ROOT / "locate" / "calib" / "Camera_Radar_Blue.json"
FIELD_MESH = _ROOT / "locate" / "calib" / "field_mesh.npz"
HOMOGRAPHY_JSON = _ROOT / "locate" / "calib" / "match_homography.json"
MINIMAP_JSON = _ROOT / "locate" / "calib" / "minimap.json"
MINIMAP_IMAGE = _ROOT / "scene" / "materials" / "rmuc_fig4_1_topdown.png"
LIVE_PNG = _ROOT / "scene" / "locate_out" / "live_frame.png"
LIVE_META = _ROOT / "scene" / "locate_out" / "live_meta.json"
SHOW_GT = False
TRAIL_LEN = 24
IDENTITY_LOG = _ROOT / "scene" / "locate_out" / "identity_log.jsonl"
# =========================

ImageU8 = NDArray[uint8]


class FrameSource(StrEnum):
    BLENDER = "blender"
    VIDEO = "video"
    HIK = "hik"


def _read_meta() -> tuple[int, Path, tuple[SimRobot, ...]] | None:
    if not LIVE_META.is_file():
        return None
    try:
        raw = json.loads(LIVE_META.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    image = Path(str(raw["image"])) if "image" in raw else LIVE_PNG
    if not image.is_file():
        alt = LIVE_PNG.with_suffix(".jpg")
        if not alt.is_file():
            return None
        image = alt
    frame_id = int(raw["frame"])
    try:
        robots = _robots(raw)
    except (TruthFieldError, UnmappedRobotName) as exc:
        print(exc, flush=True)
        robots = ()
    return frame_id, image, robots


def _robots(raw: dict[str, object]) -> tuple[SimRobot, ...]:  # noqa: OBJECT_OK
    items = raw.get("robots", [])
    if not isinstance(items, list):
        raise TruthFieldError(key="robots", reason="not a list")
    mappings: list[dict[str, str | int | float | bool | None]] = []
    for item in items:
        if not isinstance(item, dict):
            raise TruthFieldError(key="robots", reason="item is not an object")
        mappings.append(item)
    return parse_sim_robots(mappings)


def _dot(board: ImageU8, u: float, v: float, color: tuple[int, int, int], tag: str) -> None:
    pt = (int(round(u)), int(round(v)))
    cv2.circle(board, pt, 11, color, -1)
    cv2.circle(board, pt, 11, (0, 0, 0), 2)
    cv2.putText(board, tag, (pt[0] + 12, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def _draw_trail(board: ImageU8, pts: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        cv2.line(board, pts[i - 1], pts[i], color, 2)


def inset_minimap(camera: ImageU8, board: ImageU8) -> ImageU8:
    """Sports RADAR：小地图贴在主画面右下，半透明。"""
    h, w = camera.shape[:2]
    rw = max(w // 2, 1)
    rh = max(int(board.shape[0] * rw / board.shape[1]), 1)
    radar = cv2.resize(board, (rw, rh))
    x = w - rw - 12
    y = h - rh - 12
    x = max(x, 0)
    y = max(y, 0)
    roi = camera[y : y + rh, x : x + rw]
    if roi.shape[0] != rh or roi.shape[1] != rw:
        return camera
    camera[y : y + rh, x : x + rw] = cv2.addWeighted(radar, 0.78, roi, 0.22, 0)
    cv2.rectangle(camera, (x, y), (x + rw - 1, y + rh - 1), (255, 255, 255), 2)
    return camera


def _slot_bgr(team: Team) -> tuple[int, int, int]:
    match team:
        case Team.RED:
            return (0, 0, 255)
        case Team.BLUE:
            return (255, 0, 0)
        case unreachable:
            assert_never(unreachable)


def overlay(
    frame: ImageU8,
    result: vis2.FrameResult,
    calib,
    minimap: ImageU8,
    truth: tuple[SimRobot, ...],
    publishes: tuple[PublishedTrack, ...],
    trails: dict[str, list[tuple[int, int]]],
    fps: float,
) -> ImageU8:
    canvas = vis2.draw_result(frame, result, fps)
    board = minimap.copy()
    if SHOW_GT:
        for gt in truth:
            px = calib.field_to_pixel(FieldXY(x=gt.x, y=gt.y))
            cv2.circle(board, (int(px.u), int(px.v)), 8, (0, 255, 255), 2)
            cv2.putText(board, gt.label, (int(px.u) + 10, int(px.v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    by_bot = {item.bot_id: item.label for item in publishes if item.bot_id is not None}
    for robot in result.robots:
        vis_map.draw_foot(canvas, robot, by_bot.get(robot.car.track_id))
    seen: set[str] = set()
    for item in publishes:
        mapped = calib.field_to_pixel(FieldXY(x=item.x, y=item.y))
        color = _slot_bgr(item.team)
        pt = (int(round(mapped.u)), int(round(mapped.v)))
        trail = trails.setdefault(item.label, [])
        trail.append(pt)
        del trail[:-TRAIL_LEN]
        seen.add(item.label)
        _draw_trail(board, trail, color)
        _dot(board, mapped.u, mapped.v, color, item.label)
    for tag in list(trails):
        if tag not in seen:
            trails.pop(tag, None)
    return inset_minimap(canvas, board)


def main() -> None:
    source = FrameSource(SOURCE)
    match source:
        case FrameSource.HIK:
            raise SourceNotReadyError(source=source.value)
        case FrameSource.BLENDER | FrameSource.VIDEO:
            pass
        case unreachable:
            raise SourceNotReadyError(source=str(unreachable))

    pose = None
    homography: HomographyMap | None = None
    mesh: FieldMesh | None = None
    if LOCATE_MODE == "homography":
        homography = homography_from_path(HOMOGRAPHY_JSON)
    else:
        pose = camera_from_path(CAMERA_JSON)
        if FIELD_MESH.is_file():
            mesh = FieldMesh.from_npz(FIELD_MESH)
            print("run_radar mesh", FIELD_MESH.name, flush=True)
    calib = minimap_from_path(MINIMAP_JSON)
    minimap = cv2.imread(str(MINIMAP_IMAGE))
    if minimap is None:
        raise vis2.ImageReadError(MINIMAP_IMAGE)

    cap: cv2.VideoCapture | None = None
    if source is FrameSource.VIDEO:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise vis2.ImageReadError(Path(VIDEO_PATH))

    print("run_radar source", source.value, "loading detector…", flush=True)
    net, transform, device = vis2._load_classifier()
    detector = vis2._load_detector(vis2.PatternStage(net=net, transform=transform, device=str(device)))
    vis2.warmup(detector, zeros((3000, 4096, 3), dtype=uint8))
    last_frame = -1
    trails: dict[str, list[tuple[int, int]]] = {}
    tracker = CascadeMatchTracker()
    last_tick: float | None = None
    if source is FrameSource.BLENDER:
        IDENTITY_LOG.parent.mkdir(parents=True, exist_ok=True)
        IDENTITY_LOG.write_text("", encoding="utf-8")
    print("run_radar waiting for frames…", flush=True)
    while True:
        truth: tuple[SimRobot, ...] = ()
        frame: ImageU8 | None = None
        if source is FrameSource.BLENDER:
            meta = _read_meta()
            if meta is None or meta[0] == last_frame:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            last_frame, image_path, truth = meta
            try:
                encoded = image_path.read_bytes()
            except OSError:
                continue
            decoded = cv2.imdecode(frombuffer(encoded, dtype=uint8), cv2.IMREAD_COLOR)
            frame = decoded if decoded is not None else None
        elif cap is None:
            raise vis2.ImageReadError(Path(VIDEO_PATH))
        else:
            ok, frame = cap.read()
            if not ok:
                break
            last_frame += 1

        if frame is None:
            continue
        t0 = time.perf_counter()
        result = detector.infer(frame)
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        now = time.perf_counter()
        dt_s = 0.1 if last_tick is None else max(now - last_tick, 1e-3)
        last_tick = now
        publishes = tracker.step(observations_from_result(result, pose, homography, mesh), dt_s)
        if source is FrameSource.BLENDER:
            append_frame(IDENTITY_LOG, last_frame, truth, publishes)
        vis = overlay(frame, result, calib, minimap, truth, publishes, trails, fps)
        cv2.imshow("radar", vis2._scale(vis))
        print(f"[{last_frame}] cars {len(result.robots)} slots {len(publishes)}", flush=True)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
