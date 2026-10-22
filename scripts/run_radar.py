# ─── How to run ───
# 终端 1（仿真出图）：
#     & "D:\Program Files\blender\blender.exe" scene/RMUC2027_V1.0.0_radar_fixed.blend --python scene/armor_assets/radar_sim_loop.py -- --motion patrol
# 终端 2（检测 + 小地图）：
#     python scripts/run_radar.py
# SOURCE=video 时不需要 Blender。q 退出。
# ──────────────────

"""主循环：图像源 → 三级检测 → 平面射线 → 相机画面内嵌小地图。"""

from __future__ import annotations

import json
import os
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
from detect import FrameResult, ImageReadError, PatternStage, warmup  # noqa: E402
from detect.runtime import load_classifier, load_detector  # noqa: E402
from locate.camera import camera_from_path  # noqa: E402
from locate.ray_mesh import FieldMesh  # noqa: E402
from locate.errors import SourceNotReadyError  # noqa: E402
from locate.homography import HomographyMap, homography_from_path  # noqa: E402
from locate.minimap import minimap_from_path  # noqa: E402

from locate.types import FieldXY  # noqa: E402
from referee.commands import MarkProgress, RadarInfo, marks_from_publishes  # noqa: E402
from referee.link import RefereeLink  # noqa: E402
from referee.policy import should_trigger_double  # noqa: E402
from referee.stub import StubReferee  # noqa: E402
from referee.transport import LoopbackPair  # noqa: E402
from track.cascade import CascadeMatchTracker  # noqa: E402
from track.guess import apply_guess_to_lost  # noqa: E402
from track.errors import TruthFieldError, UnmappedRobotName  # noqa: E402
from track.log import append_frame  # noqa: E402
from track.observe import observations_from_result  # noqa: E402
from track.truth import parse_sim_robots  # noqa: E402
from track.types import PublishedTrack, SimRobot, Team, TrackPhase  # noqa: E402

# ========== 开关 ==========
# RADAR_SOURCE / RADAR_IMAGES_DIR 环境变量可覆盖（打分跑批用，不用改代码）。
SOURCE = os.environ.get("RADAR_SOURCE", "blender")  # blender | video | images | hik
LOCATE_MODE = "ray"  # ray | homography
VIDEO_PATH = str(_ROOT / "scripts" / "RM_TestVideo.mp4")
# images 源：generate_radar_car_sim.py --motion patrol --dt 0.1 渲染的序列，
# 每帧同名 .json 是真值（frame/camera/dt/robots），10Hz 固定步长回放。
IMAGES_DIR = Path(os.environ.get("RADAR_IMAGES_DIR", str(_ROOT / "scene" / "locate_out" / "videoseq")))
# 打分跑批设 RADAR_HEADLESS=1，不弹窗不抢焦点。
HEADLESS = os.environ.get("RADAR_HEADLESS", "0") == "1"
CAMERA_BLUE_JSON = _ROOT / "locate" / "calib" / "Camera_Radar_Blue.json"
CAMERA_RED_JSON = _ROOT / "locate" / "calib" / "Camera_Radar_Red.json"
FIELD_MESH = _ROOT / "locate" / "calib" / "field_mesh.npz"
HOMOGRAPHY_JSON = _ROOT / "locate" / "calib" / "match_homography.json"
MINIMAP_JSON = _ROOT / "locate" / "calib" / "minimap.json"
MINIMAP_IMAGE = _ROOT / "scene" / "materials" / "rmuc_fig4_1_topdown.png"
LIVE_PNG = _ROOT / "scene" / "locate_out" / "live_frame.png"
LIVE_META = _ROOT / "scene" / "locate_out" / "live_meta.json"
SHOW_GT = False
# 尾线只留最新 1 个点 = 不画线。发布帧稀疏（几秒一帧、跳好几米），连线全是误导。
# 注意不能填 0：del trail[:-0] 删不掉东西，列表会无限涨还照画不误。
TRAIL_LEN = 1
IDENTITY_LOG = Path(os.environ.get("RADAR_IDENTITY_LOG", str(_ROOT / "scene" / "locate_out" / "identity_log.jsonl")))
# 裁判 stub（打分端）：RADAR_REFEREE_STUB=1 后，每帧把发布坐标打包成 0x0305
# 发进内存回环，stub 裁判按规则 5.6.6 算 P，逐帧写 REFEREE_LOG。
REFEREE_STUB = os.environ.get("RADAR_REFEREE_STUB", "0") == "1"
REFEREE_LOG = Path(os.environ.get("RADAR_REFEREE_LOG", str(_ROOT / "scene" / "locate_out" / "referee_log.jsonl")))
# 质量门 A/B：RADAR_SEND_LOST=1 时 LOST（卡尔曼外推）槽也发 0x0305。
SEND_LOST = os.environ.get("RADAR_SEND_LOST", "0") == "1"
# RADAR_GUESS=1：LOST 槽坐标换成战略猜点（配合 SEND_LOST 才会真的发出去）。
GUESS_LOST = os.environ.get("RADAR_GUESS", "0") == "1"
# 海康/真相机源：RADAR_HIK_URL（RTSP 地址或设备号）。
# USB 直连（MV-CH120 系列）：RADAR_HIK_URL=0
# RTSP（网络相机）：       RADAR_HIK_URL=rtsp://user:pass@192.168.1.64:554/...
HIK_URL = os.environ.get("RADAR_HIK_URL", "0")  # 默认 USB 第 0 路
# stub 裁判阶段压缩：1.0=真实时长（准备3min+自检15s+倒计时5s+战斗7min）。
# 0.05 → 准备 9s+自检 0.75s+倒计时 0.25s+战斗 21s——回放 20s 序列能测到阶段切换。
PHASE_SCALE = float(os.environ.get("RADAR_PHASE_SCALE", "0.05"))
# RADAR_SENTRY_GUIDE=1：通过 0x0301 机器人间通信把敌方位置发给己方哨兵。
SENTRY_GUIDE = os.environ.get("RADAR_SENTRY_GUIDE", "0") == "1"
# =========================

ImageU8 = NDArray[uint8]


class FrameSource(StrEnum):
    BLENDER = "blender"
    VIDEO = "video"
    IMAGES = "images"
    HIK = "hik"


def _preflight_gpu() -> None:
    """启动自检（借鉴 T-DT）：CUDA 不可用时 .engine 加载必然失败，提前给可读错误。

    engine 是 GPU 绑定的（换卡/进 Docker 要重导），把"跑到一半抛 TRT 反序列化
    错误"提前成一句人话；纯 .pt 配置只警告不拦截（CPU 能跑，慢）。
    """
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except Exception:  # noqa: BLE001 — torch 装坏也该在这里报，而不是后面莫名崩
        cuda_ok = False
    if cuda_ok:
        return
    weights_dir = _ROOT / "weights"
    engines = [
        item
        for item in (
            weights_dir / "car_best.engine",
            weights_dir / "armor_best.engine",
            weights_dir / "pattern_best.engine",
        )
        if item.is_file()
    ]
    if engines:
        names = ", ".join(item.name for item in engines)
        raise SystemExit(
            f"preflight FAIL: CUDA 不可用，但存在 TensorRT engine（{names}）。"
            "engine 是 GPU 绑定的，加载必然失败——先 nvidia-smi 查驱动；"
            "急用可把 .engine 改名移走，回退 .pt 跑（慢）。"
        )
    print("preflight WARN: CUDA 不可用——engine 缺失，将回退 .pt CPU 推理，速度显著下降", flush=True)


def _read_meta() -> tuple[int, Path, tuple[SimRobot, ...], str] | None:
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
    camera_name = str(raw.get("camera", "Camera_Radar_Blue"))
    try:
        robots = _robots(raw)
    except (TruthFieldError, UnmappedRobotName) as exc:
        print(exc, flush=True)
        robots = ()
    return frame_id, image, robots, camera_name


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


def _dot(
    board: ImageU8,
    u: float,
    v: float,
    color: tuple[int, int, int],
    tag: str,
    *,
    extrapolating: bool = False,
) -> None:
    """实心 = 这帧真的看见；空圈+? = LOST 外推/猜点，画面上没这辆车。"""
    pt = (int(round(u)), int(round(v)))
    if extrapolating:
        cv2.circle(board, pt, 11, color, 3)  # 纯空圈：背景透出，一眼不是实车
        cv2.putText(board, tag + "?", (pt[0] + 12, pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    else:
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
    result: FrameResult,
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
        _dot(
            board,
            mapped.u,
            mapped.v,
            color,
            item.label,
            extrapolating=item.phase is TrackPhase.LOST,
        )
    for tag in list(trails):
        if tag not in seen:
            trails.pop(tag, None)
    return inset_minimap(canvas, board)


def main() -> None:
    source = FrameSource(SOURCE)
    match source:
        case FrameSource.BLENDER | FrameSource.VIDEO | FrameSource.IMAGES | FrameSource.HIK:
            pass
        case unreachable:
            raise SourceNotReadyError(source=str(unreachable))

    _preflight_gpu()

    pose = None
    homography: HomographyMap | None = None
    mesh: FieldMesh | None = None
    poses = {}
    if LOCATE_MODE == "homography":
        homography = homography_from_path(HOMOGRAPHY_JSON)
    else:
        poses = {
            "Camera_Radar_Blue": camera_from_path(CAMERA_BLUE_JSON),
            "Camera_Radar_Red": camera_from_path(CAMERA_RED_JSON),
        }
        pose = poses["Camera_Radar_Blue"]
        if FIELD_MESH.is_file():
            mesh = FieldMesh.from_npz(FIELD_MESH)
            print("run_radar mesh", FIELD_MESH.name, flush=True)
    calib = minimap_from_path(MINIMAP_JSON)
    minimap = cv2.imread(str(MINIMAP_IMAGE))
    if minimap is None:
        raise ImageReadError(MINIMAP_IMAGE)

    cap: cv2.VideoCapture | None = None
    if source is FrameSource.VIDEO:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise ImageReadError(Path(VIDEO_PATH))
    if source is FrameSource.HIK:
        url = HIK_URL
        # 纯数字当设备号，其余当 RTSP URL
        cap = cv2.VideoCapture(int(url)) if url.isdigit() else cv2.VideoCapture(url)
        if not cap.isOpened():
            raise ImageReadError(Path(f"camera:{url}"))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"run_radar hik camera {url} → {w}x{h} @ {fps:.0f}fps", flush=True)
    img_files: list[Path] = []
    if source is FrameSource.IMAGES:
        img_files = sorted(IMAGES_DIR.glob("*.jpg"))
        if not img_files:
            raise ImageReadError(IMAGES_DIR)

    print("run_radar source", source.value, "loading detector…", flush=True)
    net, transform, device = load_classifier()
    detector = load_detector(PatternStage(net=net, transform=transform, device=str(device)))
    warmup(detector, zeros((3000, 4096, 3), dtype=uint8))
    last_frame = -1
    camera_name = "Camera_Radar_Blue"
    img_idx = 0
    trails: dict[str, list[tuple[int, int]]] = {}
    tracker = CascadeMatchTracker()
    last_tick: float | None = None
    t_sim = 0.0
    referee_link: RefereeLink | None = None
    referee_stub: StubReferee | None = None
    referee_own: Team | None = None
    if REFEREE_STUB:
        REFEREE_LOG.parent.mkdir(parents=True, exist_ok=True)
        REFEREE_LOG.write_text("", encoding="utf-8")
        print("run_radar referee stub on ->", REFEREE_LOG.name, flush=True)
    if source is FrameSource.BLENDER or source is FrameSource.IMAGES:
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
            last_frame, image_path, truth, camera_name = meta
            try:
                encoded = image_path.read_bytes()
            except OSError:
                continue
            decoded = cv2.imdecode(frombuffer(encoded, dtype=uint8), cv2.IMREAD_COLOR)
            frame = decoded if decoded is not None else None
        elif source is FrameSource.IMAGES:
            if img_idx >= len(img_files):
                break
            image_path = img_files[img_idx]
            img_idx += 1
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            last_frame += 1
            img_dt = 0.1
            meta_path = image_path.with_suffix(".json")
            if meta_path.is_file():
                try:
                    raw = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    raw = {}
                camera_name = str(raw.get("camera", "Camera_Radar_Blue"))
                try:
                    img_dt = float(raw.get("dt", 0.1))
                except (TypeError, ValueError):
                    img_dt = 0.1
                try:
                    truth = _robots(raw)
                except (TruthFieldError, UnmappedRobotName) as exc:
                    print(exc, flush=True)
                    truth = ()
        elif cap is None:
            raise ImageReadError(Path(VIDEO_PATH))
        else:
            ok, frame = cap.read()
            if not ok:
                if source is FrameSource.HIK:
                    # 断流重连：真相机偶尔掉线，等 1 秒重试，不退出
                    print("run_radar hik frame drop, reconnecting…", flush=True)
                    time.sleep(1.0)
                    cap.release()
                    cap = cv2.VideoCapture(int(HIK_URL)) if HIK_URL.isdigit() else cv2.VideoCapture(HIK_URL)
                    if cap.isOpened():
                        continue
                    print("run_radar hik reconnect failed, retrying…", flush=True)
                    time.sleep(2.0)
                    continue
                break  # VIDEO 文件读完正常退出
            last_frame += 1

        if frame is None:
            continue
        t0 = time.perf_counter()
        if poses:
            pose = poses.get(camera_name, poses["Camera_Radar_Blue"])
        # persist_tracks=False：级联跟踪器拥有跨帧关联。检测内 BoT-SORT 在
        # 视角轮换/幻灯片式跳帧下只会晃晕，还会吞框。
        result = detector.infer(frame, persist_tracks=False)
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        now = time.perf_counter()
        first_tick = last_tick is None
        dt_s = 0.1 if first_tick else max(now - last_tick, 1e-3)
        if source is FrameSource.IMAGES:
            dt_s = img_dt
        last_tick = now
        t_sim = 0.0 if first_tick else t_sim + dt_s
        publishes = tracker.step(
            observations_from_result(
                result, pose, homography, mesh,
                own_team=Team.BLUE if "Blue" in camera_name else Team.RED,
            ),
            dt_s,
        )
        if GUESS_LOST:
            publishes = apply_guess_to_lost(publishes)
        if REFEREE_STUB:
            if referee_link is None or referee_stub is None or referee_own is None:
                referee_own = Team.BLUE if "Blue" in camera_name else Team.RED
                pair = LoopbackPair()
                referee_link = RefereeLink(pair.radar, own_team=referee_own)
                referee_stub = StubReferee(pair.peer, own_team=referee_own)
                referee_stub.set_phase_scale(PHASE_SCALE)
                print("run_radar referee own_team", referee_own.value, "phase_scale", PHASE_SCALE, flush=True)
            referee_stub.set_truth(truth)
            _marks = marks_from_publishes(referee_own, publishes, include_lost=SEND_LOST)
            referee_link.send_map(_marks, now_s=t_sim)
            if SENTRY_GUIDE:
                referee_link.send_sentry_guidance(_marks)
            referee_stub.step(now_s=t_sim)
            radar_info_state: RadarInfo | None = None
            mark_state: MarkProgress | None = None
            for event in referee_link.poll(timeout_s=0.0):
                if event.radar_info is not None:
                    radar_info_state = event.radar_info
                if event.mark_progress is not None:
                    mark_state = event.mark_progress
            if radar_info_state is not None and mark_state is not None:
                if should_trigger_double(radar_info_state, mark_state):
                    referee_link.trigger_double()
                    print(f"[{last_frame}] double triggered, chances left {radar_info_state.double_chances - 1}", flush=True)
            referee_stub.log_frame(REFEREE_LOG, last_frame, now_s=t_sim)
        if source is FrameSource.BLENDER or source is FrameSource.IMAGES:
            append_frame(IDENTITY_LOG, last_frame, truth, publishes)
        vis = overlay(frame, result, calib, minimap, truth, publishes, trails, fps)
        print(f"[{last_frame}] cars {len(result.robots)} slots {len(publishes)}", flush=True)
        if HEADLESS:
            continue
        cv2.imshow("radar", vis2._scale(vis))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap is not None:
        cap.release()
    if referee_stub is not None:
        summary = referee_stub.summary()
        print(
            "referee frames", summary.frames,
            "union_vuln_s", round(summary.union_vuln_s, 2),
            "double_chances", summary.double_chances_earned,
            flush=True,
        )
        print("score: python scripts/score_referee.py", REFEREE_LOG, flush=True)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
