# ─── How to run ───
# 用已经装好 CUDA PyTorch 的解释器直接跑，不要 `uv run --script`。
# 本文件曾经写过 PEP 723：uv 会另建隔离环境，只声明 ultralytics/opencv/numpy，
# PyPI 上 ultralytics 依赖的 torch 默认是 CPU 轮子，把本机 cu130 盖掉。
#
#     python scripts/visualize_two_stage.py
# ──────────────────

"""三级检测可视化与演示入口：读视频/图像 → 检测 → 画框。

权重查找回退、engine 自动编译、TrtPatternNet 等加载逻辑在 detect/runtime.py；
run_radar 主循环的绘制（draw_result / _scale）也来自本文件。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import torch
from numpy import uint8
from numpy.typing import NDArray

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from detect import (  # noqa: E402
    FrameResult,
    ImageReadError,
    PatternStage,
    TwoStageDetector,
    warmup,
)
from detect.runtime import (  # noqa: E402
    PATTERN_IMGSZ,
    PatternNet,
    load_classifier,
    load_detector,
)
from detect.types import Detection  # noqa: E402

# ========== 填这里（演示入口参数；三级权重路径在 detect/runtime.py 顶部） ==========
SOURCE_PATH = str(_ROOT / "scripts" / "RM_TestVideo.mp4")
SAVE_VIDEO = True
SHOW_WINDOW = True
MAX_SHOW_WIDTH = 1600
MAX_SHOW_HEIGHT = 860
MAX_FRAMES = 0  # 0 = 整段视频
# ===========================

ImageU8 = NDArray[uint8]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".ts"}


def _color(label: str) -> tuple[int, int, int]:
    match label:
        case "robot":
            return (0, 255, 0)
        case "dead":
            return (180, 180, 180)
        case "red":
            return (0, 0, 255)
        case "blue":
            return (255, 0, 0)
        case _:
            return (0, 255, 255)


def _pattern_tag(color_label: str, pattern: str) -> str:
    match color_label:
        case "red":
            return f"R{pattern}"
        case "blue":
            return f"B{pattern}"
        case "dead":
            return f"G{pattern}"
        case _:
            return pattern


def draw_detection(frame: ImageU8, det: Detection, thickness: int, caption: str | None = None) -> None:
    color = _color(det.label)
    x1, y1, x2, y2 = det.box.x1, det.box.y1, det.box.x2, det.box.y2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if caption is None:
        prefix = f"#{det.track_id} " if det.track_id is not None else ""
        caption = f"{prefix}{det.label} {det.conf:.2f}"
    cv2.putText(frame, caption, (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_result(frame: ImageU8, result: FrameResult, fps: float) -> ImageU8:
    canvas = frame.copy()
    for robot in result.robots:
        draw_detection(canvas, robot.car, 2)
        for armor in robot.armors:
            caption = None
            if armor.pattern is not None:
                caption = f"{_pattern_tag(armor.label, armor.pattern.name)} {armor.pattern.conf:.2f}"
            draw_detection(canvas, armor, 2, caption)
    armor_n = sum(len(robot.armors) for robot in result.robots)
    cv2.putText(
        canvas,
        f"E2E {fps:.1f}  cars {len(result.robots)}  armor {armor_n}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return canvas


def _scale(frame: ImageU8) -> ImageU8:
    height, width = frame.shape[:2]
    scale = min(MAX_SHOW_WIDTH / width, MAX_SHOW_HEIGHT / height, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(width * scale), int(height * scale)))


def run_image(
    detector: TwoStageDetector,
    net: PatternNet,
    device: torch.device,
    source: Path,
    out_dir: Path,
) -> Path:
    frame = cv2.imread(str(source))
    if frame is None:
        raise ImageReadError(source)
    warmup(detector, frame)
    with torch.no_grad():
        net(torch.zeros(1, 3, PATTERN_IMGSZ, PATTERN_IMGSZ, device=device))
    t0 = time.perf_counter()
    result = detector.infer(frame)
    elapsed = time.perf_counter() - t0
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    vis = draw_result(frame, result, fps)
    out = out_dir / f"{source.stem}_three_stage.jpg"
    cv2.imwrite(str(out), vis)
    if SHOW_WINDOW:
        cv2.imshow("three-stage", _scale(vis))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return out


def run_video(
    detector: TwoStageDetector,
    net: PatternNet,
    device: torch.device,
    source: Path,
    out_dir: Path,
) -> Path | None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ImageReadError(source)
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise ImageReadError(source)
    warmup(detector, first)
    with torch.no_grad():
        net(torch.zeros(1, 3, PATTERN_IMGSZ, PATTERN_IMGSZ, device=device))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer: cv2.VideoWriter | None = None
    out_path: Path | None = None
    frame_id = 0
    t_all = time.perf_counter()
    e2e_fps = 0.0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        result = detector.infer(frame)
        vis = draw_result(frame, result, e2e_fps)
        shown = _scale(vis)
        if SAVE_VIDEO:
            if writer is None:
                out_path = out_dir / f"{source.stem}_three_stage.mp4"
                h, w = shown.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
            writer.write(shown)
        if SHOW_WINDOW:
            cv2.imshow("three-stage", shown)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_id += 1
        elapsed = time.perf_counter() - t0
        e2e_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        armor_n = sum(len(robot.armors) for robot in result.robots)
        ids = ",".join(str(robot.car.track_id) for robot in result.robots)
        tags = ",".join(
            _pattern_tag(armor.label, armor.pattern.name)
            for robot in result.robots
            for armor in robot.armors
            if armor.pattern is not None
        )
        print(f"[{frame_id}] FPS {e2e_fps:.1f} cars {len(result.robots)} [{ids}] armor {armor_n} [{tags}]")
        if MAX_FRAMES > 0 and frame_id >= MAX_FRAMES:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if frame_id:
        print(f"\nframes {frame_id}  wall FPS {frame_id / (time.perf_counter() - t_all):.2f}")
    return out_path


def main() -> None:
    source = Path(SOURCE_PATH).expanduser()
    if not source.is_file():
        raise ImageReadError(source)
    net, transform, device = load_classifier()
    detector = load_detector(PatternStage(net=net, transform=transform, device=str(device)))
    ext = source.suffix.lower()
    if ext in IMAGE_EXTS:
        out = run_image(detector, net, device, source, source.parent)
        print(f"saved {out}")
        return
    if ext in VIDEO_EXTS:
        out = run_video(detector, net, device, source, source.parent)
        if out is not None:
            print(f"saved {out}")
        return
    raise ImageReadError(source)


if __name__ == "__main__":
    main()
