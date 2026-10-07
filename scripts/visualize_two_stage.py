# ─── How to run ───
# 用已经装好 CUDA PyTorch 的解释器直接跑，不要 `uv run --script`。
# 本文件曾经写过 PEP 723：uv 会另建隔离环境，只声明 ultralytics/opencv/numpy，
# PyPI 上 ultralytics 依赖的 torch 默认是 CPU 轮子，把本机 cu130 盖掉。
#
#     python scripts/visualize_two_stage.py
# ──────────────────

"""港科大粗到细可视化：车 YOLO → 装甲 YOLO → MobileNetV3-Small 图案。

检测全部走 detect.TwoStageDetector（含图案）。本文件只负责读视频、画框、TensorRT 包装。
.engine 优先：本机 cuDNN 和 TensorRT 同卡容易抢上下文，车/装甲都用 engine 时不要再加载 .pt。
图案 .pth 会强制 CPU，避免和 YOLO engine 抢 cuDNN。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import cv2
import torch
from numpy import uint8
from numpy.typing import NDArray
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from detect import (  # noqa: E402
    FrameResult,
    ImageReadError,
    ModelFileNotFoundError,
    PatternStage,
    TwoStageConfig,
    TwoStageDetector,
    default_config,
    first_existing,
    warmup,
    with_device,
)
from detect.types import Detection  # noqa: E402
from models.pattern.model import MobileNetV3SmallClassifier  # noqa: E402
from models.pattern.transforms import val_transform  # noqa: E402

# ========== 填这里 ==========
CAR_PATH = "C:/Users/YQS/Desktop/DEUS_VISION_RADER_TEST_reproduce/weights/car_best.engine"  # 空则按 car_best.engine / car_best.pt / car_last.pt 自动找
ARMOR_PATH = "C:/Users/YQS/Desktop/DEUS_VISION_RADER_TEST_reproduce/weights/armor_best.engine"  # 空则按 armor_best.engine / armor_best.pt 自动找
PATTERN_PATH = "C:/Users/YQS/Desktop/DEUS_VISION_RADER_TEST_reproduce/weights/pattern_best.engine"
SOURCE_PATH = str(_ROOT / "scripts" / "RM_TestVideo.mp4")
DEVICE = "0"
SAVE_VIDEO = True
SHOW_WINDOW = True
MAX_SHOW_WIDTH = 1600
MAX_FRAMES = 0  # 0 = 整段视频
PATTERN_IMGSZ = 64
# ===========================

ImageU8 = NDArray[uint8]
PatternTransform = Callable[[Image.Image], torch.Tensor]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".ts"}
class PatternNet(Protocol):
    def __call__(self, batch: torch.Tensor) -> torch.Tensor: ...


class TrtPatternNet:
    """MobileNetV3 TensorRT。engine 静态 batch 由导出脚本决定，不足则零填充一次 enqueue。"""

    def __init__(self, engine_path: Path) -> None:
        import tensorrt as trt

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self._engine = engine
        self._context = engine.create_execution_context()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self._input = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        self._output = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        in_shape = tuple(int(dim) for dim in engine.get_tensor_shape(self._input))
        out_shape = tuple(int(dim) for dim in engine.get_tensor_shape(self._output))
        self._batch = max(1, int(in_shape[0]))
        self._stream = torch.cuda.Stream()
        self._inp = torch.empty(in_shape, device="cuda", dtype=torch.float32)
        self._out = torch.empty(out_shape, device="cuda", dtype=torch.float32)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.contiguous().to(device="cuda", dtype=torch.float32)
        outputs: list[torch.Tensor] = []
        stream = self._stream
        for start in range(0, batch.shape[0], self._batch):
            chunk = batch[start : start + self._batch]
            self._inp.zero_()
            self._inp[: chunk.shape[0]].copy_(chunk)
            self._context.set_tensor_address(self._input, int(self._inp.data_ptr()))
            self._context.set_tensor_address(self._output, int(self._out.data_ptr()))
            if not self._context.execute_async_v3(int(stream.cuda_stream)):
                raise RuntimeError("pattern engine execute failed")
            stream.synchronize()
            outputs.append(self._out[: chunk.shape[0]].clone())
        return torch.cat(outputs, dim=0)


def _torch_device() -> torch.device:
    if DEVICE == "cpu":
        return torch.device("cpu")
    if DEVICE.isdigit():
        return torch.device(f"cuda:{DEVICE}")
    return torch.device(DEVICE)


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
    width = frame.shape[1]
    if width <= MAX_SHOW_WIDTH:
        return frame
    scale = MAX_SHOW_WIDTH / width
    return cv2.resize(frame, (MAX_SHOW_WIDTH, int(frame.shape[0] * scale)))


def _resolve_car(model_dir: Path) -> Path:
    if CAR_PATH:
        return first_existing((Path(CAR_PATH),))
    trained = (
        model_dir / "car_best.engine",
        model_dir / "car_best.pt",
        model_dir / "car_last.pt",
    )
    try:
        return first_existing(trained)
    except ModelFileNotFoundError:
        fallback = first_existing((model_dir / "yolo26s.pt",))
        print(f"no trained car detector; fallback {fallback} (COCO, not robot)")
        return fallback


def _load_detector(pattern: PatternStage | None) -> TwoStageDetector:
    model_dir = _ROOT / "weights"
    car = _resolve_car(model_dir)
    armor = (
        first_existing((Path(ARMOR_PATH),))
        if ARMOR_PATH
        else first_existing((model_dir / "armor_best.engine", model_dir / "armor_best.pt"))
    )
    print(f"car:   {car}")
    print(f"armor: {armor}")
    cfg = default_config(str(car), str(armor), DEVICE)
    if car.suffix.lower() == ".pt" and armor.suffix.lower() == ".engine":
        cfg = TwoStageConfig(
            car_weights=cfg.car_weights,
            armor_weights=cfg.armor_weights,
            car=with_device(cfg.car, "cpu"),
            armor=cfg.armor,
        )
        print("car device=cpu (pt), armor device=", cfg.armor.device)
    return TwoStageDetector.from_config(cfg, pattern=pattern)


def _load_classifier() -> tuple[PatternNet, PatternTransform, torch.device]:
    if PATTERN_PATH:
        path = first_existing((Path(PATTERN_PATH),))
    else:
        path = first_existing(
            (
                _ROOT / "weights" / "pattern_best.engine",
                _ROOT / "weights" / "pattern_best.pth",
                _ROOT / "runs" / "pattern" / "mobilenetv3-small" / "models" / "best_mobilenetv3_small.pth",
            )
        )
    if path.suffix.lower() == ".engine":
        device = _torch_device()
        print(f"pattern: {path}  device={device}")
        return TrtPatternNet(path), val_transform(PATTERN_IMGSZ), device
    device = torch.device("cpu")
    print(f"pattern: {path}  device={device} (pth fallback, CPU to avoid cuDNN vs TRT)")
    net = MobileNetV3SmallClassifier(pretrained=False)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device)
    net.eval()
    return net, val_transform(PATTERN_IMGSZ), device


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
    net, transform, device = _load_classifier()
    detector = _load_detector(PatternStage(net=net, transform=transform, device=str(device)))
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
