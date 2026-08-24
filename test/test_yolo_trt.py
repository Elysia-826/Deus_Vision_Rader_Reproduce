"""TensorRT YOLO 推理测试。填好 MODEL_PATH / SOURCE_PATH 后直接运行。"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# ========== 填这里 ==========
MODEL_PATH = r"C:\Users\YQS\Desktop\DEUS_VISION_RADER_TEST_reproduce\model\armor_best.engine"
SOURCE_PATH = r"C:\Users\YQS\Desktop\DEUS_VISION_RADER_TEST_reproduce\test\RM_TestVideo.mp4"
IMGSZ = 192  # 装甲板 192；这份 engine 已按 192 编好
CONF = 0.25
DEVICE = 0
# ===========================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".ts"}


def load_trt_model(model_path: Path) -> YOLO:
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    suffix = model_path.suffix.lower()
    if suffix == ".engine":
        return YOLO(str(model_path))

    if suffix in {".pt", ".onnx"}:
        engine_path = model_path.with_suffix(".engine")
        if not engine_path.is_file():
            print(f"导出 TensorRT engine -> {engine_path}")
            exported = YOLO(str(model_path)).export(
                format="engine",
                imgsz=IMGSZ,
                device=DEVICE,
                half=True,
            )
            engine_path = Path(str(exported))
        return YOLO(str(engine_path))

    raise ValueError(f"不支持的模型格式: {suffix}，请用 .engine / .pt / .onnx")


def collect_boxes(result) -> list[tuple[str, float, int, int, int, int]]:
    boxes_out: list[tuple[str, float, int, int, int, int]] = []
    if result.boxes is None:
        return boxes_out

    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        if isinstance(names, dict):
            label = str(names.get(cls_id, cls_id))
        else:
            label = str(names[cls_id])
        boxes_out.append((label, conf, x1, y1, x2, y2))
    return boxes_out


def draw_boxes(
    frame,
    boxes: list[tuple[str, float, int, int, int, int]],
) -> None:
    for label, conf, x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def print_boxes(
    boxes: list[tuple[str, float, int, int, int, int]],
    fps: float,
    frame_id: int | None = None,
) -> None:
    prefix = f"[{frame_id}] " if frame_id is not None else ""
    print(f"{prefix}FPS: {fps:.2f}  dets: {len(boxes)}")
    for label, conf, x1, y1, x2, y2 in boxes:
        print(f"  {label} {conf:.3f}  [{x1}, {y1}, {x2}, {y2}]")


def infer_frame(model: YOLO, frame):
    t0 = time.perf_counter()
    result = model.predict(frame, verbose=False, device=DEVICE, conf=CONF)[0]
    dt = time.perf_counter() - t0
    fps = 1.0 / dt if dt > 0 else 0.0
    return result, fps, dt


def draw_e2e_fps(frame, e2e_fps: float) -> None:
    cv2.putText(
        frame,
        f"E2E FPS: {e2e_fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )


def run_image(model: YOLO, source: Path) -> None:
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"无法读取图片: {source}")

    model.predict(frame, verbose=False, device=DEVICE)
    t0 = time.perf_counter()
    result, _, _ = infer_frame(model, frame)
    boxes = collect_boxes(result)
    draw_boxes(frame, boxes)
    elapsed = time.perf_counter() - t0
    e2e_fps = 1.0 / elapsed if elapsed > 0 else 0.0
    print_boxes(boxes, e2e_fps)
    draw_e2e_fps(frame, e2e_fps)
    out = source.with_name(f"{source.stem}_trt.jpg")
    cv2.imwrite(str(out), frame)
    print(f"已保存: {out}")
    cv2.imshow("YOLO TensorRT", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(model: YOLO, source: Path) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {source}")

    ok, warmup = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("视频为空")
    model.predict(warmup, verbose=False, device=DEVICE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_id = 0
    infer_sum = 0.0
    e2e_fps = 0.0
    t_all = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        result, _, dt = infer_frame(model, frame)
        infer_sum += dt
        frame_id += 1
        boxes = collect_boxes(result)
        print_boxes(boxes, e2e_fps, frame_id)
        draw_boxes(frame, boxes)
        draw_e2e_fps(frame, e2e_fps)
        cv2.imshow("YOLO TensorRT", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elapsed = time.perf_counter() - t0
        e2e_fps = 1.0 / elapsed if elapsed > 0 else 0.0

    cap.release()
    cv2.destroyAllWindows()
    if frame_id:
        wall_fps = frame_id / (time.perf_counter() - t_all)
        infer_fps = frame_id / infer_sum if infer_sum > 0 else 0.0
        print(
            f"平均端到端 FPS: {wall_fps:.2f}  平均推理 FPS: {infer_fps:.2f}  总帧: {frame_id}"
        )


def main() -> None:
    if not MODEL_PATH or not SOURCE_PATH:
        raise SystemExit("请先填写 MODEL_PATH 和 SOURCE_PATH")

    model_path = Path(MODEL_PATH).expanduser()
    source = Path(SOURCE_PATH).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"测试文件不存在: {source}")

    model = load_trt_model(model_path)
    ext = source.suffix.lower()
    if ext in IMAGE_EXTS:
        run_image(model, source)
    elif ext in VIDEO_EXTS:
        run_video(model, source)
    else:
        raise ValueError(f"不支持的媒体格式: {ext}")


if __name__ == "__main__":
    main()
