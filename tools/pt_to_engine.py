"""把 Ultralytics YOLO .pt 导出为 TensorRT .engine。填好 PT_PATH 后直接运行。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["PATH"] = os.pathsep.join(
    part for part in os.environ.get("PATH", "").split(os.pathsep) if "CUDNN" not in part.upper()
)

import ultralytics
from ultralytics import YOLO

# ========== 填这里 ==========
PT_PATH = r"C:\Users\YQS\Desktop\DEUS_VISION_RADER_TEST_reproduce\weights\car_best.pt"
ENGINE_PATH = r"C:\Users\YQS\Desktop\DEUS_VISION_RADER_TEST_reproduce\weights\car_best.engine"
IMGSZ = 1280  # 装甲板 192，车辆 1280
BATCH = 1  # 车=1；装甲导出改 8，多 ROI 一次 enqueue
DEVICE = 0
HALF = True  # TensorRT FP16（ONNX 先出 half，再交给 TRT 11 strongly-typed）
WORKSPACE = None  # GiB；None 用 TensorRT 默认
# ===========================


def _names_table(raw: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    return {index: str(name) for index, name in enumerate(raw)}


def _metadata(model: YOLO) -> dict[str, object]:
    names = _names_table(model.names)
    if names == {0: "item"}:
        names = {0: "robot"}
    stride = getattr(model.model, "stride", None)
    stride_i = int(max(stride)) if stride is not None else 32
    return {
        "description": "Ultralytics YOLO TensorRT engine",
        "author": "Ultralytics",
        "license": "AGPL-3.0",
        "date": datetime.now(timezone.utc).isoformat(),
        "version": ultralytics.__version__,
        "stride": stride_i,
        "task": "detect",
        "batch": BATCH,
        "imgsz": [IMGSZ, IMGSZ],
        "names": names,
        "args": {"imgsz": IMGSZ, "half": HALF, "batch": BATCH},
        "channels": 3,
    }


def _export_onnx(model: YOLO) -> Path:
    exported = model.export(
        format="onnx",
        imgsz=IMGSZ,
        device=DEVICE,
        half=HALF,
        simplify=True,
        opset=18,
        dynamic=False,
        batch=BATCH,
    )
    return Path(str(exported)).resolve()


def _build_engine(onnx_path: Path, engine_path: Path, metadata: dict[str, object]) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    if WORKSPACE is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(WORKSPACE * (1 << 30)))

    print(f"TensorRT {trt.__version__} building engine from {onnx_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(metadata)
    with engine_path.open("wb") as fh:
        fh.write(len(meta).to_bytes(4, byteorder="little", signed=True))
        fh.write(meta.encode())
        fh.write(bytes(serialized))


def main() -> None:
    pt_path = Path(PT_PATH).expanduser().resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f"pt 不存在: {pt_path}")
    if pt_path.suffix.lower() != ".pt":
        raise ValueError(f"输入必须是 .pt: {pt_path}")

    out_path = Path(ENGINE_PATH).expanduser() if ENGINE_PATH else pt_path.with_suffix(".engine")
    print(f"导出 TensorRT engine: {pt_path} -> {out_path}")

    model = YOLO(str(pt_path))
    onnx_path = _export_onnx(model)
    _build_engine(onnx_path, out_path.resolve(), _metadata(model))
    print(f"已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
