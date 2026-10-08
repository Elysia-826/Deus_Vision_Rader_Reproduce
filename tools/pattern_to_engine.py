"""把 EfficientNet-B0 分类权重导出为 TensorRT .engine。填好路径后直接运行。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PATH"] = os.pathsep.join(
    part for part in os.environ.get("PATH", "").split(os.pathsep) if "CUDNN" not in part.upper()
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from models.pattern.model import EfficientNetB0Classifier

# ========== 填这里 ==========
PT_PATH = str(_ROOT / "weights" / "pattern_efficientnet_b0.pth")
ENGINE_PATH = str(_ROOT / "weights" / "pattern_efficientnet_b0.engine")
ONNX_PATH = str(_ROOT / "weights" / "pattern_efficientnet_b0.onnx")
IMGSZ = 96
BATCH = 8
OPSET = 18
# ===========================


def _load_net(pt_path: Path) -> EfficientNetB0Classifier:
    net = EfficientNetB0Classifier(pretrained=False)
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


def _export_onnx(net: EfficientNetB0Classifier, onnx_path: Path) -> None:
    dummy = torch.zeros(BATCH, 3, IMGSZ, IMGSZ)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        str(onnx_path),
        input_names=["images"],
        output_names=["logits"],
        opset_version=OPSET,
        dynamo=False,
    )


def _build_engine(onnx_path: Path, engine_path: Path) -> None:
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
    print(f"TensorRT {trt.__version__} building engine from {onnx_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


def main() -> None:
    pt_path = Path(PT_PATH).expanduser().resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f"pt 不存在: {pt_path}")
    onnx_path = Path(ONNX_PATH).expanduser().resolve()
    engine_path = Path(ENGINE_PATH).expanduser().resolve()
    print(f"load {pt_path}")
    net = _load_net(pt_path)
    print(f"onnx  {onnx_path}")
    _export_onnx(net, onnx_path)
    print(f"engine {engine_path}")
    _build_engine(onnx_path, engine_path)
    print(f"已保存: {engine_path}")


if __name__ == "__main__":
    main()
