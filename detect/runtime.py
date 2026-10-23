"""运行时加载器：三级检测链的权重查找回退、engine 自动编译、TensorRT 图案包装。

从 scripts/visualize_two_stage.py 上提——detect/ 是推理侧库（见 docs/architecture.md §2
分层），加载器住这里，scripts/ 只留绘制与入口。消费方：run_radar / diagnose /
score_stage_funnel / visualize_* / dataset 三级收割脚本。

引擎优先：本机 cuDNN 与 TensorRT 同卡易抢上下文，车/装甲都走 engine 时
不要再加载 .pt；图案 .pth 强制 CPU，避免和 YOLO engine 抢 cuDNN。
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image

from detect import (
    ModelFileNotFoundError,
    PatternStage,
    TwoStageConfig,
    TwoStageDetector,
    default_config,
    first_existing,
    with_device,
)
from models.pattern.model import MobileNetV3SmallClassifier
from models.pattern.transforms import val_transform

_ROOT = Path(__file__).resolve().parents[1]

# ========== 填这里（三级权重路径）==========
# 留空 = weights/ 默认三件套：engine 缺失时自动编译（RADAR_AUTO_ENGINE=0 关），
# 再缺失回退同名 .pt/.pth。填路径 = 只用该文件（调试指向别处时用）。
CAR_PATH = ""  # 空 = weights/car_best.engine → car_best.pt
ARMOR_PATH = ""  # 空 = weights/armor_best.engine → armor_best.pt
PATTERN_PATH = ""  # 空 = weights/pattern_best.engine → pattern_best.pth
DEVICE = "0"
PATTERN_IMGSZ = 64
# ===========================================

# engine 缺失自动编译（借鉴 T-DT 的启动时 onnx→engine 兜底）。默认开，
# RADAR_AUTO_ENGINE=0 关闭。只对 weights/ 下的默认三件套生效；
# 编译在子进程做（cuDNN 与 TRT 上下文隔离），失败静默回退 .pt/.pth 加载。
AUTO_ENGINE = os.environ.get("RADAR_AUTO_ENGINE", "1") != "0"


def _maybe_build_engine(tool: Path, env: dict[str, str], engine_path: Path) -> bool:
    """engine 缺失且 CUDA 在场 → 子进程调 tools 导出脚本编译；失败回退 .pt/.pth。"""
    if engine_path.is_file() or not AUTO_ENGINE:
        return False
    if not torch.cuda.is_available():
        print(f"auto-engine skip: CUDA 不可用，不编译 {engine_path.name}，回退 .pt/.pth")
        return False
    print(f"engine 缺失: {engine_path.name} -> 自动编译（{tool.name}，FP16，需数分钟）…")
    try:
        proc = subprocess.run(
            [sys.executable, str(tool)],
            env={**os.environ, **env},
            cwd=str(_ROOT),
        )
    except OSError as exc:
        print(f"auto-engine 失败，回退 .pt/.pth：{exc}")
        return False
    if proc.returncode == 0 and engine_path.is_file():
        print(f"auto-engine 完成: {engine_path}")
        return True
    print(f"auto-engine 失败（exit {proc.returncode}），回退 .pt/.pth 加载")
    return False


PatternTransform = Callable[[Image.Image], torch.Tensor]


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
        self._inp = torch.empty(in_shape, device="cuda", dtype=torch.float32)
        self._out = torch.empty(out_shape, device="cuda", dtype=torch.float32)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.contiguous().to(device="cuda", dtype=torch.float32)
        outputs: list[torch.Tensor] = []
        # 全部走当前（默认）流：zero_/copy_/execute/读回天然全序。
        # 之前 execute 提交到自建流、拷贝在默认流，两个流无事件依赖，
        # GPU 乱序执行会偶发读到旧输入（同图推理结果随机翻转）。
        stream = torch.cuda.current_stream()
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


def _resolve_car(model_dir: Path) -> Path:
    if CAR_PATH:
        return first_existing((Path(CAR_PATH),))
    _maybe_build_engine(
        _ROOT / "tools" / "pt_to_engine.py",
        {
            "YOLO_PT": str(model_dir / "car_best.pt"),
            "YOLO_ENGINE": str(model_dir / "car_best.engine"),
            "YOLO_IMGSZ": "1280",
            "YOLO_BATCH": "1",
        },
        model_dir / "car_best.engine",
    )
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


def load_detector(pattern: PatternStage | None) -> TwoStageDetector:
    """按 weights/ 查找顺序装配两级 YOLO 检测器（engine 优先，缺失自动编译）。"""
    model_dir = _ROOT / "weights"
    car = _resolve_car(model_dir)
    if ARMOR_PATH:
        armor = first_existing((Path(ARMOR_PATH),))
    else:
        _maybe_build_engine(
            _ROOT / "tools" / "pt_to_engine.py",
            {
                "YOLO_PT": str(model_dir / "armor_best.pt"),
                "YOLO_ENGINE": str(model_dir / "armor_best.engine"),
                "YOLO_IMGSZ": "192",
                "YOLO_BATCH": "8",
            },
            model_dir / "armor_best.engine",
        )
        armor = first_existing((model_dir / "armor_best.engine", model_dir / "armor_best.pt"))
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


def load_classifier() -> tuple[PatternNet, PatternTransform, torch.device]:
    """图案分类器：engine 优先（GPU），pth 回退（强制 CPU，避免与 TRT 抢 cuDNN）。"""
    if PATTERN_PATH:
        path = first_existing((Path(PATTERN_PATH),))
    else:
        _maybe_build_engine(
            _ROOT / "tools" / "pattern_to_engine.py",
            {
                "PATTERN_PT": str(_ROOT / "weights" / "pattern_best.pth"),
                "PATTERN_ENGINE": str(_ROOT / "weights" / "pattern_best.engine"),
            },
            _ROOT / "weights" / "pattern_best.engine",
        )
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
