"""Ultralytics engine 的直接 TensorRT 推理。

laugh12321/TensorRT-YOLO 要另编一套带 EfficientNMS 插件的 engine，而且没有 Python 3.14 wheel。
本机 armor/car engine 已经是端到端 (1, 300, 6)。这里去掉 Ultralytics predict 包装，
预处理、enqueue、后处理留在同一条 CUDA stream 上。车级跟踪仍用 BoT-SORT。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
from numpy import float32, uint8
from numpy.typing import NDArray

from detect.e2e import select_e2e
from detect.errors import EngineLoadError
from detect.geometry import ImageU8
from detect.letterbox import Letterbox, letterbox_params, undo_xyxy

_PAD = 114


@dataclass(frozen=True, slots=True)
class _HostTensor:
    rows: list[float] | list[list[float]]

    def detach(self) -> _HostTensor:
        return self

    def cpu(self) -> _HostTensor:
        return self

    def tolist(self) -> list[float] | list[list[float]]:
        return self.rows


@dataclass(frozen=True, slots=True)
class _HostBoxes:
    xyxy: _HostTensor
    conf: _HostTensor
    cls: _HostTensor
    id: _HostTensor | None = None


@dataclass(frozen=True, slots=True)
class _HostResult:
    boxes: _HostBoxes | None
    names: dict[int, str]


@dataclass(frozen=True, slots=True)
class _DetView:
    """BoT-SORT 要的最小 Boxes：xyxy / xywh / conf / cls，支持布尔索引。"""

    xyxy: NDArray[float32]
    xywh: NDArray[float32]
    conf: NDArray[float32]
    cls: NDArray[float32]

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, index: NDArray[np.bool_] | NDArray[np.intp]) -> _DetView:
        return _DetView(
            xyxy=self.xyxy[index],
            xywh=self.xywh[index],
            conf=self.conf[index],
            cls=self.cls[index],
        )


class TrtDetectModel:  # noqa: MUTABLE_OK — 持有 engine context、stream 和跟踪器状态
    """一个静态 batch=1 engine。多张图复用同一 context，不再走 YOLO.predict。"""

    def __init__(self, weights: Path) -> None:
        payload, meta = _split_engine(weights)
        logger = trt.Logger(trt.Logger.ERROR)
        engine = trt.Runtime(logger).deserialize_cuda_engine(payload)
        if engine is None:
            raise EngineLoadError(weights, "TensorRT refused the engine")
        self._engine = engine
        self._context = engine.create_execution_context()
        self._input, self._output = _io_names(engine)
        in_shape = tuple(int(dim) for dim in engine.get_tensor_shape(self._input))
        self._size = int(in_shape[2])
        self._half = engine.get_tensor_dtype(self._input) == trt.DataType.HALF
        self._names = _class_names(meta)
        self._stream = torch.cuda.Stream()
        dtype = torch.float16 if self._half else torch.float32
        self._inp = torch.empty((1, 3, self._size, self._size), device="cuda", dtype=dtype)
        self._out = torch.empty((1, 300, 6), device="cuda", dtype=torch.float32)
        self._tracker: object | None = None
        self._tracker_yaml: str | None = None

    def predict(
        self,
        source: ImageU8 | list[ImageU8],
        *,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        device: str,
        verbose: bool,
    ) -> list[_HostResult]:
        del imgsz, iou, device, verbose
        images = source if isinstance(source, list) else [source]
        return [self._one(image, conf, max_det, track_ids=None) for image in images]

    def track(
        self,
        source: ImageU8 | list[ImageU8],
        *,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        device: str,
        verbose: bool,
        persist: bool,
        tracker: str,
    ) -> list[_HostResult]:
        del imgsz, iou, device, verbose, persist
        images = source if isinstance(source, list) else [source]
        return [self._track_one(image, conf, max_det, tracker) for image in images]

    def _one(
        self,
        image: ImageU8,
        conf: float,
        max_det: int,
        track_ids: list[int] | None,
    ) -> _HostResult:
        rows = self._forward(image)
        picked = select_e2e(rows, conf, max_det)
        if picked.conf.shape[0] == 0:
            return _HostResult(boxes=None, names=self._names)
        ids = None if track_ids is None else _HostTensor([float(item) for item in track_ids])
        return _HostResult(
            boxes=_HostBoxes(
                xyxy=_HostTensor(picked.xyxy.tolist()),
                conf=_HostTensor(picked.conf.tolist()),
                cls=_HostTensor(picked.cls.tolist()),
                id=ids,
            ),
            names=self._names,
        )

    def _track_one(self, image: ImageU8, conf: float, max_det: int, tracker: str) -> _HostResult:
        rows = self._forward(image)
        picked = select_e2e(rows, conf=0.01, max_det=max(max_det, 10))
        view = _view_from(picked.xyxy, picked.conf, picked.cls)
        tracks = self._botsort(tracker).update(view, image, None)
        if tracks.shape[0] == 0:
            return _HostResult(boxes=None, names=self._names)
        kept = tracks[:max_det]
        return _HostResult(
            boxes=_HostBoxes(
                xyxy=_HostTensor(kept[:, :4].tolist()),
                conf=_HostTensor(kept[:, 5].tolist()),
                cls=_HostTensor(kept[:, 6].tolist()),
                id=_HostTensor(kept[:, 4].tolist()),
            ),
            names=self._names,
        )

    def _forward(self, image: ImageU8) -> NDArray[float32]:
        height, width = image.shape[0], image.shape[1]
        pad = letterbox_params(width, height, self._size)
        canvas = _canvas(image, pad, self._size)
        uploaded = torch.from_numpy(canvas).cuda()
        rgb = uploaded[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
        self._inp.copy_(rgb.to(dtype=self._inp.dtype).div_(255))
        self._context.set_tensor_address(self._input, int(self._inp.data_ptr()))
        self._context.set_tensor_address(self._output, int(self._out.data_ptr()))
        if not self._context.execute_async_v3(int(self._stream.cuda_stream)):
            raise EngineLoadError(Path(self._input), "execute_async_v3 failed")
        self._stream.synchronize()
        raw = self._out[0].detach().cpu().numpy().copy()
        raw[:, :4] = undo_xyxy(raw[:, :4], pad)
        return raw

    def _botsort(self, tracker: str) -> object:
        if self._tracker is not None and self._tracker_yaml == tracker:
            return self._tracker
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import YAML, IterableSimpleNamespace

        cfg = IterableSimpleNamespace(**YAML.load(tracker))
        self._tracker = BOTSORT(args=cfg, frame_rate=30)
        self._tracker_yaml = tracker
        return self._tracker


def _split_engine(path: Path) -> tuple[bytes, dict[str, object]]:
    data = path.read_bytes()
    prefix = int.from_bytes(data[:4], "little", signed=True)
    if prefix <= 0 or prefix > 1_000_000:
        raise EngineLoadError(path, "missing Ultralytics metadata prefix")
    meta_raw = json.loads(data[4 : 4 + prefix].decode("utf-8"))
    if not isinstance(meta_raw, dict):
        raise EngineLoadError(path, "metadata is not an object")
    return data[4 + prefix :], meta_raw


def _io_names(engine: trt.ICudaEngine) -> tuple[str, str]:
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    inputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    outputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
    if len(inputs) != 1 or len(outputs) != 1:
        raise EngineLoadError(Path(inputs[0] if inputs else "engine"), "expected one input and one output")
    return inputs[0], outputs[0]


def _class_names(meta: dict[str, object]) -> dict[int, str]:
    raw = meta.get("names", {})
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {index: str(name) for index, name in enumerate(raw)}
    return {0: "item"}


def _canvas(image: ImageU8, pad: Letterbox, size: int) -> NDArray[uint8]:
    """灰边画布保持 uint8。归一化和换通道在 GPU 上做，避免 1280² float32 的 CPU 拷贝。"""
    resized = cv2.resize(image, (pad.resized_w, pad.resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), _PAD, dtype=uint8)
    canvas[pad.top : pad.top + pad.resized_h, pad.left : pad.left + pad.resized_w] = resized
    return np.ascontiguousarray(canvas)


def _view_from(xyxy: NDArray[float32], conf: NDArray[float32], cls: NDArray[float32]) -> _DetView:
    if xyxy.shape[0] == 0:
        empty4 = xyxy.reshape(0, 4)
        empty = conf.reshape(0)
        return _DetView(xyxy=empty4, xywh=empty4, conf=empty, cls=empty)
    xywh = np.empty_like(xyxy)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return _DetView(xyxy=xyxy, xywh=xywh, conf=conf, cls=cls)
