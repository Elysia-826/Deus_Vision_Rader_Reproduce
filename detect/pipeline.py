"""全图车辆 YOLO → ROI 装甲 YOLO →（可选）图案分类。

阈值抄港科大 params.yaml：车 conf 0.2 / 1280 / max_det 10；装甲 conf 0.3 / 192 / max_det 1。
车级开 BoT-SORT：雷达相机常俯仰扫场，ByteTrack 没有 GMC，全图一平移就集体换 ID。
装甲级只用 predict：裁剪没有帧间连续性，persist 只会串 ID。

pattern 是注入的 PatternStage，不是 YOLO。没注入时 infer 行为与两级完全相同。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, assert_never

from detect.geometry import ImageU8, clamp_box, crop_roi, remap_box

CAR_TRACKER_YAML: Path = Path(__file__).resolve().parent / "trackers" / "botsort.yaml"
from detect.parse import detections_from_rows
from detect.pattern import PatternStage, attach_patterns
from detect.types import (
    Detection,
    FrameResult,
    LinkedRobot,
    StageInferConfig,
    TwoStageConfig,
)


def with_device(cfg: StageInferConfig, device: str) -> StageInferConfig:
    """只换 device。.pt 和 .engine 同卡混用时把 pt 那一级改到 cpu。"""
    return StageInferConfig(
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        max_det=cfg.max_det,
        device=device,
        tracker=cfg.tracker,
    )


class _TensorLike(Protocol):
    def detach(self) -> _TensorLike: ...
    def cpu(self) -> _TensorLike: ...
    def tolist(self) -> list[float] | list[list[float]]: ...


class _YoloBoxes(Protocol):
    xyxy: _TensorLike
    conf: _TensorLike
    cls: _TensorLike
    id: _TensorLike | None


class _YoloResult(Protocol):
    boxes: _YoloBoxes | None
    names: dict[int, str] | list[str]


class _YoloModel(Protocol):
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
    ) -> list[_YoloResult]: ...

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
    ) -> list[_YoloResult]: ...


def default_config(car_weights: str, armor_weights: str, device: str = "0") -> TwoStageConfig:
    """港科大 params.yaml 的推理阈值 + 本仓 YOLO26 输入尺寸。"""
    return TwoStageConfig(
        car_weights=car_weights,
        armor_weights=armor_weights,
        car=StageInferConfig(
            imgsz=1280,
            conf=0.2,
            iou=0.5,
            max_det=10,
            device=device,
            tracker=str(CAR_TRACKER_YAML),
        ),
        armor=StageInferConfig(imgsz=192, conf=0.3, iou=0.4, max_det=1, device=device),
    )


class TwoStageDetector:
    """车 YOLO → 装甲 YOLO → 可选图案分类。pattern=None 时退回两级。"""

    def __init__(
        self,
        config: TwoStageConfig,
        car_model: _YoloModel,
        armor_model: _YoloModel,
        *,
        pattern: PatternStage | None = None,
    ) -> None:
        self._config = config
        self._car = car_model
        self._armor = armor_model
        self._pattern = pattern

    @classmethod
    def from_config(cls, config: TwoStageConfig, *, pattern: PatternStage | None = None) -> TwoStageDetector:
        return cls(
            config,
            _load_detector(config.car_weights),
            _load_detector(config.armor_weights),
            pattern=pattern,
        )

    def infer(self, frame: ImageU8, *, persist_tracks: bool = True) -> FrameResult:
        """一帧：跟车 → 裁 ROI → 检装甲并回投全图 → 可选图案。

        persist_tracks=False 给 warmup 用，避免首帧编译写进 BoT-SORT 轨迹。
        """
        height, width = frame.shape[0], frame.shape[1]
        kept: list[Detection] = []
        crops: list[ImageU8] = []
        for car in self._run(self._car, frame, self._config.car, persist_tracks=persist_tracks):
            roi = clamp_box(car.box, width, height)
            if roi is None:
                continue
            crops.append(crop_roi(frame, roi))
            kept.append(Detection(label=car.label, conf=car.conf, box=roi, track_id=car.track_id))
        if not crops:
            return FrameResult(robots=())

        armor_groups = self._run_batch(self._armor, crops, self._config.armor, persist_tracks=False)
        robots = tuple(
            LinkedRobot(
                car=car,
                armors=tuple(
                    Detection(
                        label=armor.label,
                        conf=armor.conf,
                        box=remap_box(armor.box, car.box),
                        track_id=armor.track_id,
                    )
                    for armor in armors
                ),
            )
            for car, armors in zip(kept, armor_groups, strict=True)
        )
        if self._pattern is not None:
            robots = attach_patterns(frame, robots, self._pattern)
        return FrameResult(robots=robots)

    def _run(
        self,
        model: _YoloModel,
        image: ImageU8,
        cfg: StageInferConfig,
        persist_tracks: bool,
    ) -> tuple[Detection, ...]:
        tracker = cfg.tracker
        if persist_tracks and tracker is not None:
            result = model.track(
                image,
                imgsz=cfg.imgsz,
                conf=cfg.conf,
                iou=cfg.iou,
                max_det=cfg.max_det,
                device=cfg.device,
                verbose=False,
                persist=True,
                tracker=tracker,
            )[0]
        else:
            result = model.predict(
                image,
                imgsz=cfg.imgsz,
                conf=cfg.conf,
                iou=cfg.iou,
                max_det=cfg.max_det,
                device=cfg.device,
                verbose=False,
            )[0]
        return _detections_from_result(result)

    def _run_batch(
        self,
        model: _YoloModel,
        images: list[ImageU8],
        cfg: StageInferConfig,
        persist_tracks: bool,
    ) -> tuple[tuple[Detection, ...], ...]:
        # armor_best.engine 是静态 batch=1，不能把多辆车 ROI 叠成一次 predict
        return tuple(self._run(model, image, cfg, persist_tracks) for image in images)


def _names_table(raw: dict[int, str] | list[str]) -> dict[int, str]:
    match raw:
        case dict() as mapping:
            return {int(key): str(value) for key, value in mapping.items()}
        case list() as seq:
            return {index: str(name) for index, name in enumerate(seq)}
        case unreachable:
            assert_never(unreachable)


def _as_float_rows(value: _TensorLike) -> list[list[float]]:
    raw = value.detach().cpu().tolist()
    return [[float(cell) for cell in row] for row in raw]


def _as_floats(value: _TensorLike) -> list[float]:
    raw = value.detach().cpu().tolist()
    return [float(item) for item in raw]


def _detections_from_result(result: _YoloResult) -> tuple[Detection, ...]:
    boxes = result.boxes
    if boxes is None:
        return ()
    xyxy_rows = _as_float_rows(boxes.xyxy)
    if not xyxy_rows:
        return ()
    confs = _as_floats(boxes.conf)
    class_ids = [int(item) for item in _as_floats(boxes.cls)]
    raw_ids = boxes.id
    track_ids = [int(item) for item in _as_floats(raw_ids)] if raw_ids is not None else None
    return detections_from_rows(xyxy_rows, confs, class_ids, _names_table(result.names), track_ids)


def _load_detector(weights: str) -> _YoloModel:
    """`.engine` 走直接 TensorRT。`.pt` 仍用 Ultralytics，方便没编 engine 时调试。"""
    path = Path(weights)
    if path.suffix.lower() == ".engine":
        from detect.trt_model import TrtDetectModel

        return TrtDetectModel(path)
    from ultralytics import YOLO

    return YOLO(weights, task="detect")


def warmup(detector: TwoStageDetector, frame: ImageU8) -> None:
    """丢掉首帧编译/engine 加载。走 predict，避免污染 BoT-SORT 状态。"""
    detector.infer(frame, persist_tracks=False)
