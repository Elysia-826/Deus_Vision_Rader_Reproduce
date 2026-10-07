from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import numpy as np
from numpy import uint8
from numpy.typing import NDArray

from detect.pipeline import TwoStageDetector, default_config, with_device
from detect.types import BBox, Detection

ImageU8 = NDArray[uint8]


@dataclass(frozen=True, slots=True)
class _FakeTensor:
    rows: list[float] | list[list[float]]

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> list[float] | list[list[float]]:
        return self.rows


@dataclass(frozen=True, slots=True)
class _FakeBoxes:
    xyxy: _FakeTensor
    conf: _FakeTensor
    cls: _FakeTensor
    id: _FakeTensor | None = None


@dataclass(frozen=True, slots=True)
class _FakeResult:
    boxes: _FakeBoxes | None
    names: dict[int, str]


@dataclass(slots=True)
class _ScriptedModel:  # noqa: MUTABLE_OK — scripted queue is the test double
    """按调用顺序吐出预先排好的 Ultralytics 形结果。"""

    queue: list[_FakeResult]

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
    ) -> list[_FakeResult]:
        del imgsz, conf, iou, max_det, device, verbose
        match source:
            case list() as images:
                return [self.queue.pop(0) for _ in images]
            case np.ndarray():
                return [self.queue.pop(0)]
            case unreachable:
                assert_never(unreachable)

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
    ) -> list[_FakeResult]:
        del persist, tracker
        return self.predict(
            source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=verbose,
        )


def test_infer_remaps_armor_into_full_frame() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    car_model = _ScriptedModel(
        [
            _FakeResult(
                boxes=_FakeBoxes(
                    xyxy=_FakeTensor([[20.0, 10.0, 80.0, 60.0]]),
                    conf=_FakeTensor([0.9]),
                    cls=_FakeTensor([0.0]),
                ),
                names={0: "robot"},
            )
        ]
    )
    armor_model = _ScriptedModel(
        [
            _FakeResult(
                boxes=_FakeBoxes(
                    xyxy=_FakeTensor([[5.0, 6.0, 15.0, 16.0]]),
                    conf=_FakeTensor([0.7]),
                    cls=_FakeTensor([1.0]),
                ),
                names={0: "dead", 1: "red", 2: "blue"},
            )
        ]
    )
    detector = TwoStageDetector(default_config("car.pt", "armor.pt"), car_model, armor_model)
    result = detector.infer(frame)
    assert len(result.robots) == 1
    robot = result.robots[0]
    assert robot.car == Detection(label="robot", conf=0.9, box=BBox(20, 10, 80, 60))
    assert robot.armors == (Detection(label="red", conf=0.7, box=BBox(25, 16, 35, 26)),)


def test_with_device_only_changes_device() -> None:
    cfg = default_config("car.pt", "armor.pt", "0")
    cpu = with_device(cfg.car, "cpu")
    assert cpu.device == "cpu"
    assert cpu.imgsz == cfg.car.imgsz
    assert cpu.max_det == cfg.car.max_det


def test_infer_drops_tall_false_car() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    car_model = _ScriptedModel(
        [
            _FakeResult(
                boxes=_FakeBoxes(
                    xyxy=_FakeTensor([[10.0, 0.0, 30.0, 80.0]]),
                    conf=_FakeTensor([0.9]),
                    cls=_FakeTensor([0.0]),
                ),
                names={0: "robot"},
            )
        ]
    )
    armor_model = _ScriptedModel([])
    detector = TwoStageDetector(default_config("car.pt", "armor.pt"), car_model, armor_model)
    assert detector.infer(frame).robots == ()


def test_infer_skips_empty_car_frame() -> None:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    car_model = _ScriptedModel([_FakeResult(boxes=None, names={0: "robot"})])
    armor_model = _ScriptedModel([])
    detector = TwoStageDetector(default_config("car.pt", "armor.pt"), car_model, armor_model)
    assert detector.infer(frame).robots == ()
