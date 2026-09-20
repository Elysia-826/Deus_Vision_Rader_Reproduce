"""检测链路上的值对象。不进 Ultralytics，不进磁盘。

坐标约定：像素 xyxy，左上含、右下不含，和 numpy 切片 `img[y1:y2, x1:x2]` 一致。
装甲板 Detection.label 永远是灯条颜色（dead/red/blue），兵种放 pattern。
港科大把颜色和数字揉进一个分类器；本仓拆开，避免红 1 / 蓝 1 抢同一视觉特征。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BBox:
    """像素坐标 xyxy，左上含、右下不含（与 numpy 切片一致）。"""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True, slots=True)
class PatternPred:
    """第三级图案。name ∈ CLASS_NAMES（1/2/3/4/S/Q），S=哨兵，Q=前哨（港科大文件夹 B0/R0）。"""

    name: str
    conf: float


@dataclass(frozen=True, slots=True)
class Detection:
    """单阶段一次检出。

    车辆：label=robot，track_id 来自 BoT-SORT（predict 路径为 None）。
    装甲：label=dead/red/blue，pattern 为兵种；没有接 PatternStage 时 pattern 为 None。
    """

    label: str
    conf: float
    box: BBox
    track_id: int | None = None
    pattern: PatternPred | None = None


@dataclass(frozen=True, slots=True)
class LinkedRobot:
    """一辆车 + 其 ROI 内回投到全图的装甲板。后续定位/报点都按这辆走。"""

    car: Detection
    armors: tuple[Detection, ...]


@dataclass(frozen=True, slots=True)
class FrameResult:
    robots: tuple[LinkedRobot, ...]


@dataclass(frozen=True, slots=True)
class StageInferConfig:
    """单级 YOLO 推理参数。和港科大 params.yaml 里 car/armor 两段对应。

    tracker：None 走 predict；给 yaml 路径则 model.track(persist=True)。
    装甲级必须 tracker=None：ROI 每帧都是新裁剪，没有跨帧运动模型可跟。
    """

    imgsz: int
    conf: float
    iou: float
    max_det: int
    device: str
    tracker: str | None = None


@dataclass(frozen=True, slots=True)
class TwoStageConfig:
    """一次双阶段推理的已解析配置。weights 必须是已存在的文件。"""

    car_weights: str
    armor_weights: str
    car: StageInferConfig
    armor: StageInferConfig
