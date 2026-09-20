"""HKUST RM2025 雷达检测阶段配方，骨干从 YOLOv12 换成 YOLO26。

港科大原仓库是「粗到细」三级：
  1. 全图车辆检测  YOLOv12-s @ 1280
  2. 车辆 ROI 装甲板检测  YOLOv12-n @ 192
  3. 装甲板图案分类  MobileNet-V2 @ 64   ← 不是 YOLO，见 models.pattern

本文件只固化前两级的默认权重、输入尺寸和类别名。
CLI 没显式覆盖时，trainer 一律读这里。

他们开源的 car_training_config.yaml 把 imgsz 写成了 192，和 README 的 1280 矛盾。
雷达图是大视角，车在远处只有几十像素，必须按 README 用 1280。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never


class DetectStage(StrEnum):
    """两个 Ultralytics detect 阶段，字符串值就是 CLI 参数。"""

    CAR = "car"
    ARMOR = "armor"


@dataclass(frozen=True, slots=True)
class StageRecipe:
    """单个检测阶段的不可变配方。

    data_yaml / cfg_yaml 是相对 models/ 的路径，由 data_yaml.package_file 解析。
    names 的下标必须和数据集 txt 里的 class id 对齐，否则训练会学错类。
    """

    stage: DetectStage
    weights: str  # Ultralytics 会按文件名自动下载官方预训练权重
    imgsz: int
    epochs: int
    data_yaml: str
    cfg_yaml: str
    names: tuple[str, ...]


# 港科大 README：YOLOv12-s、全图 1280、单类 robot。
# 他们开源的 car_training_config.yaml 误写成 imgsz=192，这里按 README 纠正。
# s 档对应原 YOLOv12-s：精度/速度折中，适合 2000 万像素雷达图缩到 1280。
CAR_RECIPE: StageRecipe = StageRecipe(
    stage=DetectStage.CAR,
    weights="yolo26s.pt",
    imgsz=1280,
    epochs=500,
    data_yaml="configs/data_car.yaml",
    cfg_yaml="configs/car.yaml",
    names=("robot",),
)

# 港科大 README：YOLOv12-n、ROI 192。
# 标签：0 死亡灯条 / 1 红方 / 2 蓝方。图案数字不在这一级，交给分类器。
# n 档对应原 nano：输入小、要跑在每辆车的裁剪图上，优先延迟。
ARMOR_RECIPE: StageRecipe = StageRecipe(
    stage=DetectStage.ARMOR,
    weights="yolo26n.pt",
    imgsz=192,
    epochs=500,
    data_yaml="configs/data_armor.yaml",
    cfg_yaml="configs/armor.yaml",
    names=("dead", "red", "blue"),
)


def recipe_for(stage: DetectStage) -> StageRecipe:
    """按阶段取配方。match + assert_never：以后加阶段漏写分支会在类型检查期爆。"""
    match stage:
        case DetectStage.CAR:
            return CAR_RECIPE
        case DetectStage.ARMOR:
            return ARMOR_RECIPE
        case unreachable:
            assert_never(unreachable)
