"""YOLO26 训练包：对外只暴露阶段枚举、请求类型和 train()。

推理不在这里，在 detect/。图案训练在 models.pattern。
"""

from models.schema import TrainJob
from models.stages import DetectStage, recipe_for
from models.trainer import train

__all__ = ["DetectStage", "TrainJob", "recipe_for", "train"]
