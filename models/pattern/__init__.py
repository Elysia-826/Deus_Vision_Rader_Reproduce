"""第三级：装甲图案分类。骨干 MobileNetV3-Small @ 64。

训练入口 python -m models.pattern.train。推理时把 PatternStage 注入 TwoStageDetector。
"""

from models.pattern.classes import CLASS_NAMES, PatternId
from models.pattern.schema import PatternTrainJob
from models.pattern.trainer import train

__all__ = ["CLASS_NAMES", "PatternId", "PatternTrainJob", "train"]
