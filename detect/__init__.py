"""检测包：港科大雷达「粗到细」前三级的推理侧。

港科大上场链是 相机 → 三级检测 → CascadeMatchTracker → 射线投影 → 裁判。
本包只覆盖检测：

1. 全图车辆 YOLO（可选 BoT-SORT，给车一个跨帧 track_id）
2. 车辆 ROI 内装甲板 YOLO（dead / red / blue）
3. 装甲裁剪上的图案分类（1/2/3/4/S/Q），颜色不在这一级

训练代码在 models/，这里只加载已有权重做 infer。
"""

from detect.errors import EmptyCropError, ImageReadError, ModelFileNotFoundError
from detect.pattern import PatternStage, attach_patterns
from detect.pipeline import TwoStageDetector, default_config, warmup, with_device
from detect.types import Detection, FrameResult, LinkedRobot, PatternPred, TwoStageConfig
from detect.weights import first_existing

__all__ = [
    "Detection",
    "EmptyCropError",
    "FrameResult",
    "ImageReadError",
    "LinkedRobot",
    "ModelFileNotFoundError",
    "PatternPred",
    "PatternStage",
    "TwoStageConfig",
    "TwoStageDetector",
    "attach_patterns",
    "default_config",
    "first_existing",
    "warmup",
    "with_device",
]
