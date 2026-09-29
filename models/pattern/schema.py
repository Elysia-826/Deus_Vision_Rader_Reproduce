"""第三级训练请求。CLI 原始输入只在这里解析一次。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.pattern.model import PatternBackbone

DEFAULT_PATTERN_DATA: Path = Path(
    r"C:\Users\YQS\Downloads\OneDrive_3_2026-8-21\RM2025-Armor-Pattern-Public-Dataset"
)


class PatternTrainJob(BaseModel):
    """一次 MobileNetV3-Small 图案分类训练。frozen：拼好后不许再改。"""

    model_config = ConfigDict(frozen=True)

    data: Path
    epochs: int = Field(default=100, ge=1)
    batch: int = Field(default=32, ge=1)
    imgsz: int = Field(default=64, ge=32)
    learning_rate: float = Field(default=0.0003, gt=0.0)
    val_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42
    device: str = "0"
    project: Path = Path("runs/pattern")
    name: str = "mobilenetv3-small"
    workers: int | None = Field(default=None, ge=0)
    pretrained: bool = True
    use_class_weight: bool = True
    backbone: PatternBackbone = PatternBackbone.MOBILENET_V3_SMALL
    init_weights: Path | None = None

    @field_validator("data", mode="before")
    @classmethod
    def _expand_data(cls, value: Path | str) -> Path:
        return Path(value).expanduser().resolve()
