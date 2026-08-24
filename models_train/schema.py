"""训练请求的边界类型：CLI / 调用方的原始输入只在这里解析一次。

内部 trainer 只收 TrainJob，不再二次校验路径或枚举。
None 字段的含义统一是「用 StageRecipe 默认值」，不要在调用处散落 or。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.stages import DetectStage, StageRecipe, recipe_for


class DatasetConfigError(Exception):
    """--data 既不是 yaml 也不是合法数据集根目录。"""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class TrainJob(BaseModel):
    """一次检测阶段训练的已解析请求。frozen：拼好后不允许再改。"""

    model_config = ConfigDict(frozen=True)

    stage: DetectStage
    data: Path
    # 下面几个 Optional：None = 走配方默认，有值 = CLI 覆盖
    model: str | None = None
    epochs: int | None = Field(default=None, ge=1)
    imgsz: int | None = Field(default=None, ge=32)
    batch: int | None = None  # -1 是 Ultralytics AutoBatch，合法，所以不加 ge
    device: str = "0"
    project: Path = Path("runs/detect")
    name: str | None = None
    resume: bool = False
    workers: int | None = Field(default=None, ge=0)
    pretrained: bool = True

    @field_validator("data", mode="before")
    @classmethod
    def _expand_data(cls, value: Path | str) -> Path:
        # ~ 和相对路径都收口成绝对路径，避免 cwd 一变数据找不到
        return Path(value).expanduser().resolve()

    @property
    def recipe(self) -> StageRecipe:
        return recipe_for(self.stage)

    @property
    def weights(self) -> str:
        """优先 CLI --model，否则配方里的 yolo26s/n.pt。"""
        return self.model if self.model is not None else self.recipe.weights

    @property
    def run_name(self) -> str:
        return self.name if self.name is not None else f"{self.stage}-yolo26"

    def resolved_epochs(self) -> int:
        return self.epochs if self.epochs is not None else self.recipe.epochs

    def resolved_imgsz(self) -> int:
        return self.imgsz if self.imgsz is not None else self.recipe.imgsz
