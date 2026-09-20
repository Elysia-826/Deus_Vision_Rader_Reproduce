"""把 --data 收成 Ultralytics 能吃的 data.yaml。

两种合法输入（港科大开源集就是标准 YOLO 目录）：
  1. 现成 yaml：直接用，不改你的 path/names（推荐 models/configs/data_*.yaml）
  2. 数据集根目录：现场写出 data.yaml，names 取当前阶段配方

names 必须和 txt 里的 class id 对齐。车是 0=robot；装甲是 0=dead 1=red 2=blue。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from models.schema import DatasetConfigError
from models.stages import StageRecipe

# 无论从哪启动 CLI，配置都相对本文件所在的 models/ 找，不依赖 cwd
_PACKAGE_DIR: Path = Path(__file__).resolve().parent
_YAML_SUFFIXES: frozenset[str] = frozenset({".yaml", ".yml"})


def package_file(relative: str) -> Path:
    """解析 models/ 包内相对路径，例如 configs/car.yaml。"""
    return _PACKAGE_DIR / relative


def resolve_data_yaml(data: Path, recipe: StageRecipe) -> Path:
    """yaml 文件原样返回；目录则按配方生成 data.yaml。"""
    if data.suffix.lower() in _YAML_SUFFIXES:
        if not data.is_file():
            raise DatasetConfigError(data, "data yaml does not exist")
        return data
    if not data.is_dir():
        raise DatasetConfigError(data, "expected a data yaml or dataset directory")
    generated = data / "data.yaml"
    generated.write_text(_render_data_yaml(data, recipe), encoding="utf-8")
    return generated


def _render_data_yaml(root: Path, recipe: StageRecipe) -> str:
    # train/val 相对 path。港科大标注就是 images/ + labels/ 这套 YOLO 布局
    names = {index: name for index, name in enumerate(recipe.names)}
    payload: dict[str, str | int | dict[int, str]] = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "nc": len(recipe.names),
        "names": names,
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def load_cfg_overrides(cfg_path: Path) -> dict[str, str | int | float | bool]:
    """读 configs/*.yaml。None 和嵌套结构丢掉：Ultralytics train() 只要标量覆盖项。"""
    if not cfg_path.is_file():
        raise DatasetConfigError(cfg_path, "train cfg yaml does not exist")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DatasetConfigError(cfg_path, "train cfg must be a mapping")
    overrides: dict[str, str | int | float | bool] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, bool | int | float | str):
            overrides[str(key)] = value
    return overrides
