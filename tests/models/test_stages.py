from pathlib import Path

import pytest

from models.data_yaml import resolve_data_yaml
from models.schema import DatasetConfigError, TrainJob
from models.stages import DetectStage, recipe_for
from models.trainer import build_train_kwargs


def test_car_recipe_uses_yolo26s_and_full_frame() -> None:
    recipe = recipe_for(DetectStage.CAR)
    assert recipe.weights == "yolo26s.pt"
    assert recipe.imgsz == 1280
    assert recipe.names == ("robot",)


def test_armor_recipe_uses_yolo26n_and_roi_size() -> None:
    recipe = recipe_for(DetectStage.ARMOR)
    assert recipe.weights == "yolo26n.pt"
    assert recipe.imgsz == 192
    assert recipe.names == ("dead", "red", "blue")


def test_resolve_data_yaml_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "gone.yaml"
    with pytest.raises(DatasetConfigError):
        resolve_data_yaml(missing, recipe_for(DetectStage.CAR))


def test_resolve_data_yaml_writes_root_layout(tmp_path: Path) -> None:
    generated = resolve_data_yaml(tmp_path, recipe_for(DetectStage.ARMOR))
    text = generated.read_text(encoding="utf-8")
    assert generated.name == "data.yaml"
    assert "dead" in text
    assert "images/train" in text


def test_build_train_kwargs_cli_overrides_recipe(tmp_path: Path) -> None:
    data_yaml = tmp_path / "custom.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\n", encoding="utf-8")
    job = TrainJob(
        stage=DetectStage.CAR,
        data=data_yaml,
        epochs=12,
        imgsz=640,
        batch=8,
        device="cpu",
        name="smoke",
    )
    kwargs = build_train_kwargs(job)
    assert kwargs["epochs"] == 12
    assert kwargs["imgsz"] == 640
    assert kwargs["batch"] == 8
    assert kwargs["device"] == "cpu"
    assert kwargs["name"] == "smoke"
    assert Path(str(kwargs["data"])).name == "custom.yaml"
