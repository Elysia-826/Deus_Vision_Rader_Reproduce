from pathlib import Path

import pytest

from models.pattern.classes import PatternId
from models.pattern.errors import PatternDatasetError
from models.pattern.samples import (
    resolve_pattern_root,
    scan_class_folders,
    split_samples,
)


def _touch_jpg(folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(b"\xff\xd8\xff")


def test_scan_mixes_color_and_drops_five(tmp_path: Path) -> None:
    _touch_jpg(tmp_path / "B1", "a.jpg")
    _touch_jpg(tmp_path / "R1", "b.jpg")
    _touch_jpg(tmp_path / "B5", "skip.jpg")
    samples = scan_class_folders(tmp_path)
    assert len(samples) == 2
    assert {item.pattern for item in samples} == {PatternId.DIGIT_1}


def test_split_is_stratified_and_seeded(tmp_path: Path) -> None:
    for index in range(10):
        _touch_jpg(tmp_path / "B2", f"{index}.jpg")
    samples = scan_class_folders(tmp_path)
    train_a, val_a = split_samples(samples, val_ratio=0.2, seed=7)
    train_b, val_b = split_samples(samples, val_ratio=0.2, seed=7)
    assert len(val_a) == 2
    assert len(train_a) == 8
    assert train_a == train_b
    assert val_a == val_b


def test_resolve_nested_official_folder(tmp_path: Path) -> None:
    nested = tmp_path / "RM2025-Armor-Pattern-Dataset"
    _touch_jpg(nested / "B3", "x.jpg")
    assert resolve_pattern_root(tmp_path) == nested.resolve()


def test_resolve_rejects_xany_flat_dir(tmp_path: Path) -> None:
    (tmp_path / "B1__foo.jpg").write_bytes(b"\xff\xd8\xff")
    with pytest.raises(PatternDatasetError):
        resolve_pattern_root(tmp_path)
