"""扫描 ImageFolder、分层切 train/val。不碰 torch。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Final

from models.pattern.classes import PatternId, parse_armor_folder, pattern_for_folder
from models.pattern.errors import PatternDatasetError

_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png"})
_NESTED_ROOTS: Final[tuple[str, ...]] = ("RM2025-Armor-Pattern-Dataset", "armor_pattern")


@dataclass(frozen=True, slots=True)
class Sample:
    path: Path
    pattern: PatternId


def resolve_pattern_root(data: Path) -> Path:
    """接受公开集根、官方子目录、或已切好的 train/val。"""
    root = data.expanduser().resolve()
    if not root.exists():
        raise PatternDatasetError(root, "does not exist")
    for candidate in (root, *(root / name for name in _NESTED_ROOTS)):
        if _is_usable_root(candidate):
            return candidate
    raise PatternDatasetError(root, "no class folders or train/val split found")


def scan_class_folders(root: Path) -> tuple[Sample, ...]:
    """读 B0/B1/... 文件夹。5 号目录存在也会被丢掉。"""
    samples: list[Sample] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        folder = parse_armor_folder(child.name)
        if folder is None:
            continue
        pattern = pattern_for_folder(folder)
        if pattern is None:
            continue
        samples.extend(
            Sample(path=image, pattern=pattern)
            for image in sorted(child.iterdir())
            if image.is_file() and image.suffix.lower() in _IMAGE_SUFFIXES
        )
    if not samples:
        raise PatternDatasetError(root, "no trainable images under class folders")
    return tuple(samples)


def scan_train_val(root: Path) -> tuple[tuple[Sample, ...], tuple[Sample, ...]]:
    """已切好的 train/ + val/ 原样读，不再二次 split。"""
    return scan_class_folders(root / "train"), scan_class_folders(root / "val")


def split_samples(
    samples: tuple[Sample, ...],
    val_ratio: float,
    seed: int,
) -> tuple[tuple[Sample, ...], tuple[Sample, ...]]:
    """按图案类分层切分，同 seed 可复现。"""
    rng = Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []
    for pattern in PatternId:
        group = [item for item in samples if item.pattern is pattern]
        rng.shuffle(group)
        n_val = int(len(group) * val_ratio)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    if not train or not val:
        raise PatternDatasetError(Path("."), "split produced empty train or val")
    return tuple(train), tuple(val)


def load_split(root: Path, val_ratio: float, seed: int) -> tuple[tuple[Sample, ...], tuple[Sample, ...]]:
    """train/val 布局直接用；否则从类文件夹分层切。"""
    if _has_train_val(root):
        return scan_train_val(root)
    return split_samples(scan_class_folders(root), val_ratio=val_ratio, seed=seed)


def _has_train_val(root: Path) -> bool:
    return (root / "train").is_dir() and (root / "val").is_dir()


def _has_class_folders(root: Path) -> bool:
    if not root.is_dir():
        return False
    return any(parse_armor_folder(child.name) is not None for child in root.iterdir() if child.is_dir())


def _is_usable_root(root: Path) -> bool:
    return _has_train_val(root) or _has_class_folders(root)
