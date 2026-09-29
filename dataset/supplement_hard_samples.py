"""补难例，不改原公开集。

图案：公开集按训练同款 seed 切分，S/Q 只在 train 里扩到约 600。
装甲：小框样本做暗光/模糊副本，val/test 不动。
车辆：真实 car_yolo 全留，另从仿真里抽 ≥3 车的图只进 train。
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.pattern.classes import PatternId  # noqa: E402
from models.pattern.samples import resolve_pattern_root, scan_class_folders, split_samples  # noqa: E402

PATTERN_SRC = Path(r"C:\Users\YQS\Downloads\OneDrive_3_2026-8-21\RM2025-Armor-Pattern-Public-Dataset")
PATTERN_OUT = _ROOT / "dataset" / "pattern_balanced"
ARMOR_SRC = _ROOT / "dataset" / "armor_yolo"
ARMOR_OUT = _ROOT / "dataset" / "armor_hard"
CAR_SRC = _ROOT / "dataset" / "car_yolo"
CAR_SIM = _ROOT / "dataset" / "car_sim"
CAR_OUT = _ROOT / "dataset" / "car_hard"
RARE = {PatternId.SENTRY: "BS", PatternId.OUTPOST: "B0"}
RARE_TARGET = 600
SMALL_SIDE = 0.12
ARMOR_EXTRA = 600
CAR_SIM_CAP = 600
SEED = 42


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _write_pattern() -> str:
    samples = scan_class_folders(resolve_pattern_root(PATTERN_SRC))
    train, val = split_samples(samples, val_ratio=0.2, seed=SEED)
    _reset(PATTERN_OUT)
    for split, items in (("train", train), ("val", val)):
        for item in items:
            folder = PATTERN_OUT / split / item.path.parent.name
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.path, folder / item.path.name)
    rng = random.Random(SEED)
    added = 0
    for pattern, folder_name in RARE.items():
        pool = [item.path for item in train if item.pattern is pattern]
        dest = PATTERN_OUT / "train" / folder_name
        dest.mkdir(parents=True, exist_ok=True)
        have = len(pool)
        index = 0
        while have < RARE_TARGET and pool:
            image = cv2.imread(str(pool[index % len(pool)]))
            if image is None:
                index += 1
                continue
            aug = _photometric(image, rng)
            cv2.imwrite(str(dest / f"aug_{pattern.name}_{have}.jpg"), aug)
            have += 1
            added += 1
            index += 1
    return f"pattern train extra {added}"


def _photometric(image: np.ndarray, rng: random.Random) -> np.ndarray:
    scale = rng.uniform(0.45, 0.85)
    dark = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        dark = cv2.GaussianBlur(dark, (3, 3), 0)
    return dark


def _small_label(path: Path) -> bool:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return False
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        if max(float(parts[3]), float(parts[4])) < SMALL_SIDE:
            return True
    return False


def _copy_yolo(src: Path, dst: Path) -> None:
    _reset(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _write_armor() -> str:
    _copy_yolo(ARMOR_SRC, ARMOR_OUT)
    train_img = ARMOR_OUT / "images" / "train"
    train_lbl = ARMOR_OUT / "labels" / "train"
    small = [path for path in sorted(train_lbl.glob("*.txt")) if _small_label(path)]
    rng = random.Random(SEED)
    added = 0
    for label in small:
        if added >= ARMOR_EXTRA:
            break
        image_path = train_img / f"{label.stem}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        stem = f"{label.stem}_hard{added}"
        cv2.imwrite(str(train_img / f"{stem}.jpg"), _photometric(image, rng))
        shutil.copy2(label, train_lbl / f"{stem}.txt")
        added += 1
    return f"armor small copies {added}"


def _box_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8").strip()
    return 0 if not text else len(text.splitlines())


def _write_car() -> str:
    _copy_yolo(CAR_SRC, CAR_OUT)
    crowded = [path for path in sorted(CAR_SIM.glob("*.txt")) if _box_count(path) >= 3][:CAR_SIM_CAP]
    added = 0
    for label in crowded:
        image = label.with_suffix(".jpg")
        if not image.is_file():
            continue
        stem = f"sim_{label.stem}"
        shutil.copy2(image, CAR_OUT / "images" / "train" / f"{stem}.jpg")
        shutil.copy2(label, CAR_OUT / "labels" / "train" / f"{stem}.txt")
        added += 1
    return f"car sim crowded {added}"


def main() -> None:
    lines = [_write_pattern(), _write_armor(), _write_car()]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
