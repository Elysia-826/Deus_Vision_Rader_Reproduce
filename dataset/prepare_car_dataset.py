"""Split the flat car dump into a YOLO detect dataset.

Keeps dataset/car untouched. Writes dataset/car_yolo/ at 7:2:1.
Labels stay class 0 robot: `class cx cy w h` (normalized).
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

SPLITS: Final[tuple[str, str, str]] = ("train", "val", "test")
SPLIT_RATIO: Final[tuple[float, float, float]] = (0.7, 0.2, 0.1)


@dataclass(frozen=True, slots=True)
class Sample:
    stem: str
    image: Path
    label: Path
    box_count: int


class DatasetLayoutError(Exception):
    """jpg/txt pair missing or label line is not YOLO detect format."""


def parse_box_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    count = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise DatasetLayoutError(f"{path.name}: expected 5 fields, got {len(parts)}")
        count += 1
    return count


def load_samples(src: Path) -> list[Sample]:
    images = sorted(src.glob("*.jpg"))
    if not images:
        raise DatasetLayoutError(f"no jpg files in {src}")
    samples: list[Sample] = []
    for image in images:
        label = image.with_suffix(".txt")
        if not label.exists():
            raise DatasetLayoutError(f"missing label for {image.name}")
        samples.append(
            Sample(
                stem=image.stem,
                image=image,
                label=label,
                box_count=parse_box_count(label),
            )
        )
    return samples


def bucket_key(box_count: int) -> str:
    if box_count <= 2:
        return str(box_count)
    if box_count <= 4:
        return "3-4"
    if box_count <= 7:
        return "5-7"
    if box_count <= 10:
        return "8-10"
    return "11+"


def split_counts(n: int) -> tuple[int, int, int]:
    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])
    n_test = int(n * SPLIT_RATIO[2])
    leftover = n - n_train - n_val - n_test
    n_train += leftover
    if n >= 3 and n_test == 0:
        n_test = 1
        n_train = max(1, n_train - 1)
    if n >= 2 and n_val == 0:
        n_val = 1
        n_train = max(1, n_train - 1)
    return n_train, n_val, n_test


def stratified_split(samples: list[Sample], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    buckets: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        buckets[bucket_key(sample.box_count)].append(sample)

    assignment: dict[str, str] = {}
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n_train, n_val, _n_test = split_counts(len(bucket))
        parts = [
            bucket[:n_train],
            bucket[n_train : n_train + n_val],
            bucket[n_train + n_val :],
        ]
        for split, members in zip(SPLITS, parts, strict=True):
            for sample in members:
                assignment[sample.stem] = split
    return assignment


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def write_yaml(path: Path, dataset_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Auto-generated from dataset/prepare_car_dataset.py",
                "# class matches HKUST RM car stage: single-class robot",
                f"path: {dataset_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names:",
                "  0: robot",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_summary(
    yolo_root: Path,
    samples: list[Sample],
    assignment: dict[str, str],
    seed: int,
) -> None:
    by_split: dict[str, list[Sample]] = {name: [] for name in SPLITS}
    for sample in samples:
        by_split[assignment[sample.stem]].append(sample)
    lines = [
        "car 数据集整理报告",
        "",
        f"原始目录：dataset/car  （未改动，{len(samples)} 对 jpg+txt）",
        f"YOLO 目录：{yolo_root}",
        f"随机种子：{seed}",
        "划分按每图框数分桶：1 / 2 / 3-4 / 5-7 / 8-10 / 11+",
        "",
        "划分（按图）：",
    ]
    for name in SPLITS:
        members = by_split[name]
        boxes = sum(s.box_count for s in members)
        lines.append(f"  {name}: {len(members)} 图  boxes={boxes}")
    lines.extend(
        [
            "",
            "YOLO 目录结构：",
            "  images/{train,val,test}/*.jpg",
            "  labels/{train,val,test}/*.txt",
            "  data.yaml",
            "",
        ]
    )
    text = "\n".join(lines)
    (yolo_root / "SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text)


def prepare(src: Path, yolo_root: Path, seed: int) -> None:
    samples = load_samples(src)
    assignment = stratified_split(samples, seed=seed)
    reset_dir(yolo_root)
    for name in SPLITS:
        (yolo_root / "images" / name).mkdir(parents=True)
        (yolo_root / "labels" / name).mkdir(parents=True)
    for sample in samples:
        split = assignment[sample.stem]
        shutil.copy2(sample.image, yolo_root / "images" / split / sample.image.name)
        shutil.copy2(sample.label, yolo_root / "labels" / split / sample.label.name)
    write_yaml(yolo_root / "data.yaml", yolo_root)
    write_summary(yolo_root, samples, assignment, seed)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Organize car dump into YOLO 7:2:1 splits")
    parser.add_argument("--src", type=Path, default=root / "car")
    parser.add_argument("--yolo-out", type=Path, default=root / "car_yolo")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.src, args.yolo_out, args.seed)
