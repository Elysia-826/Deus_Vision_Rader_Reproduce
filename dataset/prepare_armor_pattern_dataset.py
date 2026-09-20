"""Reorganize RM2025 armor-pattern crops for X-AnyLabeling classification.

Keeps the original class folders (copy). Writes a flat X-AnyLabeling dir:
  classes.txt
  <class>__<stem>.jpg
  <class>__<stem>.json   flags one-hot, same schema as X-AnyLabeling Image Classifier
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image

CLASS_NAMES: Final[tuple[str, ...]] = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "BS",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "RS",
)
IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
JSON_VERSION: Final[str] = "3.2.3"


@dataclass(frozen=True, slots=True)
class Sample:
    cls: str
    image: Path
    stem: str


class DatasetLayoutError(Exception):
    """Source is not the RM2025 folder-per-class dump."""


def flags_for(true_class: str) -> dict[str, bool]:
    if true_class not in CLASS_NAMES:
        raise DatasetLayoutError(f"unknown class folder: {true_class}")
    return {name: name == true_class for name in CLASS_NAMES}


def xany_stem(sample: Sample) -> str:
    return f"{sample.cls}__{sample.image.stem}"


def write_classes(path: Path) -> None:
    path.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")


def write_label_json(path: Path, image_name: str, sample: Sample, width: int, height: int) -> None:
    payload = {
        "version": JSON_VERSION,
        "flags": flags_for(sample.cls),
        "shapes": [],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "description": "",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_samples(src: Path) -> list[Sample]:
    if not src.is_dir():
        raise DatasetLayoutError(f"not a directory: {src}")
    samples: list[Sample] = []
    missing = [name for name in CLASS_NAMES if not (src / name).is_dir()]
    extra = sorted(
        path.name
        for path in src.iterdir()
        if path.is_dir() and path.name not in CLASS_NAMES
    )
    if missing:
        raise DatasetLayoutError(f"missing class folders: {missing}")
    if extra:
        raise DatasetLayoutError(f"unexpected folders: {extra}")
    for name in CLASS_NAMES:
        images = sorted(
            path
            for path in (src / name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise DatasetLayoutError(f"no images in {src / name}")
        for image in images:
            samples.append(Sample(cls=name, image=image, stem=image.stem))
    return samples


def copy_class_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def prepare(src: Path, archive: Path, xany_root: Path) -> None:
    samples = load_samples(src)
    copy_class_tree(src, archive)

    if xany_root.exists():
        shutil.rmtree(xany_root)
    xany_root.mkdir(parents=True)
    write_classes(xany_root / "classes.txt")
    write_classes(archive / "classes.txt")

    used: set[str] = set()
    counts = {name: 0 for name in CLASS_NAMES}
    for sample in samples:
        stem = xany_stem(sample)
        if stem in used:
            raise DatasetLayoutError(f"name collision: {stem}")
        used.add(stem)
        image_name = f"{stem}{sample.image.suffix.lower()}"
        dest_image = xany_root / image_name
        shutil.copy2(sample.image, dest_image)
        with Image.open(sample.image) as im:
            width, height = im.size
        write_label_json(xany_root / f"{stem}.json", image_name, sample, width, height)
        counts[sample.cls] += 1

    lines = [
        "armor pattern 数据集（X-AnyLabeling 分类）",
        "",
        f"原始：{src}",
        f"归档：{archive}  （按类文件夹，未改标注）",
        f"X-AnyLabeling：{xany_root}",
        "",
        "classes.txt 顺序（0-index）：",
    ]
    for index, name in enumerate(CLASS_NAMES):
        lines.append(f"  {index}: {name}  n={counts[name]}")
    lines.extend(
        [
            "",
            f"合计：{len(samples)} 张",
            "",
            "X-AnyLabeling 用法：",
            "  1. Image Classifier 上传 classes.txt 作为 flags",
            "  2. 打开 armor_pattern_xany 目录（jpg 与 json 同级）",
            "",
            "B/R = 蓝/红；0-5 = 图案编号；S = 哨兵",
            "",
        ]
    )
    text = "\n".join(lines) + "\n"
    (xany_root / "SUMMARY.txt").write_text(text, encoding="utf-8")
    (archive / "SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="RM2025 armor-pattern -> X-AnyLabeling classification")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(
            r"C:\Users\YQS\Downloads\OneDrive_3_2026-8-21\RM2025-Armor-Pattern-Public-Dataset\RM2025-Armor-Pattern-Dataset"
        ),
    )
    parser.add_argument("--archive", type=Path, default=root / "armor_pattern")
    parser.add_argument("--xany-out", type=Path, default=root / "armor_pattern_xany")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.src, args.archive, args.xany_out)
