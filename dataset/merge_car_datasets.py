"""Merge car_yolo (already split) with flat car_sim into car_combined."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Final

from prepare_car_dataset import Sample, load_samples, stratified_split, write_yaml

SPLITS: Final[tuple[str, str, str]] = ("train", "val", "test")


def copy_split_tree(src_root: Path, dst_root: Path) -> dict[str, int]:
    counts = {name: 0 for name in SPLITS}
    for split in SPLITS:
        src_img = src_root / "images" / split
        if not src_img.is_dir():
            raise FileNotFoundError(src_img)
        for image in src_img.glob("*.jpg"):
            label = src_root / "labels" / split / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(label)
            shutil.copy2(image, dst_root / "images" / split / image.name)
            shutil.copy2(label, dst_root / "labels" / split / label.name)
            counts[split] += 1
    return counts


def copy_assigned(samples: list[Sample], assignment: dict[str, str], dst_root: Path) -> dict[str, int]:
    counts = {name: 0 for name in SPLITS}
    for sample in samples:
        split = assignment[sample.stem]
        shutil.copy2(sample.image, dst_root / "images" / split / sample.image.name)
        shutil.copy2(sample.label, dst_root / "labels" / split / sample.label.name)
        counts[split] += 1
    return counts


def merge(real_root: Path, sim_root: Path, out_root: Path, seed: int) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in SPLITS:
        (out_root / "images" / split).mkdir(parents=True)
        (out_root / "labels" / split).mkdir(parents=True)

    real_counts = copy_split_tree(real_root, out_root)
    sim_samples = load_samples(sim_root)
    sim_assignment = stratified_split(sim_samples, seed=seed)
    sim_counts = copy_assigned(sim_samples, sim_assignment, out_root)
    write_yaml(out_root / "data.yaml", out_root)

    lines = [
        "car combined 数据集",
        "",
        f"真实 {real_root}: {real_counts}",
        f"仿真 {sim_root}: {sim_counts}  (n={len(sim_samples)})",
        "合计：",
    ]
    for split in SPLITS:
        lines.append(f"  {split}: {real_counts[split] + sim_counts[split]}")
    text = "\n".join(lines) + "\n"
    (out_root / "SUMMARY.txt").write_text(text, encoding="utf-8")
    print(text)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Merge real YOLO car splits with sim dump")
    parser.add_argument("--real", type=Path, default=root / "car_yolo")
    parser.add_argument("--sim", type=Path, default=root / "car_sim")
    parser.add_argument("--out", type=Path, default=root / "car_combined")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge(args.real, args.sim, args.out, args.seed)
