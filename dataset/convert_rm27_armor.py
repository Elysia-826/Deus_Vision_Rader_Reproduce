"""把 RM27 考核集 LabelMe 框转成装甲 YOLO 和图案裁剪。标签只做确定映射，不猜 2/4/Q。"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import assert_never

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.pattern.classes import ArmorFolder

ARMOR_OUT = ROOT / "dataset" / "armor_hard"
PATTERN_OUT = ROOT / "dataset" / "pattern_balanced" / "train"
PAD = 0.12


def default_src() -> Path:
    desktop = Path("C:/Users/YQS/Desktop")
    for exam in desktop.iterdir():
        if not exam.is_dir() or not exam.name.startswith("RM27"):
            continue
        for sub in exam.iterdir():
            if sub.is_dir() and (sub / "140.json").is_file():
                return sub
    raise FileNotFoundError("RM27 armor json dataset")


class UnknownRm27Label(Exception):
    def __init__(self, label: str, path: Path) -> None:
        self.label = label
        self.path = path
        super().__init__(f"{path}: unknown label {label}")


def pattern_folder(label: str) -> ArmorFolder:
    match label:
        case "blue1":
            return ArmorFolder.B1
        case "red1":
            return ArmorFolder.R1
        case "blue3":
            return ArmorFolder.B3
        case "red3":
            return ArmorFolder.R3
        case "bluesb":
            return ArmorFolder.BS
        case "redsb":
            return ArmorFolder.RS
        case _:
            raise UnknownRm27Label(label, Path("."))


def armor_class(label: str) -> int:
    """0 dead / 1 red / 2 blue。这批没有 dead。"""
    folder = pattern_folder(label)
    match folder:
        case ArmorFolder.R1 | ArmorFolder.R3 | ArmorFolder.RS | ArmorFolder.R0 | ArmorFolder.R2 | ArmorFolder.R4 | ArmorFolder.R5:
            return 1
        case ArmorFolder.B1 | ArmorFolder.B3 | ArmorFolder.BS | ArmorFolder.B0 | ArmorFolder.B2 | ArmorFolder.B4 | ArmorFolder.B5:
            return 2
        case unreachable:
            assert_never(unreachable)


def xyxy_from_points(points: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def main() -> None:
    src = default_src()
    if not src.is_dir():
        raise FileNotFoundError(src)
    img_dir = ARMOR_OUT / "images" / "train"
    lbl_dir = ARMOR_OUT / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_img = 0
    n_box = 0
    n_crop = 0
    for label_path in sorted(src.glob("*.json")):
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        stem = f"rm27_{label_path.stem}"
        src_img = src / f"{label_path.stem}.jpg"
        if not src_img.is_file():
            raise FileNotFoundError(src_img)
        encoded = np.fromfile(src_img, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(src_img)
        height, width = image.shape[0], image.shape[1]
        yolo_lines: list[str] = []
        for index, shape in enumerate(payload.get("shapes") or []):
            label = str(shape["label"])
            folder = pattern_folder(label)
            cls_id = armor_class(label)
            x1, y1, x2, y2 = xyxy_from_points(shape["points"])
            cx, cy, bw, bh = to_yolo(x1, y1, x2, y2, width, height)
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            pad_x = (x2 - x1) * PAD
            pad_y = (y2 - y1) * PAD
            ix1 = max(int(x1 - pad_x), 0)
            iy1 = max(int(y1 - pad_y), 0)
            ix2 = min(int(x2 + pad_x), width)
            iy2 = min(int(y2 + pad_y), height)
            crop = image[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                continue
            dest = PATTERN_OUT / folder.value
            dest.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", crop)
            if not ok:
                continue
            buf.tofile(str(dest / f"{stem}_{index}.jpg"))
            n_crop += 1
            n_box += 1
        if not yolo_lines:
            continue
        shutil.copy2(src_img, img_dir / f"{stem}.jpg")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
        n_img += 1
    print(f"armor images {n_img} boxes {n_box} pattern crops {n_crop}")


if __name__ == "__main__":
    main()
