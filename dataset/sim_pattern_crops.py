"""用仿真装甲贴图生成雷达视角图案裁剪，只进 train，不动 val。

车仿真图没有兵种标签，不能直接当 MobileNet 监督。贴图是仿真同一套官方字形，
透视/缩放到雷达远距离装甲的尺度，专门补 S/Q。
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from numpy import uint8
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "scene" / "armor_assets" / "textures"
STICKER = ROOT / "scene" / "armor_assets" / "stickers"
OUT = ROOT / "dataset" / "pattern_balanced" / "train"
RARE_N = 240
DIGIT_N = 40
SEED = 7
ImageU8 = NDArray[uint8]

# (贴图 stem 后缀, 红文件夹, 蓝文件夹)
PLATES: tuple[tuple[str, str, str], ...] = (
    ("1", "R1", "B1"),
    ("2", "R2", "B2"),
    ("3", "R3", "B3"),
    ("4", "R4", "B4"),
    ("sentry", "RS", "BS"),
)


def jitter_plate(plate: ImageU8, rng: random.Random) -> ImageU8:
    """把正视图贴图变成远距离雷达裁剪：透视、缩、暗、糊。"""
    height, width = plate.shape[:2]
    inset = rng.uniform(0.02, 0.12)
    src = np.float32(
        [
            [width * inset, height * inset],
            [width * (1 - inset), height * inset],
            [width * (1 - inset), height * (1 - inset)],
            [width * inset, height * (1 - inset)],
        ]
    )
    jx = rng.uniform(0.08, 0.22)
    jy = rng.uniform(0.08, 0.22)
    dst = np.float32(
        [
            [width * rng.uniform(0, jx), height * rng.uniform(0, jy)],
            [width * (1 - rng.uniform(0, jx)), height * rng.uniform(0, jy)],
            [width * (1 - rng.uniform(0, jx)), height * (1 - rng.uniform(0, jy))],
            [width * rng.uniform(0, jx), height * (1 - rng.uniform(0, jy))],
        ]
    )
    warped = cv2.warpPerspective(plate, cv2.getPerspectiveTransform(src, dst), (width, height), borderValue=(8, 8, 8))
    edge = rng.randint(28, 72)
    small = cv2.resize(warped, (edge, edge), interpolation=cv2.INTER_AREA)
    pad = rng.randint(2, 10)
    canvas = np.full((edge + 2 * pad, edge + 2 * pad, 3), rng.randint(4, 28), dtype=uint8)
    canvas[pad : pad + edge, pad : pad + edge] = small
    gain = rng.uniform(0.35, 0.95)
    dark = np.clip(canvas.astype(np.float32) * gain, 0, 255).astype(uint8)
    if rng.random() < 0.55:
        k = rng.choice((3, 5))
        dark = cv2.GaussianBlur(dark, (k, k), 0)
    return dark


def compose_outpost(color: str) -> ImageU8:
    glyph = cv2.imread(str(STICKER / "cutouts_ascii" / "outpost.jpg"))
    if glyph is None:
        glyph = cv2.imread(str(STICKER / "rm2026_cropped" / "outpost_small.png"))
    if glyph is None:
        raise FileNotFoundError("outpost sticker")
    size = 512
    canvas = np.full((size, size, 3), 8, dtype=uint8)
    led = (40, 40, 255) if color == "red" else (255, 80, 40)
    inset, led_w, vm = int(size * 0.08), int(size * 0.10), int(size * 0.08)
    for x0 in (inset, size - inset - led_w):
        canvas[vm : size - vm, x0 : x0 + led_w] = led
    gray = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) > 127:
        gray = 255 - gray
    inner = size - 2 * (inset + led_w)
    side = max(8, int(inner * 0.7))
    glyph_r = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
    ox = (size - side) // 2
    oy = (size - side) // 2
    alpha = (glyph_r.astype(np.float32) / 255.0)[..., None]
    region = canvas[oy : oy + side, ox : ox + side].astype(np.float32)
    white = np.array([245, 245, 245], dtype=np.float32)
    canvas[oy : oy + side, ox : ox + side] = np.clip(region * (1 - alpha) + white * alpha, 0, 255).astype(uint8)
    return canvas


def _write_folder(folder: str, plate: ImageU8, count: int, rng: random.Random) -> int:
    dest = OUT / folder
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for index in range(count):
        crop = jitter_plate(plate, rng)
        path = dest / f"sim_{folder}_{index:03d}.jpg"
        if cv2.imwrite(str(path), crop):
            written += 1
    return written


def main() -> None:
    rng = random.Random(SEED)
    counts: dict[str, int] = {}
    for stem, red, blue in PLATES:
        n = RARE_N if stem == "sentry" else DIGIT_N
        for color, folder in (("red", red), ("blue", blue)):
            plate = cv2.imread(str(TEX / f"ArmorTex_{color}_{stem}.png"))
            if plate is None:
                raise FileNotFoundError(TEX / f"ArmorTex_{color}_{stem}.png")
            counts[folder] = _write_folder(folder, plate, n, rng)
    for color, folder in (("red", "R0"), ("blue", "B0")):
        counts[folder] = _write_folder(folder, compose_outpost(color), RARE_N, rng)
    print("sim pattern crops", counts)


if __name__ == "__main__":
    main()
