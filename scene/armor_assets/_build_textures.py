"""Build YOLO armor textures from official RM2026 appendix stickers + LED bars."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\YQS\Desktop\DEUS_VISION_RADER_TEST_reproduce\scene")
ASSETS = ROOT / "armor_assets"
APPENDIX = ASSETS / "stickers" / "rm2026_appendix"
CUTOUT = ASSETS / "stickers" / "cutouts_ascii"
OUT_STICKER = ASSETS / "stickers" / "rm2026_cropped"
OUT_TEX = ASSETS / "textures"
SCENE_TEX = ROOT  # existing ArmorTex_*.png live here

OUT_STICKER.mkdir(parents=True, exist_ok=True)
OUT_TEX.mkdir(parents=True, exist_ok=True)

# mm
SMALL_W, SMALL_H = 140.0, 125.0  # AM02
LARGE_W, LARGE_H = 235.0, 127.0  # AM12
PX_PER_MM = 8.0
LED_W_MM = 10.0
LED_INSET_MM = 4.0
LED_VMARGIN_MM = 8.0

BLUE = (255, 80, 40)  # BGR, saturated blue-ish for LED
RED = (40, 40, 255)
LED_CORE_BLUE = (255, 220, 180)
LED_CORE_RED = (180, 220, 255)
BLACK = (8, 8, 8)


def find_black_panels(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < 40).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = gray.shape
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 8000:
            continue
        if bw < 80 or bh < 80:
            continue
        if x < 5 or y < 5 or x + bw > w - 5 or y + bh > h - 5:
            # keep anyway if reasonably interior
            pass
        boxes.append((x, y, bw, bh))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def crop_appendix():
    mapping = {
        103: [("hero_1_large", 0), ("engineer_2_small", 1)],
        104: [("infantry_3_small", 0), ("infantry_4_small", 1)],
        105: [("sentry_small", 0), ("outpost_small", 1)],
    }
    results = {}
    for page, items in mapping.items():
        path = APPENDIX / f"page_{page:03d}.png"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        boxes = find_black_panels(img)
        # keep the two largest
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:2]
        boxes.sort(key=lambda b: b[1])
        for name, idx in items:
            if idx >= len(boxes):
                continue
            x, y, bw, bh = boxes[idx]
            pad = 2
            crop = img[y + pad : y + bh - pad, x + pad : x + bw - pad]
            outp = OUT_STICKER / f"{name}.png"
            cv2.imwrite(str(outp), crop)
            results[name] = outp
            print(f"crop {name} {crop.shape} from p{page} box={boxes[idx]}")
    return results


def sticker_mask(img_bgr: np.ndarray) -> np.ndarray:
    """White glyph on black -> grayscale 0-255 glyph."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # invert so glyph is white
    if gray.mean() > 127:
        gray = 255 - gray
    # stretch
    lo, hi = np.percentile(gray, (5, 99))
    if hi <= lo:
        hi = lo + 1
    g = np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return g


def load_cutout(filename: str) -> np.ndarray | None:
    p = CUTOUT / filename
    if not p.exists():
        return None
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


def compose_armor(glyph: np.ndarray, width_mm: float, height_mm: float, color: str) -> np.ndarray:
    # Square so cube-face UVs do not crop. LED bars sit well inside the face.
    size = 1024
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:] = BLACK

    led_bgr = RED if color == "red" else BLUE
    core = LED_CORE_RED if color == "red" else LED_CORE_BLUE
    inset = int(size * 0.08)
    led_w = int(size * 0.10)
    vm = int(size * 0.08)
    for x0 in (inset, size - inset - led_w):
        canvas[vm : size - vm, x0 : x0 + led_w] = led_bgr
        core_w = max(3, led_w // 3)
        cx = x0 + (led_w - core_w) // 2
        canvas[vm + 6 : size - vm - 6, cx : cx + core_w] = core

    gx0 = inset + led_w + int(size * 0.06)
    gx1 = size - inset - led_w - int(size * 0.06)
    gy0 = int(size * 0.16)
    gy1 = size - int(size * 0.16)
    gw, gh = gx1 - gx0, gy1 - gy0
    glyph_u8 = sticker_mask(glyph) if glyph.ndim == 3 else glyph
    gh0, gw0 = glyph_u8.shape[:2]
    scale = min(gw / gw0, gh / gh0) * 0.62
    nw, nh = max(1, int(gw0 * scale)), max(1, int(gh0 * scale))
    glyph_r = cv2.resize(glyph_u8, (nw, nh), interpolation=cv2.INTER_AREA)
    ox = gx0 + (gw - nw) // 2
    oy = gy0 + (gh - nh) // 2
    region = canvas[oy : oy + nh, ox : ox + nw].astype(np.float32)
    a = (glyph_r.astype(np.float32) / 255.0)[..., None]
    white = np.array([245, 245, 245], dtype=np.float32)
    canvas[oy : oy + nh, ox : ox + nw] = np.clip(region * (1 - a) + white * a, 0, 255).astype(np.uint8)
    # Plate UV is 90° CW from the image; CCW the sticker so number is upright and LEDs sit left/right.
    return cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)


def tight_sticker(img_bgr: np.ndarray) -> np.ndarray:
    """Keep the solid black sticker plate, drop dimension arrows."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < 50).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 40 or bh < 40:
            continue
        fill = cv2.contourArea(c) / float(bw * bh)
        ar = bw / float(bh)
        if fill < 0.55:
            continue
        if not (0.7 < ar < 2.4):
            continue
        score = fill * bw * bh
        if score > best_score:
            best_score = score
            best = (x, y, bw, bh)
    if best is None:
        return img_bgr
    x, y, bw, bh = best
    pad = 1
    return img_bgr[y + pad : y + bh - pad, x + pad : x + bw - pad]


def main():
    crops = crop_appendix()
    jobs = [
        ("1", "hero_1_large", "1.jpg", True),
        ("2", "engineer_2_small", "2.jpg", False),
        ("3", "infantry_3_small", "3_small.jpg", False),
        ("4", "infantry_4_small", "4_small.jpg", False),
        ("sentry", "sentry_small", "sentry.jpg", False),
    ]
    for key, crop_name, cutout_name, large in jobs:
        # 抠图处理后 is the official number/icon without dimension arrows.
        glyph = load_cutout(cutout_name)
        if glyph is None:
            crop_path = crops.get(crop_name)
            if crop_path and crop_path.exists():
                pdf_crop = cv2.imread(str(crop_path))
                if pdf_crop is not None:
                    glyph = tight_sticker(pdf_crop)
        if glyph is None:
            raise RuntimeError(f"missing glyph for {key}")
        print(f"glyph {key} source-cutout={cutout_name} shape={glyph.shape}")
        wmm, hmm = (LARGE_W, LARGE_H) if large else (SMALL_W, SMALL_H)
        for color in ("red", "blue"):
            tex = compose_armor(glyph, wmm, hmm, color)
            if key != "sentry":
                tex = cv2.rotate(tex, cv2.ROTATE_180)
            name = f"ArmorTex_{color}_{key}.png"
            cv2.imwrite(str(OUT_TEX / name), tex)
            cv2.imwrite(str(SCENE_TEX / name), tex)
            print(f"wrote {name} {tex.shape}")


if __name__ == "__main__":
    main()
