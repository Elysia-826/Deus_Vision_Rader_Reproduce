"""Official sticker = full armor face albedo. Only the two stock white strips emit."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\YQS\Desktop\DEUS_VISION_RADER_TEST_reproduce\scene")
CUTOUT = ROOT / "armor_assets" / "stickers" / "cutouts_ascii"
OUT = ROOT / "armor_assets" / "textures"
OUT.mkdir(parents=True, exist_ok=True)

JOBS = [
    ("1", "1.jpg", True),
    ("2", "2.jpg", True),
    ("3", "3_small.jpg", True),
    ("4", "4_small.jpg", True),
    ("sentry", "sentry.jpg", False),
]


def load(name: str) -> np.ndarray:
    p = CUTOUT / name
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img


def clean_albedo(bgr: np.ndarray) -> np.ndarray:
    """Keep official ink; crush near-black to matte black. No extra margin."""
    out = bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out[gray < 28] = (0, 0, 0)
    return out


def strip_mask(bgr: np.ndarray) -> np.ndarray:
    """White vertical bars at left/right of the official sticker, not the glyph."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bright = gray > 150
    col = bright.mean(axis=0)

    def band(lo, hi):
        # Official strips sit on the far left/right edges and are nearly solid.
        idx = np.where(col[lo:hi] > 0.42)[0] + lo
        if idx.size == 0:
            idx = np.where(col[lo:hi] > 0.28)[0] + lo
        if idx.size == 0:
            return None
        runs = []
        s = idx[0]
        prev = idx[0]
        for x in idx[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append((s, prev))
                s = prev = x
        runs.append((s, prev))
        c0, c1 = max(runs, key=lambda r: (r[1] - r[0] + 1) * col[r[0] : r[1] + 1].mean())
        # refuse a run that is too wide (would be the glyph)
        if (c1 - c0 + 1) > w * 0.18:
            return None
        return c0, c1 + 1

    left = band(0, max(2, w * 18 // 100))
    right = band(w - max(2, w * 18 // 100), w)
    mask = np.zeros((h, w), np.uint8)
    for band_cols in (left, right):
        if band_cols is None:
            continue
        c0, c1 = band_cols
        # only the bright pixels inside the official strip
        col_slice = bright[:, c0:c1]
        mask[:, c0:c1][col_slice] = 255
    if mask.max() == 0:
        # fallback: leftmost/rightmost 11% where bright
        lw = max(2, w * 11 // 100)
        mask[:, :lw][bright[:, :lw]] = 255
        mask[:, w - lw :][bright[:, w - lw :]] = 255
    # keep strips as solid bars (close pinholes) but do not dilate outward
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def orient(img: np.ndarray, rotate_180: bool) -> np.ndarray:
    # Plate UV is 90° CW from sticker space; CCW puts number upright with bars L/R.
    out = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate_180:
        out = cv2.rotate(out, cv2.ROTATE_180)
    return out


def main():
    for key, src, rot180 in JOBS:
        raw = load(src)
        albedo = clean_albedo(raw)
        mask = strip_mask(raw)
        albedo = orient(albedo, rot180)
        mask = orient(mask, rot180)
        for color in ("red", "blue"):
            name = f"ArmorTex_{color}_{key}.png"
            cv2.imwrite(str(OUT / name), albedo)
            cv2.imwrite(str(ROOT / name), albedo)
        mname = f"ArmorMask_{key}.png"
        cv2.imwrite(str(OUT / mname), mask)
        print(
            f"{key}: albedo {albedo.shape} mask_px={int(mask.sum() / 255)} "
            f"cover={mask.mean() / 255:.3f} rot180={rot180}"
        )


if __name__ == "__main__":
    main()
