"""画面地标像素 ↔ 场地米制。sports ViewTransformer 同源：findHomography / getPerspectiveTransform。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

from locate.errors import HomographyError
from locate.types import FieldXY, Pixel

FloatMat = NDArray[np.float64]


class _PairFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    pixel: tuple[float, float]
    field: tuple[float, float]


class _HomographyFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    pairs: tuple[_PairFile, ...]


@dataclass(frozen=True, slots=True)
class HomographyMap:
    matrix: FloatMat
    landmark_ids: tuple[str, ...]

    def pixel_to_field(self, pixel: Pixel) -> FieldXY:
        import cv2

        src = np.array([[[pixel.u, pixel.v]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.matrix)
        x, y = dst.reshape(2)
        return FieldXY(x=float(x), y=float(y))


def homography_from_pairs(
    pixels: tuple[Pixel, ...],
    fields: tuple[FieldXY, ...],
    landmark_ids: tuple[str, ...],
) -> HomographyMap:
    import cv2

    if len(pixels) != len(fields) or len(pixels) != len(landmark_ids):
        raise HomographyError(path=None, reason="pixels/fields/ids length mismatch")
    if len(pixels) < 4:
        raise HomographyError(path=None, reason=f"need at least 4 pairs, got {len(pixels)}")
    src = np.array([[p.u, p.v] for p in pixels], dtype=np.float32)
    dst = np.array([[p.x, p.y] for p in fields], dtype=np.float32)
    if len(pixels) == 4:
        matrix = cv2.getPerspectiveTransform(src, dst)
    else:
        matrix, _mask = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise HomographyError(path=None, reason="findHomography failed")
    matrix.flags.writeable = False
    return HomographyMap(matrix=matrix.astype(np.float64), landmark_ids=landmark_ids)


def homography_from_path(path: Path) -> HomographyMap:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HomographyError(path=path, reason="cannot read file") from exc
    except json.JSONDecodeError as exc:
        raise HomographyError(path=path, reason=f"invalid json: {exc.msg}") from exc
    try:
        parsed = _HomographyFile.model_validate(payload)
    except ValidationError as exc:
        raise HomographyError(path=path, reason=str(exc)) from exc
    pixels = tuple(Pixel(u=item.pixel[0], v=item.pixel[1]) for item in parsed.pairs)
    fields = tuple(FieldXY(x=item.field[0], y=item.field[1]) for item in parsed.pairs)
    ids = tuple(item.id for item in parsed.pairs)
    return homography_from_pairs(pixels, fields, ids)


def dump_homography_pairs(
    landmark_ids: tuple[str, ...],
    pixels: tuple[Pixel, ...],
    fields: tuple[FieldXY, ...],
) -> dict[str, object]:
    return {
        "pairs": [
            {
                "id": landmark_id,
                "pixel": [pixel.u, pixel.v],
                "field": [field.x, field.y],
            }
            for landmark_id, pixel, field in zip(landmark_ids, pixels, fields, strict=True)
        ]
    }
