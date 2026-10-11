# ─── How to run ───
#     python scripts/eval_foot_ray.py
#     python scripts/eval_foot_ray.py --weights runs/detect/runs/detect/car-yolo26s-patrol-foot/weights/best.pt --old-weights weights/car_best.pt
# ──────────────────

"""新车权重的上场验收：held-out sim 帧上，预测框底边打 mesh，和标签底边比。

标签底边按构造就是地面触地点（generate_radar_car_sim.yolo_box 把下边拉到
触地点投影），所以“标签底边射线”即 GT 足点。验收标准（和 track/observe 同链路）：
预测框走 foot_pixel → mesh.pixel_to_field，落到场内且和 GT 足点误差 ≤1.2m。
过线才允许替换 weights/car_best.engine，不过线不换。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ultralytics import YOLO

from detect.geometry import clamp_box, plausible_car_box
from detect.types import BBox
from locate.camera import CameraPose, camera_from_path
from locate.field import FIELD_X_MAX, FIELD_X_MIN, FIELD_Y_MAX, FIELD_Y_MIN
from locate.ray_mesh import FieldMesh
from locate.ray_plane import foot_pixel
from locate.types import Pixel

CALIB = _ROOT / "locate" / "calib"
BLUE_JSON = CALIB / "Camera_Radar_Blue.json"
RED_JSON = CALIB / "Camera_Radar_Red.json"
MESH_NPZ = CALIB / "field_mesh.npz"
TEST_IMG = _ROOT / "dataset" / "car_combined_patrol_foot_8mm" / "images" / "test"
TEST_LBL = _ROOT / "dataset" / "car_combined_patrol_foot_8mm" / "labels" / "test"

CONF = 0.45
IOU_NMS = 0.5
IOU_MATCH = 0.5
PASS_M = 1.2
_SLACK_M = 2.0  # 和 track/observe._on_field 一致


def _on_field(x: float, y: float) -> bool:
    return (
        FIELD_X_MIN - _SLACK_M <= x <= FIELD_X_MAX + _SLACK_M
        and FIELD_Y_MIN - _SLACK_M <= y <= FIELD_Y_MAX + _SLACK_M
    )


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _read_labels(path: Path, width: int, height: int) -> list[BBox]:
    boxes: list[BBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = (float(v) for v in parts)
        x1 = int((cx - bw / 2.0) * width)
        y1 = int((cy - bh / 2.0) * height)
        x2 = int((cx + bw / 2.0) * width)
        y2 = int((cy + bh / 2.0) * height)
        box = clamp_box(BBox(x1=x1, y1=y1, x2=x2, y2=y2), width, height)
        if box is not None:
            boxes.append(box)
    return boxes


def _foot_world(mesh: FieldMesh, pose: CameraPose, box: BBox) -> tuple[float, float] | None:
    """框底边的 mesh 世界足点；射线落空则 None。"""
    hit = mesh.pixel_to_field(pose, foot_pixel(box))
    if hit is None:
        return None
    return (hit.x, hit.y)


def _eval_weights(
    tag: str, weights: Path, frames: list[Path], blue, red, mesh: FieldMesh
) -> dict[str, float | int]:
    model = YOLO(str(weights))
    n_gt = 0
    n_matched = 0
    n_gt_ray_ok = 0
    n_pred_off = 0
    n_pred_miss = 0
    errors: list[float] = []
    dists: list[float] = []
    over_detail: list[tuple[str, BBox, BBox, float, float, float]] = []
    for image_path in frames:
        index = int(image_path.stem.split("_")[1])
        pose = blue if index % 2 == 0 else red
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        gt_boxes = _read_labels(TEST_LBL / f"{image_path.stem}.txt", width, height)
        n_gt += len(gt_boxes)
        result = model.predict(
            image, imgsz=1280, conf=CONF, iou=IOU_NMS, max_det=10, verbose=False
        )[0]
        preds: list[tuple[BBox, float]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            for row, conf in zip(xyxy, confs, strict=True):
                box = clamp_box(
                    BBox(x1=int(row[0]), y1=int(row[1]), x2=int(row[2]), y2=int(row[3])),
                    width,
                    height,
                )
                if box is not None and plausible_car_box(box):
                    preds.append((box, float(conf)))
        used = [False] * len(gt_boxes)
        for pred, _ in sorted(preds, key=lambda item: item[1], reverse=True):
            best, best_iou, best_k = None, IOU_MATCH, -1
            for k, gt in enumerate(gt_boxes):
                if used[k]:
                    continue
                iou = _iou(pred, gt)
                if iou >= best_iou:
                    best, best_iou, best_k = gt, iou, k
            if best is None:
                continue
            used[best_k] = True
            n_matched += 1
            gt_world = _foot_world(mesh, pose, best)
            if gt_world is None:
                continue
            n_gt_ray_ok += 1
            pred_world = _foot_world(mesh, pose, pred)
            if pred_world is None:
                n_pred_miss += 1
                continue
            if not _on_field(*pred_world):
                n_pred_off += 1
                continue
            errors.append(
                ((pred_world[0] - gt_world[0]) ** 2 + (pred_world[1] - gt_world[1]) ** 2)
                ** 0.5
            )
            dists.append((gt_world[0] ** 2 + gt_world[1] ** 2) ** 0.5)
            if errors[-1] > PASS_M:
                over_detail.append(
                    (image_path.stem, pred, best, best_iou, errors[-1], dists[-1])
                )
    over = sum(1 for e in errors if e > PASS_M)
    details = sorted(over_detail, key=lambda d: d[4], reverse=True)
    for stem, pred, gt, iou, err, dist in details:
        print(
            f"  OVER {stem} iou={iou:.2f} err={err:.3f}m gtdist={dist:.1f}m "
            f"pred=({pred.x1},{pred.y1},{pred.x2},{pred.y2}) "
            f"gt=({gt.x1},{gt.y1},{gt.x2},{gt.y2})"
        )
    stats: dict[str, float | int] = {
        "frames": len(frames),
        "n_gt": n_gt,
        "n_matched": n_matched,
        "n_gt_ray_ok": n_gt_ray_ok,
        "n_pred_miss": n_pred_miss,
        "n_pred_off": n_pred_off,
        "n_err": len(errors),
        "over": over,
    }
    if errors:
        ordered = sorted(errors)
        stats["median"] = ordered[len(ordered) // 2]
        stats["max"] = ordered[-1]
        near = [e for e, d in zip(errors, dists, strict=True) if d <= 12.0]
        stats["near_n"] = len(near)
        stats["near_max"] = max(near) if near else -1.0
    print(
        f"[{tag}] frames={len(frames)} gt={n_gt} matched={n_matched} "
        f"gt_ray_ok={n_gt_ray_ok} pred_miss={n_pred_miss} pred_offfield={n_pred_off} "
        + (
            f"err_n={len(errors)} median={stats['median']:.3f}m "
            f"max={stats['max']:.3f}m over_{PASS_M}m={over} "
            f"near12m_n={stats['near_n']} near12m_max={stats['near_max']:.3f}m"
            if errors
            else "no errors measured"
        )
    )
    return stats


def _passed(stats: dict[str, float | int]) -> bool:
    return (
        int(stats["n_err"]) > 0
        and int(stats["n_pred_miss"]) == 0
        and int(stats["n_pred_off"]) == 0
        and int(stats["over"]) == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="foot-ray acceptance for new car weights")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--old-weights", type=Path, default=None)
    args = parser.parse_args()

    blue = camera_from_path(BLUE_JSON)
    red = camera_from_path(RED_JSON)
    mesh = FieldMesh.from_npz(MESH_NPZ)
    frames = sorted(TEST_IMG.glob("sim_*.jpg"))
    if not frames:
        print("no sim_*.jpg in test")
        return 2
    print(f"sim test frames: {len(frames)}")

    if args.old_weights is not None and args.old_weights.exists():
        _eval_weights("old", args.old_weights, frames, blue, red, mesh)
    stats = _eval_weights("new", args.weights, frames, blue, red, mesh)
    ok = _passed(stats)
    print("ACCEPT" if ok else "REJECT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
