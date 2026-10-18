# ─── How to run ───
#     python dataset/render_car_crops.py --seq scene/locate_out/videoseq_dwell --tag dwell
# ──────────────────

"""从渲染序列收割车 YOLO 训练数据（全帧 + 检测器伪标签 + GT 验证）。

车检测是形状任务（找"机器人"），不受灯条颜色/图案影响——收割的主要
目的是补 Standard 视图变换域的样本。检测器定位是准的，GT 验证滤掉
匹配不上真值的车框（防伪影/误检进训练集）。

输出 YOLO 格式：images/train/<tag>_*.jpg（1920×1080） + labels/train/<tag>_*.txt
  class cx cy w h（归一化）
  class: 0 = robot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy import uint8, zeros

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import visualize_two_stage as vis2  # noqa: E402
from locate.camera import camera_from_path  # noqa: E402
from locate.ray_mesh import FieldMesh  # noqa: E402
from track.observe import foot_pixel  # noqa: E402

POSE_BLUE = _ROOT / "locate" / "calib" / "Camera_Radar_Blue.json"
MESH = _ROOT / "locate" / "calib" / "field_mesh.npz"
DEFAULT_SEQ = _ROOT / "scene" / "locate_out" / "videoseq_dwell"
DEFAULT_OUT = _ROOT / "dataset" / "car_combined"
MATCH_M = 2.0
OUT_W, OUT_H = 1920, 1080  # 与现有训练集同分辨率


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=Path, default=DEFAULT_SEQ)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", default="sim")
    parser.add_argument("--max", type=int, default=300)
    parser.add_argument("--pose", type=Path, default=POSE_BLUE, help="相机外参 JSON")
    parser.add_argument("--mesh", type=Path, default=MESH, help="场地网格 npz")
    args = parser.parse_args()

    pose = camera_from_path(args.pose)
    mesh = FieldMesh.from_npz(args.mesh)
    net, transform, device = vis2._load_classifier()
    detector = vis2._load_detector(vis2.PatternStage(net=net, transform=transform, device=str(device)))
    detector.infer(zeros((3000, 4096, 3), dtype=uint8), persist_tracks=False)

    img_dir = args.out / "images" / "train"
    lbl_dir = args.out / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for frame_path in sorted(args.seq.glob("*.jpg")):
        if written >= args.max:
            break
        meta_path = frame_path.with_suffix(".json")
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(meta.get("camera", "Camera_Radar_Blue")) != "Camera_Radar_Blue":
            continue
        gts = [(r["x"], r["y"]) for r in meta.get("robots", [])]
        if not gts:
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        result = detector.infer(frame, persist_tracks=False)
        if not result.robots:
            continue

        # 收集通过 GT 验证的车框
        lines: list[str] = []
        for robot in result.robots:
            pixel = foot_pixel(robot.car.box)
            hit = mesh.pixel_to_field(pose, pixel)
            if hit is None:
                continue
            best = min(gts, key=lambda g: (g[0] - hit.x) ** 2 + (g[1] - hit.y) ** 2)
            if float(np.hypot(best[0] - hit.x, best[1] - hit.y)) > MATCH_M:
                continue
            box = robot.car.box
            cx = (box.x1 + box.x2) / 2.0 / frame.shape[1]
            cy = (box.y1 + box.y2) / 2.0 / frame.shape[0]
            w = (box.x2 - box.x1) / frame.shape[1]
            h = (box.y2 - box.y1) / frame.shape[0]
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if not lines:
            continue

        stem = f"{args.tag}_{frame_path.stem}"
        resized = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        if cv2.imwrite(str(img_dir / f"{stem}.jpg"), resized, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            written += 1
    print(f"harvested {written} frames -> {img_dir}")


if __name__ == "__main__":
    main()
