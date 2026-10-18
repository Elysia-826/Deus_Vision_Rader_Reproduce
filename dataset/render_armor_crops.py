# ─── How to run ───
#     python dataset/render_armor_crops.py
#     python dataset/render_armor_crops.py --seq scene/locate_out/videoseq_lamp --tag lamp
# ──────────────────

"""从渲染序列收割装甲 YOLO 训练数据（车 ROI + 装甲框 + GT 队伍重标）。

同 pattern 收割的思路：检测器的定位是好的（框准），只是红蓝类别在
仿真域 ~40% 误读——用 GT 队伍重标类别、保留检测器自己的框做定位，
就是干净的训练数据。

输出 YOLO 格式：images/<tag>_*.jpg + labels/<tag>_*.txt
  class cx cy w h（归一化，ROI 局部坐标系）
  class: 1=red 2=blue（0=dead 不收割：仿真无死亡，无法验证）

只收模型标了红/蓝的装甲（dead 不收），且车必须匹配上 GT（<2m）。
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
DEFAULT_SEQ = _ROOT / "scene" / "locate_out" / "videoseq_lamp"
DEFAULT_OUT = _ROOT / "dataset" / "armor_sim_yolo"
MATCH_M = 2.0
MIN_ARMOR_EDGE_PX = 6  # ROI 内的装甲至少 6px，太小无信息量
MIN_CROP_EDGE_PX = 24  # ROI 至少 24px，太小 YOLO 学不到东西


def _gt_team(name: str) -> str:
    return "red" if "Red" in name else "blue"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=Path, default=DEFAULT_SEQ)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", default="sim")
    parser.add_argument("--max-per-class", type=int, default=600)
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

    counts = {"red": 0, "blue": 0}
    for frame_path in sorted(args.seq.glob("*.jpg")):
        meta_path = frame_path.with_suffix(".json")
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(meta.get("camera", "Camera_Radar_Blue")) != "Camera_Radar_Blue":
            continue
        gts = [(_gt_team(r["name"]), r["x"], r["y"]) for r in meta.get("robots", [])]
        if not gts:
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        result = detector.infer(frame, persist_tracks=False)
        for robot_i, robot in enumerate(result.robots):
            car = robot.car
            if car.box.width < MIN_CROP_EDGE_PX or car.box.height < MIN_CROP_EDGE_PX:
                continue
            pixel = foot_pixel(car.box)
            hit = mesh.pixel_to_field(pose, pixel)
            if hit is None:
                continue
            best = min(gts, key=lambda g: (g[1] - hit.x) ** 2 + (g[2] - hit.y) ** 2)
            if float(np.hypot(best[1] - hit.x, best[2] - hit.y)) > MATCH_M:
                continue
            gt_team = best[0]
            if counts[gt_team] >= args.max_per_class:
                continue
            has_valid = False
            lines: list[str] = []
            crop_h = car.box.height
            crop_w = car.box.width
            for armor in robot.armors:
                if armor.label not in ("red", "blue"):
                    continue  # dead 不收：仿真无死亡验证不了
                # 全图装甲框 → ROI 局部坐标（remap_box 的逆）
                lx1 = armor.box.x1 - car.box.x1
                ly1 = armor.box.y1 - car.box.y1
                lx2 = armor.box.x2 - car.box.x1
                ly2 = armor.box.y2 - car.box.y1
                bw = lx2 - lx1
                bh = ly2 - ly1
                if bw < MIN_ARMOR_EDGE_PX or bh < MIN_ARMOR_EDGE_PX:
                    continue
                if lx1 < 0 or ly1 < 0 or lx2 > crop_w or ly2 > crop_h:
                    continue  # 越界框不要
                cls = 1 if gt_team == "red" else 2
                cx = (lx1 + lx2) / 2.0 / crop_w
                cy = (ly1 + ly2) / 2.0 / crop_h
                nw = bw / crop_w
                nh = bh / crop_h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                has_valid = True
            if not has_valid:
                continue
            stem = f"{args.tag}_{frame_path.stem}_r{robot_i:02d}"
            crop = frame[car.box.y1 : car.box.y2, car.box.x1 : car.box.x2]
            if crop.size == 0:
                continue
            if cv2.imwrite(str(img_dir / f"{stem}.jpg"), crop):
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                counts[gt_team] += 1
    print(f"red: {counts['red']}  blue: {counts['blue']}  total: {sum(counts.values())}")
    print(f"-> {img_dir}")


if __name__ == "__main__":
    main()
