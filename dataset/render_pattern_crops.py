# ─── How to run ───
#     python dataset/render_pattern_crops.py
#     python dataset/render_pattern_crops.py --seq scene/locate_out/videoseq_spin --tag spin
# ──────────────────

"""从渲染序列收割真标签装甲裁剪，喂给图案分类器再训练。

和推理完全同域：同一条 检测器 → 全图装甲框 → frame[y1:y2, x1:x2] 裁剪链。
标签来自仿真真值（车底边射线定位后 2m 内最近 GT），不信旧分类器。
背景：旧模型把仿真 1/2/3 系统性高置信读成 S（conf≈0.94，91 次/200 帧），
根因是训练集里仿真数字裁剪只有贴图抖动合成、每类 40 张。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
DEFAULT_SEQ = _ROOT / "scene" / "locate_out" / "videoseq"
OUT = _ROOT / "dataset" / "pattern_balanced" / "train"
MATCH_M = 2.0
MIN_ARMOR_EDGE_PX = 12  # 小于这个的裁剪在 64px 训练分辨率下没有信息量
ImageU8 = np.ndarray  # 别名仅为可读性


def _gt_label(name: str, team: str) -> str:
    low = name.lower()
    role = "S" if "sentry" in low else "1" if "hero" in low else "2" if "engineer" in low else "3"
    return ("R" if team == "red" else "B") + role


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=Path, default=DEFAULT_SEQ, help="带同名 .json 真值的序列目录")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="render", help="产出文件名前缀（区分不同序列）")
    parser.add_argument("--max-per-class", type=int, default=300)
    parser.add_argument("--pose", type=Path, default=POSE_BLUE, help="相机外参 JSON（换新场地/新雷达位置后指向新文件）")
    parser.add_argument("--mesh", type=Path, default=MESH, help="场地网格 npz（换新场地后指向新文件）")
    args = parser.parse_args()

    pose = camera_from_path(args.pose)
    mesh = FieldMesh.from_npz(args.mesh)
    net, transform, device = vis2._load_classifier()
    detector = vis2._load_detector(vis2.PatternStage(net=net, transform=transform, device=str(device)))
    detector.infer(zeros((3000, 4096, 3), dtype=uint8), persist_tracks=False)  # 预热

    files = sorted(args.seq.glob("*.jpg"))
    if not files:
        raise SystemExit(f"no frames in {args.seq}")
    written: Counter[str] = Counter()
    for frame_path in files:
        meta_path = frame_path.with_suffix(".json")
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(meta.get("camera", "Camera_Radar_Blue")) != "Camera_Radar_Blue":
            continue  # 收割只走蓝雷达，和 POSE_BLUE 一致
        gts = [
            (_gt_label(r["name"], r["team"]), r["x"], r["y"])
            for r in meta.get("robots", [])
        ]
        if not gts:
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        result = detector.infer(frame, persist_tracks=False)
        for robot in result.robots:
            pixel = foot_pixel(robot.car.box)
            hit = mesh.pixel_to_field(pose, pixel)
            if hit is None:
                continue
            best = min(gts, key=lambda g: (g[1] - hit.x) ** 2 + (g[2] - hit.y) ** 2)
            dist = float(np.hypot(best[1] - hit.x, best[2] - hit.y))
            if dist > MATCH_M:
                continue
            folder = best[0]  # R1/R2/R3/B1/B2/B3（仿真没有 S/Q/4）
            if written[folder] >= args.max_per_class:
                continue
            for armor_i, armor in enumerate(robot.armors):
                box = armor.box
                if box.width < MIN_ARMOR_EDGE_PX or box.height < MIN_ARMOR_EDGE_PX:
                    continue
                crop = frame[box.y1 : box.y2, box.x1 : box.x2]
                if crop.size == 0:
                    continue
                dest = args.out / folder
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / f"{args.tag}_{frame_path.stem}_{armor_i:02d}.jpg"
                if cv2.imwrite(str(path), crop):
                    written[folder] += 1
    for folder, count in sorted(written.items()):
        print(f"{folder}: {count}")
    print(f"total {sum(written.values())} crops -> {args.out}")


if __name__ == "__main__":
    main()
