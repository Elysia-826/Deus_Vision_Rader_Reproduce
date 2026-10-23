# ─── How to run ───
#     python scripts/diagnose.py --mode conf      # 图案分类器：conf 真值对照
#     python scripts/diagnose.py --mode lamp     # 装甲 YOLO：红蓝颜色误读率
#     python scripts/diagnose.py --mode confusion # 槽→GT 混淆表（谁在吃谁）
#     python scripts/diagnose.py --mode digits    # 1↔3 数字混淆矩阵
#     python scripts/diagnose.py --seq scene/locate_out/videoseq_dwell --mode conf
# ──────────────────

"""诊断工具：跑完整检测链后按模式输出量化报告。

模式：
  conf      图案分类器每次预测 vs 真值（角色/队伍/置信度/闪读方向）
  lamp      装甲 YOLO 红蓝 dead 标签 vs 真值队伍（颜色误读率 + 置信度分离）
  confusion 每个 0x0305 槽实际在跟踪哪台 GT 车（吃车/被吃关系）
  digits    图案模型在收割 crop 上的 1↔3 混淆矩阵（尺寸 vs 错误率）
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

from detect import PatternStage  # noqa: E402
from detect.runtime import load_classifier, load_detector  # noqa: E402
from locate.camera import camera_from_path  # noqa: E402
from locate.ray_mesh import FieldMesh  # noqa: E402
from track.observe import foot_pixel, observations_from_result  # noqa: E402

POSE_BLUE = _ROOT / "locate" / "calib" / "Camera_Radar_Blue.json"
MESH = _ROOT / "locate" / "calib" / "field_mesh.npz"
DEFAULT_SEQ = _ROOT / "scene" / "locate_out" / "videoseq_dwell"
MATCH_M = 2.0


def _gt_label(name: str, team: str) -> str:
    low = name.lower()
    role = "S" if "sentry" in low else "1" if "hero" in low else "2" if "engineer" in low else "3"
    return ("R" if team == "red" else "B") + role


def _load_detector():
    pose = camera_from_path(POSE_BLUE)
    mesh = FieldMesh.from_npz(MESH)
    net, transform, device = load_classifier()
    detector = load_detector(PatternStage(net=net, transform=transform, device=str(device)))
    detector.infer(zeros((3000, 4096, 3), dtype=uint8), persist_tracks=False)
    return pose, mesh, detector


def _walk_frames(seq: Path):
    for fp in sorted(seq.glob("*.jpg")):
        meta_path = fp.with_suffix(".json")
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(meta.get("camera", "Camera_Radar_Blue")) != "Camera_Radar_Blue":
            continue
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        yield fp, frame, meta


def mode_conf(seq: Path) -> None:
    """图案分类器 conf-truth 对照。"""
    pose, mesh, detector = _load_detector()
    true_confs: list[float] = []
    wrong_confs: list[float] = []
    flick = Counter()
    flick_conf: dict[str, list[float]] = {}
    for _, frame, meta in _walk_frames(seq):
        gts = [(_gt_label(r["name"], r["team"]), r["x"], r["y"]) for r in meta.get("robots", [])]
        if not gts:
            continue
        result = detector.infer(frame, persist_tracks=False)
        for obs in observations_from_result(result, pose, None, mesh):
            if obs.team is None or obs.role is None:
                continue
            best = min(gts, key=lambda g: (g[1]-obs.x)**2 + (g[2]-obs.y)**2)
            d = float(np.hypot(best[1]-obs.x, best[2]-obs.y))
            if d > MATCH_M:
                continue
            ok = (obs.team.value == "red") == (best[0][0] == "R") and obs.role.value == best[0][1]
            (true_confs if ok else wrong_confs).append(obs.pattern_conf)
            if not ok:
                flick[(best[0][1], obs.role.value)] += 1
                flick_conf.setdefault(obs.role.value, []).append(obs.pattern_conf)

    total = len(true_confs) + len(wrong_confs)
    print(f"labeled observations: {total}")
    print(f"正确: n={len(true_confs)}  conf p50={np.median(true_confs):.2f}" if true_confs else "正确: 0")
    if wrong_confs:
        print(f"错误: n={len(wrong_confs)}  conf p50={np.median(wrong_confs):.2f}  p90={np.percentile(wrong_confs, 90):.2f}")
        print("闪读方向 (真角色→误读角色):")
        for (tr, pr), n in flick.most_common(8):
            cs = flick_conf.get(pr, [])
            p50 = f"{np.median(cs):.2f}" if cs else "-"
            print(f"  {tr}→{pr}  n={n}  conf p50={p50}")


def mode_lamp(seq: Path) -> None:
    """装甲 YOLO 红蓝 dead 颜色误读率。"""
    pose, mesh, detector = _load_detector()
    ok_confs: list[float] = []
    flip_confs: list[float] = []
    labels = Counter()
    for _, frame, meta in _walk_frames(seq):
        gts = [(("red" if "Red" in r["name"] else "blue"), r["x"], r["y"]) for r in meta.get("robots", [])]
        if not gts:
            continue
        result = detector.infer(frame, persist_tracks=False)
        for robot in result.robots:
            pixel = foot_pixel(robot.car.box)
            hit = mesh.pixel_to_field(pose, pixel)
            if hit is None:
                continue
            best = min(gts, key=lambda g: (g[1]-hit.x)**2 + (g[2]-hit.y)**2)
            if float(np.hypot(best[1]-hit.x, best[2]-hit.y)) > MATCH_M:
                continue
            for armor in robot.armors:
                labels[armor.label] += 1
                if armor.label in ("red", "blue"):
                    (ok_confs if armor.label == best[0] else flip_confs).append(armor.conf)

    print(f"装甲标签分布: {dict(labels)}")
    total = len(ok_confs) + len(flip_confs)
    if total:
        print(f"颜色正确: n={len(ok_confs)} ({len(ok_confs)/total:.0%})  conf p50={np.median(ok_confs):.2f}" if ok_confs else "颜色正确: 0")
        print(f"颜色翻转: n={len(flip_confs)} ({len(flip_confs)/total:.0%})  conf p50={np.median(flip_confs):.2f}" if flip_confs else "颜色翻转: 0")


def mode_confusion(seq: Path) -> None:
    """槽→GT 混淆表（从 identity_log.jsonl 读）。"""
    log = seq.parent / "identity_log.jsonl"
    if not log.is_file():
        raise SystemExit(f"identity log not found: {log}（先跑一次 RADAR_REFEREE_STUB=1 回放）")
    confusion = Counter()
    for line in log.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        gts = [(_gt_label(r["name"], r["team"]), r["x"], r["y"]) for r in rec.get("gt", [])]
        for p in rec.get("pred", []):
            if not gts:
                confusion[(p["slot"], "NO_GT")] += 1
                continue
            best = min(gts, key=lambda g: (g[1]-p["x"])**2 + (g[2]-p["y"])**2)
            d = float(np.hypot(best[1]-p["x"], best[2]-p["y"]))
            key = best[0] if d < MATCH_M else "far>2m"
            confusion[(p["slot"], key)] += 1
    print("pred → nearest GT (<2m) 混淆表:")
    for (slot, gt), n in sorted(confusion.items()):
        print(f"  {slot:>4} ← {gt:<8} {n:>4}")


def mode_digits(seq: Path) -> None:
    """1↔3 数字混淆矩阵（在收割 crop 上直接跑 PyTorch 模型）。"""
    import torch
    from PIL import Image
    from models.pattern.model import MobileNetV3SmallClassifier
    from models.pattern.transforms import val_transform

    ckpt = torch.load(_ROOT / "weights/pattern_best.pth", map_location="cpu", weights_only=False)
    net = MobileNetV3SmallClassifier(pretrained=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    transform = val_transform(64)
    CLASS_NAMES = ("1", "2", "3", "4", "S", "Q")

    confusion = Counter()
    for folder in ("R1", "B1", "R3", "B3"):
        true_role = folder[-1]
        for pattern in ("rlamp_*.jpg", "sslamp_*.jpg", "dwell_*.jpg"):
            for f in (_ROOT / "dataset/pattern_balanced/train" / folder).glob(pattern):
                img = cv2.imread(str(f))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = transform(Image.fromarray(rgb)).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(net(tensor), dim=1)
                pred = CLASS_NAMES[int(probs.argmax())]
                conf = float(probs[0, int(probs.argmax())])
                confusion[(true_role, pred)] += 1

    print("1↔3 混淆矩阵（收割 crop）:")
    for (tr, pr), n in sorted(confusion.items()):
        mark = "✓" if tr == pr else "✗"
        print(f"  {tr}→{pr} {mark}  n={n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("conf", "lamp", "confusion", "digits"))
    parser.add_argument("--seq", type=Path, default=DEFAULT_SEQ)
    args = parser.parse_args()
    match args.mode:
        case "conf":
            mode_conf(args.seq)
        case "lamp":
            mode_lamp(args.seq)
        case "confusion":
            mode_confusion(args.seq)
        case "digits":
            mode_digits(args.seq)


if __name__ == "__main__":
    main()
