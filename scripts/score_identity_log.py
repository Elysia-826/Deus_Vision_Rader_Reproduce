# ─── How to run ───
#     python scripts/score_identity_log.py
# ──────────────────

"""看 identity_log.jsonl：线上点和真值的对照。GT 全部 6 台都记，但有些不在画面里，
先用蓝相机投影过滤可见 GT，再最近邻配对算误差。不跑检测，纯数数。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from locate.camera import camera_from_path
from locate.types import WorldXYZ

LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "scene" / "locate_out" / "identity_log.jsonl"
BLUE_JSON = _ROOT / "locate" / "calib" / "Camera_Radar_Blue.json"

IMG_W, IMG_H = 4096, 3000
MARGIN = 100
GATE_M = 2.5


def _visible(pose, x: float, y: float, z: float) -> bool:
    try:
        pixel = pose.project(WorldXYZ(x, y, z))
    except Exception:
        return False
    return -MARGIN <= pixel.u <= IMG_W + MARGIN and -MARGIN <= pixel.v <= IMG_H + MARGIN


def main() -> None:
    pose = camera_from_path(BLUE_JSON)
    lines = LOG.read_text(encoding="utf-8").splitlines()
    n_frames = 0
    n_gt = n_vis = 0
    n_pub = n_tent = n_conf = 0
    n_matched = 0
    errors: list[float] = []
    n_unmatched_pred = 0
    empty_frames = 0
    frame_ids: list[int] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        n_frames += 1
        frame_ids.append(int(raw.get("frame", -1)))
        gt = [
            g
            for g in raw.get("gt", [])
            if _visible(pose, float(g["x"]), float(g["y"]), float(g["z"]))
        ]
        preds = list(raw.get("pred", []))
        n_gt += len(raw.get("gt", []))
        n_vis += len(gt)
        n_pub += len(preds)
        if not preds:
            empty_frames += 1
        used = [False] * len(gt)
        for pred in preds:
            state = str(pred.get("state", ""))
            if state == "tentative":
                n_tent += 1
            elif state == "confirmed":
                n_conf += 1
            best_k, best_d = -1, GATE_M
            for k, g in enumerate(gt):
                if used[k]:
                    continue
                d = (
                    (float(pred["x"]) - float(g["x"])) ** 2
                    + (float(pred["y"]) - float(g["y"])) ** 2
                ) ** 0.5
                if d < best_d:
                    best_k, best_d = k, d
            if best_k < 0:
                n_unmatched_pred += 1
                continue
            used[best_k] = True
            n_matched += 1
            errors.append(best_d)
    ordered = sorted(errors)
    print(f"frames={n_frames} id_range=({min(frame_ids)},{max(frame_ids)})")
    print(f"gt_total={n_gt} gt_visible={n_vis} published={n_pub} (tentative={n_tent} confirmed={n_conf})")
    print(f"empty_frames={empty_frames} ({empty_frames / max(n_frames, 1) * 100:.1f}%)")
    print(f"matched_visible={n_matched} unmatched_pred={n_unmatched_pred}")
    if n_vis:
        print(f"dot_recall_vs_visible={n_matched / n_vis:.3f}")
    if ordered:
        print(
            f"pred_err median={ordered[len(ordered) // 2]:.3f}m max={ordered[-1]:.3f}m "
            f"within_1.2m={sum(1 for e in ordered if e <= 1.2)}/{len(ordered)}"
        )


if __name__ == "__main__":
    main()
