# ─── How to run ───
#     python scripts/score_stage_funnel.py
# ──────────────────

"""直播分布帧上的分级检出：GT → 车 → 带装甲 → 图案可用 → 射线落点 → 对上 GT。

链路和 scripts/run_radar.py 完全一致（同 engine、同 PatternStage、同 mesh ray），
只是不进跟踪器，一帧一帧数，看点到底死在哪一级。GT 来自采样时存下的 live meta。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from detect import PatternStage  # noqa: E402
from detect.runtime import load_classifier, load_detector  # noqa: E402
from locate.camera import camera_from_path  # noqa: E402
from locate.ray_mesh import FieldMesh  # noqa: E402
from locate.ray_plane import foot_pixel  # noqa: E402
from locate.types import WorldXYZ  # noqa: E402
from track.observe import observations_from_result  # noqa: E402

SAMPLES = _ROOT / "scene" / "locate_out" / "samples"
BLUE_JSON = _ROOT / "locate" / "calib" / "Camera_Radar_Blue.json"
RED_JSON = _ROOT / "locate" / "calib" / "Camera_Radar_Red.json"
MESH_NPZ = _ROOT / "locate" / "calib" / "field_mesh.npz"

IMG_W, IMG_H = 4096, 3000
MARGIN = 100
BIRTH_CONF = 0.5
MATCH_M = 1.2


def _visible(pose, x: float, y: float, z: float) -> bool:
    try:
        pixel = pose.project(WorldXYZ(x, y, z))
    except Exception:
        return False
    return -MARGIN <= pixel.u <= IMG_W + MARGIN and -MARGIN <= pixel.v <= IMG_H + MARGIN


def main() -> None:
    poses = {
        "Camera_Radar_Blue": camera_from_path(BLUE_JSON),
        "Camera_Radar_Red": camera_from_path(RED_JSON),
    }
    pose = poses["Camera_Radar_Blue"]
    mesh = FieldMesh.from_npz(MESH_NPZ)
    net, transform, device = load_classifier()
    detector = load_detector(PatternStage(net=net, transform=transform, device=str(device)))
    print("detector loaded", flush=True)

    frames = sorted(SAMPLES.glob("live_*.jpg"))
    t_gt = t_vis = 0
    t_car = t_car_hit = 0
    t_armor = 0
    lamp_counts: dict[str, int] = {}
    t_pat = 0
    t_obs = t_obs_hit = 0
    for image_path in frames:
        meta = json.loads((SAMPLES / f"{image_path.stem}.json").read_text(encoding="utf-8-sig"))
        pose = poses.get(str(meta.get("camera", "Camera_Radar_Blue")), poses["Camera_Radar_Blue"])
        gt = [
            g
            for g in meta.get("robots", [])
            if _visible(pose, float(g["x"]), float(g["y"]), float(g["z"]))
        ]
        t_gt += len(meta.get("robots", []))
        t_vis += len(gt)
        image = cv2.imread(str(image_path))
        # persist_tracks=False：只要纯检测。True 会把 BoT-SORT 状态带到下一帧，
        # 而采样帧之间机器人跳了好几米（slideshow），跟踪器会被晃晕，数出来的是
        # 跟踪事故不是检测能力。线上 run_radar 用的默认 True，这本身就是嫌疑人。
        result = detector.infer(image, persist_tracks=False)
        used_gt: set[int] = set()
        for robot in result.robots:
            t_car += 1
            box = robot.car.box
            pixel = foot_pixel(box)
            hit = mesh.pixel_to_field(pose, pixel)
            if hit is None:
                continue
            bk, bd = -1, MATCH_M
            for k, g in enumerate(gt):
                if k in used_gt:
                    continue
                d = ((hit.x - float(g["x"])) ** 2 + (hit.y - float(g["y"])) ** 2) ** 0.5
                if d < bd:
                    bk, bd = k, d
            if bk >= 0:
                used_gt.add(bk)
                t_car_hit += 1
            if robot.armors:
                t_armor += 1
                for armor in robot.armors:
                    lamp_counts[armor.label] = lamp_counts.get(armor.label, 0) + 1
        obs = observations_from_result(result, pose, None, mesh)
        t_obs += len(obs)
        # birth 门限和 cascade._pair_cost 完全一致：AGREE（team+role 齐）且 conf≥0.5。
        # 这是观测自带的值，不是仿版。
        t_pat += sum(
            1
            for ob in obs
            if ob.team is not None and ob.role is not None and ob.pattern_conf >= BIRTH_CONF
        )
        used2: set[int] = set()
        for ob in obs:
            bk, bd = -1, MATCH_M
            for k, g in enumerate(gt):
                if k in used2:
                    continue
                d = ((ob.x - float(g["x"])) ** 2 + (ob.y - float(g["y"])) ** 2) ** 0.5
                if d < bd:
                    bk, bd = k, d
            if bk >= 0:
                used2.add(bk)
                t_obs_hit += 1
        print(
            f"{image_path.name}: gt={len(gt)} cars={len(result.robots)} "
            f"obs={len(obs)} hits={len(used2)}",
            flush=True,
        )
    print(f"frames={len(frames)} gt={t_gt} visible={t_vis}")
    print(f"car_det={t_car} car_ray_hit_gt={t_car_hit}")
    print(f"with_armor={t_armor} lamp_split={lamp_counts} pattern_birth_eligible={t_pat}")
    print(f"obs_onfield={t_obs} obs_hit_gt={t_obs_hit}")
    if t_vis:
        print(f"car_recall={t_car_hit / t_vis:.3f} obs_recall={t_obs_hit / t_vis:.3f}")


if __name__ == "__main__":
    main()
