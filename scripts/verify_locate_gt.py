# ─── How to run ───
#     python scripts/verify_locate_gt.py
# ──────────────────

"""GT (x,y,0) → projectPoints → 平面射线，应回到同一点。不跑检测。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from locate.camera import camera_from_path
from locate.ray_plane import pixel_to_ground, world_to_pixel
from locate.types import WorldXYZ

CAMERA_JSON = _ROOT / "locate" / "calib" / "Camera_Radar_Blue.json"
GT_JSON = _ROOT / "scene" / "locate_out" / "gt.json"


def main() -> None:
    pose = camera_from_path(CAMERA_JSON)
    payload = json.loads(GT_JSON.read_text(encoding="utf-8"))
    print(f"C={pose.center.tolist()}  {pose.width}x{pose.height}")
    errors: list[float] = []
    for robot in payload["robots"]:
        world = WorldXYZ(float(robot["x"]), float(robot["y"]), 0.0)
        pixel = world_to_pixel(pose, world)
        hit = pixel_to_ground(pose, pixel)
        if hit is None:
            print(f"{robot['name']:<24} uv=({pixel.u:.1f},{pixel.v:.1f}) MISS")
            continue
        err = ((hit.x - world.x) ** 2 + (hit.y - world.y) ** 2) ** 0.5
        errors.append(err)
        print(f"{robot['name']:<24} uv=({pixel.u:7.1f},{pixel.v:7.1f})  xy=({hit.x:7.3f},{hit.y:7.3f})  err={err:.4f}m")
    if errors:
        print(f"max {max(errors):.4f}m  median {sorted(errors)[len(errors)//2]:.4f}m")


if __name__ == "__main__":
    main()
