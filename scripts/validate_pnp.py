# ─── How to run ───
#     python scripts/validate_pnp.py
# 无需图像、无需点击：用仿真蓝雷达的真值 K/R/t 把地标投影成"完美点击"，
# 解 PnP，对照真值外参报告旋转/平移/残差，以及不同点击噪声下的
# 端到端定位误差（射线打 z=0 地面）。回答"上场 3 分钟手点能点出什么精度"。
# ──────────────────

"""PnP 标定精度验证（仿真真值基准）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from locate.camera import camera_from_path
from locate.pnp import solve_pose
from locate.ray_plane import pixel_to_ground
from locate.types import Pixel, WorldXYZ

CAMERA_JSON = _ROOT / "locate" / "calib" / "Camera_Radar_Blue.json"

# RM2027 场景固定建筑（蓝雷达画面内、非共线、三层高度差、y 轴 ±7m 展开）。
LANDMARKS: tuple[WorldXYZ, ...] = (
    WorldXYZ(x=5.5, y=0.0, z=1.2),     # 蓝前哨站顶（中场高度）
    WorldXYZ(x=-5.5, y=0.0, z=1.2),    # 红前哨站顶（远场高度）
    WorldXYZ(x=0.0, y=0.0, z=1.5),     # 能量机关中心（最高点）
    WorldXYZ(x=0.0, y=7.0, z=0.5),     # 左机库地面（图像右侧、y 展开）
    WorldXYZ(x=0.0, y=-7.0, z=0.5),    # 右机库地面（图像左侧、y 展开）
    WorldXYZ(x=-11.0, y=-5.0, z=0.5),  # 红方堡垒（远端角 + 侧偏）
)
NOISE_PX: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0)


def _rotation_error_deg(a, b) -> float:
    diff = a @ b.T
    angle = np.clip((np.trace(diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(angle)))


def _end_to_end_m(gt, solved, samples: int = 400) -> tuple[float, float]:
    """随机地面点经真值位姿成像素，再用解出的位姿射线打回地面，量误差。"""
    rng = np.random.default_rng(42)
    errors: list[float] = []
    for _ in range(samples):
        point = WorldXYZ(x=float(rng.uniform(-12.0, 12.0)), y=float(rng.uniform(-6.5, 6.5)), z=0.0)
        pixel = gt.project(point)
        hit = pixel_to_ground(solved, pixel)
        if hit is None:
            continue
        errors.append(float(np.hypot(hit.x - point.x, hit.y - point.y)))
    errors.sort()
    median = errors[len(errors) // 2]
    p90 = errors[int(len(errors) * 0.9)]
    return median, p90


def main() -> None:
    gt = camera_from_path(CAMERA_JSON)
    perfect = [gt.project(point) for point in LANDMARKS]
    for point, pixel in zip(LANDMARKS, perfect, strict=True):
        in_frame = 0 <= pixel.u <= gt.width and 0 <= pixel.v <= gt.height
        print(f"landmark ({point.x:+7.2f},{point.y:+6.2f},{point.z:+5.2f}) -> ({pixel.u:7.1f},{pixel.v:7.1f}) {'ok' if in_frame else 'OUT OF FRAME'}")
    pairs = list(zip(LANDMARKS, perfect, strict=True))

    rng = np.random.default_rng(7)
    print()
    print(f"{'noise_px':>9} {'rot_err':>8} {'t_err_cm':>9} {'rms_px':>7} {'e2e_med_m':>9} {'e2e_p90_m':>9}")
    for sigma in NOISE_PX:
        clicked = [
            Pixel(u=p.u + float(rng.normal(0.0, sigma)), v=p.v + float(rng.normal(0.0, sigma)))
            for p in perfect
        ]
        result = solve_pose(
            name="pnp_validation",
            width=gt.width,
            height=gt.height,
            K=gt.K,
            dist=gt.dist,
            pairs=[(world, pixel) for world, pixel in zip(LANDMARKS, clicked, strict=True)],
        )
        solved = result.pose
        rot_err = _rotation_error_deg(solved.R, gt.R)
        t_err = float(np.linalg.norm(solved.t - gt.t)) * 100.0
        median, p90 = _end_to_end_m(gt, solved)
        print(
            f"{sigma:>9.1f} {rot_err:>7.3f}° {t_err:>8.1f} {result.rms_px:>7.2f} "
            f"{median:>9.3f} {p90:>9.3f}"
        )
    print()
    print("准确线 0.8m / 半准确 1.6m：看 e2e_p90 与哪一档相交，就是这套标定在真场上的等级。")


if __name__ == "__main__":
    main()
