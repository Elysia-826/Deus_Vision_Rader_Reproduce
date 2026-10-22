# ─── How to run ───
#     python scripts/raycast_debug.py [frame.jpg]
#     python scripts/raycast_debug.py scene/locate_out/videoseq_2027_dwell/sim_0000.jpg
# 点像素 → 打场地 mesh + z=0 平面 → 显示交点世界坐标。q 退出。
# ──────────────────

"""交互式射线调试：验证标定和网格的单点快速检查。

在帧上点一个像素，射线同时打场地 mesh（3D）和 z=0 平面（2D），
把两个交点标在图上并打印世界坐标。mesh 交点 z≠0 说明打到了
台阶/围挡/建筑；z=0 交点是地面位置。

如果 mesh 交点和 z=0 交点差距大 → 标定或网格有问题。
如果同一个特征（如前哨站底部）点击的 world 坐标和已知地标一致 → 标定正确。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from locate.camera import camera_from_path  # noqa: E402
from locate.ray_mesh import FieldMesh  # noqa: E402
from locate.ray_plane import pixel_to_ground  # noqa: E402
from locate.types import Pixel  # noqa: E402

CAMERA_JSON = _ROOT / "locate" / "calib" / "Camera_Radar_Blue.json"
MESH_NPZ = _ROOT / "locate" / "calib" / "field_mesh.npz"
DEFAULT_FRAME = _ROOT / "scene" / "locate_out" / "videoseq_2027_dwell" / "sim_0000.jpg"


def main() -> None:
    frame_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FRAME
    if not frame_path.is_file():
        # 回退到测试渲染帧
        candidates = sorted((_ROOT / "scene" / "locate_out" / "videoseq_2027_dwell").glob("*.jpg"))
        if not candidates:
            raise SystemExit(f"frame not found: {frame_path}（也没有回退帧）")
        frame_path = candidates[0]

    pose = camera_from_path(CAMERA_JSON)
    mesh = FieldMesh.from_npz(MESH_NPZ)
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise SystemExit(f"cannot read: {frame_path}")
    print(f"frame: {frame_path.name}  {frame.shape[1]}x{frame.shape[0]}")
    print("点击查看射线交点。q 退出。")

    canvas = frame.copy()

    def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
        nonlocal canvas
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        pixel = Pixel(u=float(x), v=float(y))
        hit_3d = mesh.pixel_to_field(pose, pixel)
        hit_2d = pixel_to_ground(pose, pixel)

        canvas = frame.copy()
        cv2.circle(canvas, (x, y), 8, (0, 255, 255), -1)

        # z=0 平面交点（绿色）
        if hit_2d is not None:
            cv2.putText(
                canvas,
                f"z=0: ({hit_2d.x:.2f}, {hit_2d.y:.2f})",
                (x + 12, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            print(f"  z=0   → ({hit_2d.x:7.2f}, {hit_2d.y:7.2f}) m")
        else:
            print("  z=0   → ray missed ground plane (below horizon)")

        # mesh 交点（品红）
        if hit_3d is not None:
            cv2.putText(
                canvas,
                f"mesh: ({hit_3d.x:.2f}, {hit_3d.y:.2f})",
                (x + 12, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2,
            )
            print(f"  mesh  → ({hit_3d.x:7.2f}, {hit_3d.y:7.2f}) m")
        else:
            print("  mesh  → ray missed field mesh (outside field)")

        # 两个交点差距
        if hit_2d is not None and hit_3d is not None:
            gap = ((hit_2d.x - hit_3d.x) ** 2 + (hit_2d.y - hit_3d.y) ** 2) ** 0.5
            if gap > 0.5:
                print(f"  ⚠ gap  → {gap:.2f}m（mesh 打到了非地面结构）")
            else:
                print(f"  ✓ gap  → {gap:.3f}m")

        cv2.imshow("raycast_debug", canvas)

    cv2.imshow("raycast_debug", canvas)
    cv2.setMouseCallback("raycast_debug", on_mouse)
    while cv2.waitKey(20) & 0xFF not in {ord("q"), 27}:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
