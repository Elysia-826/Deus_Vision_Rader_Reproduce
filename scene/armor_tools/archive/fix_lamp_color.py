"""灯条颜色规格化：纯净发射色 + 适配 Standard 视图变换的强度。

Standard 视图变换逐通道截断（爆曝保色，与真实相机一致），所以强度
可以推高而颜色不丢。但发射色的非主导通道（0.05 的绿色泄漏）在高
强度下也会截断可见，把蓝灯推成青色、红灯推成橙色——先清零。

蓝 [0, 0.01, 1.0] × 6 → 截断 → [0, 0.06, 1.0] 深蓝
红 [1.0, 0.01, 0.01] × 6 → 截断 → [1.0, 0.06, 0.06] 深红
灯条亮度 ~6× 白装甲面，YOLO 能明确抓住。

用法：
    blender --background scene/RMUC2027_V1.0.0_radar_fixed.blend \
        --python scene/armor_assets/fix_lamp_color.py -- [--strength 6]
"""

from __future__ import annotations

import sys

LAMP_STRENGTH = 6.0
# 材质名后缀 → (主导通道发射色)
EMISSION_BY_MAT: dict[str, tuple[float, float, float]] = {}


def argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def main() -> None:
    import bpy

    strength = LAMP_STRENGTH
    args = argv_after_double_dash()
    if "--strength" in args:
        strength = float(args[args.index("--strength") + 1])

    changed_color = changed_strength = 0
    for mat in bpy.data.materials:
        if not mat.name.startswith("ArmorMat_") or mat.node_tree is None:
            continue
        is_blue = "_blue_" in mat.name
        # 纯净发射色：非主导通道接近零，避免高强度下截断成杂色。
        pure = (0.0, 0.01, 1.0) if is_blue else (1.0, 0.01, 0.01)
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                for socket_name in ("Emission Color", "Emission"):
                    try:
                        sock = node.inputs[socket_name]
                    except LookupError:
                        continue
                    if not sock.is_linked:
                        sock.default_value = (*pure, 1.0)
                        changed_color += 1
                    break
            if node.type == "MATH" and node.operation == "MULTIPLY":
                drives_emission = any(
                    link.to_node.type == "BSDF_PRINCIPLED"
                    and "Emission Strength" in link.to_socket.name
                    for link in mat.node_tree.links
                    if link.from_node == node
                )
                if drives_emission:
                    node.inputs[1].default_value = strength
                    changed_strength += 1
    bpy.ops.wm.save_mainfile()
    print(f"emission colors -> {changed_color}, strength -> {strength} on {changed_strength} nodes")


main()
