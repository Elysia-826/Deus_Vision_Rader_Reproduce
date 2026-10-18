"""把装甲灯条的自发光强度从爆表值降到保色区间。

诊断（2026-10）：灯条 Emission Color 本身是饱和红/蓝，但强度乘数 ~40
把辐射度推到 ~100（HDR），AgX/Filmic 色调映射把超亮高光去饱和成近白
——装甲 YOLO 因此把蓝车灯条 71% 误读成红色（真机不会有此问题）。
降到 2.0：蓝通道辐射度 2.5×2=5，AgX 保色；灯条仍是暗场里最亮的东西。

用法：
    blender --background scene/RMUC2027_V1.0.0_radar_fixed.blend \
        --python scene/armor_assets/fix_lamp_strength.py -- [--strength 2.0]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

LAMP_STRENGTH = 2.0
MATH_TYPES = {"MATH", "MULTIPLY"}


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

    blend = Path(bpy.data.filepath)
    backup = blend.with_suffix(".prelamp.bak")
    if not backup.exists():
        shutil.copy2(blend, backup)
        print(f"backup {backup.name}")

    changed = 0
    for mat in bpy.data.materials:
        if not mat.name.startswith("ArmorMat_") or mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "MATH" or node.operation != "MULTIPLY":
                continue
            # 只动连到「原理化 BSDF.Emission Strength」的那个乘法节点。
            drives_emission = any(
                link.to_node.type == "BSDF_PRINCIPLED"
                and "Emission Strength" in link.to_socket.name
                for link in mat.node_tree.links
                if link.from_node == node
            )
            if not drives_emission:
                continue
            node.inputs[1].default_value = strength
            changed += 1
    bpy.ops.wm.save_mainfile()
    print(f"lamp strength -> {strength} on {changed} multiply nodes")


main()
