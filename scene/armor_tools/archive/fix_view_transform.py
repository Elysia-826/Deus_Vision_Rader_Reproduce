"""把雷达渲染的视图变换从 AgX 切到 Standard。

AgX 把高光统一去饱和成白——装甲灯条因此失去颜色，装甲 YOLO 把
蓝车 71% 误读成红（真实相机是逐通道截断，爆曝保色，所以真实录像
灯条是饱和的；Standard + 适当强度复刻这个行为）。
配套：scene/armor_assets/fix_lamp_strength.py（强度 40 → 2.0）。

用法：
    blender --background scene/RMUC2027_V1.0.0_radar_fixed.blend \
        --python scene/armor_assets/fix_view_transform.py -- [--agx]
带 --agx 则切回 AgX（回滚用）。
"""

from __future__ import annotations

import sys


def argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def main() -> None:
    import bpy

    target = "Standard" if "--agx" not in argv_after_double_dash() else "AgX"
    bpy.context.scene.view_settings.view_transform = target
    bpy.ops.wm.save_mainfile()
    print(f"view_transform -> {target}")


main()
