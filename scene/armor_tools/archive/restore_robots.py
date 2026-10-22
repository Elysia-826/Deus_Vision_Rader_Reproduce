"""Unhide every robot in the blend and save. Does not move them."""

from __future__ import annotations

import bpy


def unhide_tree(ob: bpy.types.Object) -> None:
    ob.hide_viewport = False
    ob.hide_render = False
    ob.hide_set(False)
    for child in ob.children:
        unhide_tree(child)


def main() -> None:
    n = 0
    for ob in bpy.data.objects:
        if ob.name.startswith("Robot_") and (ob.parent is None or not ob.parent.name.startswith("Robot_")):
            unhide_tree(ob)
            n += 1
    col = bpy.data.collections.get("Robots")
    if col is not None:
        col.hide_viewport = False
        col.hide_render = False
    bpy.ops.wm.save_mainfile()
    print("unhid", n, "robot roots; saved", bpy.data.filepath)


if __name__ == "__main__":
    main()
