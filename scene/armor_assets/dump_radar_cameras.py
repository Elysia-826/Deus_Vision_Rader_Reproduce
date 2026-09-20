"""Dump OpenCV K/R/t for Camera_Radar_Blue/Red. Optional: render one GT frame.

Run inside Blender, same as generate_radar_car_sim.py:

    blender.exe scene/RMUC2026_V2.0.0_radar.blend --background --python scene/armor_assets/dump_radar_cameras.py -- --render
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
CALIB_DIR = ROOT / "locate" / "calib"
OUT_DIR = ROOT / "scene" / "locate_out"
CAMS = ("Camera_Radar_Blue", "Camera_Radar_Red")
BCAM_TO_CV = Matrix(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
PLACEMENTS: tuple[tuple[str, float, float], ...] = (
    ("red", -8.0, -3.0),
    ("red", -2.0, 2.0),
    ("red", 6.0, -1.0),
    ("blue", 8.0, 3.0),
    ("blue", 2.0, -2.0),
    ("blue", -6.0, 1.0),
)


def wants_render() -> bool:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return "--render" in argv


def opencv_pose(
    obj: bpy.types.Object,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], int, int, list[float]]:
    r_c2w = obj.matrix_world.to_3x3()
    center = obj.matrix_world.to_translation()
    rotation = BCAM_TO_CV @ r_c2w.transposed()
    tvec = -(rotation @ center)
    k, width, height = intrinsics(obj)
    r_list = [[float(rotation[i][j]) for j in range(3)] for i in range(3)]
    t_list = [[float(tvec.x)], [float(tvec.y)], [float(tvec.z)]]
    return k, r_list, t_list, width, height, [float(center.x), float(center.y), float(center.z)]


def intrinsics(obj: bpy.types.Object) -> tuple[list[list[float]], int, int]:
    cam = obj.data
    render = bpy.context.scene.render
    scale = render.resolution_percentage / 100.0
    width = int(render.resolution_x * scale)
    height = int(render.resolution_y * scale)
    fit = cam.sensor_fit
    use_horizontal = width >= height if fit == "AUTO" else fit != "VERTICAL"
    if use_horizontal:
        fx = cam.lens * width / cam.sensor_width
        span = float(width)
    else:
        fx = cam.lens * height / cam.sensor_height
        span = float(height)
    fy = fx
    cx = width / 2.0 + cam.shift_x * span
    cy = height / 2.0 - cam.shift_y * span
    k = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    return k, width, height


def set_render_size() -> None:
    scene = bpy.context.scene
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"


def dump_cameras() -> None:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    for name in CAMS:
        obj = bpy.data.objects[name]
        k, rotation, tvec, width, height, center = opencv_pose(obj)
        payload = {
            "name": name,
            "image_size": [width, height],
            "K": k,
            "R": rotation,
            "t": tvec,
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "C": center,
        }
        path = CALIB_DIR / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("wrote", path, "C", center)


def team_of(name: str) -> str:
    if "_Red" in name:
        return "red"
    if "_Blue" in name:
        return "blue"
    raise RuntimeError(f"robot has no team: {name}")


def root_robots() -> list[bpy.types.Object]:
    found = [
        ob
        for ob in bpy.data.objects
        if ob.name.startswith("Robot_") and (ob.parent is None or not ob.parent.name.startswith("Robot_"))
    ]
    return sorted(found, key=lambda ob: ob.name)


def set_tree_render(ob: bpy.types.Object, hide: bool) -> None:
    ob.hide_render = hide
    for child in ob.children:
        set_tree_render(child, hide)


def render_frame() -> None:
    scene = bpy.context.scene
    cam = bpy.data.objects["Camera_Radar_Blue"]
    scene.camera = cam
    col = bpy.data.collections.get("radar_viz")
    if col is not None:
        col.hide_render = True
    spot = bpy.data.objects.get("Radar_Coverage_Spot")
    if spot is not None:
        spot.hide_render = True
    robots = root_robots()
    red = [ob for ob in robots if team_of(ob.name) == "red"]
    blue = [ob for ob in robots if team_of(ob.name) == "blue"]
    if len(red) < 3 or len(blue) < 3:
        raise RuntimeError(f"need 3 red and 3 blue, got {len(red)}/{len(blue)}")
    chosen: list[bpy.types.Object] = []
    records: list[dict[str, str | float]] = []
    red_i = 0
    blue_i = 0
    for team, x, y in PLACEMENTS:
        match team:
            case "red":
                ob = red[red_i]
                red_i += 1
            case "blue":
                ob = blue[blue_i]
                blue_i += 1
            case unreachable:
                raise RuntimeError(unreachable)
        world = ob.matrix_world.copy()
        world.translation = Vector((x, y, world.translation.z))
        ob.matrix_world = world
        chosen.append(ob)
    chosen_set = set(chosen)
    for ob in robots:
        set_tree_render(ob, hide=ob not in chosen_set)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for ob, (team, x, y) in zip(chosen, PLACEMENTS, strict=True):
        world = ob.evaluated_get(depsgraph).matrix_world.to_translation()
        records.append(
            {
                "name": ob.name,
                "team": team,
                "x": float(world.x),
                "y": float(world.y),
                "z": float(world.z),
                "placed_x": x,
                "placed_y": y,
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = OUT_DIR / "locate_frame.png"
    scene.render.filepath = str(image_path)
    bpy.ops.render.render(write_still=True)
    gt_path = OUT_DIR / "gt.json"
    gt_path.write_text(
        json.dumps({"camera": "Camera_Radar_Blue", "image": str(image_path), "robots": records}, indent=2),
        encoding="utf-8",
    )
    print("wrote", image_path, gt_path)


def main() -> None:
    set_render_size()
    dump_cameras()
    if wants_render():
        render_frame()


if __name__ == "__main__":
    main()
