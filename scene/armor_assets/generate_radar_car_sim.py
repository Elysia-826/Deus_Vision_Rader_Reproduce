"""Render radar-view car crops with YOLO labels. Does not save the blend."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

CAMS = ("Camera_Radar_Blue", "Camera_Radar_Red")
X_MIN, X_MAX = -12.5, 12.5
Y_MIN, Y_MAX = -7.0, 7.0
MIN_DIST = 1.35
MIN_BOX = 0.008
# 和 radar_sim_loop / MV-CH120-60UC 一致。旧的 1920×1080 不是这台 8 mm 相机。
RES_X, RES_Y = 4096, 3000


def argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_cli() -> tuple[Path, int, int]:
    out = Path(r"C:\Users\YQS\Desktop\DEUS_VISION_RADER_TEST_reproduce\dataset\car_sim")
    count = 1000
    seed = 42
    args = argv_after_double_dash()
    i = 0
    while i < len(args):
        match args[i]:
            case "--out":
                out = Path(args[i + 1])
                i += 2
            case "--count":
                count = int(args[i + 1])
                i += 2
            case "--seed":
                seed = int(args[i + 1])
                i += 2
            case unreachable:
                raise SystemExit(f"unknown arg: {unreachable}")
    return out, count, seed


def root_robots() -> list[bpy.types.Object]:
    found = [
        ob
        for ob in bpy.data.objects
        if ob.name.startswith("Robot_") and (ob.parent is None or not ob.parent.name.startswith("Robot_"))
    ]
    if len(found) < 5:
        raise RuntimeError(f"need >=5 robots, found {len(found)}")
    return sorted(found, key=lambda ob: ob.name)


def team_of(name: str) -> str:
    if "_Red" in name:
        return "red"
    if "_Blue" in name:
        return "blue"
    raise RuntimeError(f"robot has no team: {name}")


def iter_meshes(ob: bpy.types.Object) -> list[bpy.types.Object]:
    out = [ob] if ob.type == "MESH" else []
    for child in ob.children:
        out.extend(iter_meshes(child))
    return out


def set_tree_render(ob: bpy.types.Object, hide: bool) -> None:
    ob.hide_render = hide
    for child in ob.children:
        set_tree_render(child, hide)


def yolo_box(ob: bpy.types.Object, cam: bpy.types.Object, scene: bpy.types.Scene) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for mesh_ob in iter_meshes(ob):
        for corner in mesh_ob.bound_box:
            world = mesh_ob.matrix_world @ Vector(corner)
            ndc = world_to_camera_view(scene, cam, world)
            if ndc.z <= 0.05:
                continue
            xs.append(ndc.x)
            ys.append(1.0 - ndc.y)
    if not xs:
        return None
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    width, height = x1 - x0, y1 - y0
    if width < MIN_BOX or height < MIN_BOX:
        return None
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0, width, height


def pick_subset(robots: list[bpy.types.Object], rng: random.Random) -> list[bpy.types.Object]:
    red = [ob for ob in robots if team_of(ob.name) == "red"]
    blue = [ob for ob in robots if team_of(ob.name) == "blue"]
    n = rng.randint(5, 10)
    chosen = [rng.choice(red), rng.choice(blue)]
    rest = [ob for ob in robots if ob not in chosen]
    rng.shuffle(rest)
    chosen.extend(rest[: n - 2])
    return chosen


def place_robots(chosen: list[bpy.types.Object], rng: random.Random) -> bool:
    placed: list[tuple[float, float]] = []
    for ob in chosen:
        ok = False
        for _ in range(80):
            x = rng.uniform(X_MIN, X_MAX)
            y = rng.uniform(Y_MIN, Y_MAX)
            if all((x - px) ** 2 + (y - py) ** 2 >= MIN_DIST**2 for px, py in placed):
                ob.location.x = x
                ob.location.y = y
                ob.rotation_euler.z = rng.uniform(-3.1416, 3.1416)
                placed.append((x, y))
                ok = True
                break
        if not ok:
            return False
    return True


def snapshot(robots: list[bpy.types.Object]) -> list[tuple[bpy.types.Object, Vector, Vector, bool]]:
    return [
        (ob, ob.location.copy(), ob.rotation_euler.copy(), ob.hide_render)
        for ob in robots
    ]


def restore(state: list[tuple[bpy.types.Object, Vector, Vector, bool]]) -> None:
    for ob, loc, rot, hide in state:
        ob.location = loc
        ob.rotation_euler = rot
        set_tree_render(ob, hide)


def prepare_scene() -> None:
    col = bpy.data.collections.get("radar_viz")
    if col is not None:
        col.hide_render = True
    scene = bpy.context.scene
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 75
    scene.render.film_transparent = False
    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        if hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = 1
        if hasattr(eevee, "taa_samples"):
            eevee.taa_samples = 1


def write_label(path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_one(index: int, out_dir: Path, robots: list[bpy.types.Object], rng: random.Random) -> bool:
    scene = bpy.context.scene
    cam_name = CAMS[index % 2]
    cam = bpy.data.objects[cam_name]
    scene.camera = cam
    chosen = pick_subset(robots, rng)
    if not place_robots(chosen, rng):
        return False
    chosen_set = set(chosen)
    for ob in robots:
        set_tree_render(ob, hide=ob not in chosen_set)
    bpy.context.view_layer.update()
    boxes: list[tuple[float, float, float, float]] = []
    teams: set[str] = set()
    for ob in chosen:
        box = yolo_box(ob, cam, scene)
        if box is None:
            continue
        boxes.append(box)
        teams.add(team_of(ob.name))
    if len(boxes) < 5 or len(boxes) > 10 or teams != {"red", "blue"}:
        return False
    stem = f"sim_{index:04d}"
    scene.render.filepath = str(out_dir / f"{stem}.jpg")
    bpy.ops.render.render(write_still=True)
    write_label(out_dir / f"{stem}.txt", boxes)
    return True


def main() -> None:
    out_dir, count, seed = parse_cli()
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_scene()
    robots = root_robots()
    state = snapshot(robots)
    rng = random.Random(seed)
    kept = 0
    attempts = 0
    try:
        while kept < count:
            attempts += 1
            if attempts > count * 20:
                raise RuntimeError(f"too many retries ({attempts}) after {kept} frames")
            if render_one(kept, out_dir, robots, rng):
                kept += 1
                if kept % 10 == 0 or kept == count:
                    print(f"rendered {kept}/{count} attempts={attempts}")
    finally:
        restore(state)
    print("done", kept, "frames in", out_dir)


if __name__ == "__main__":
    main()
