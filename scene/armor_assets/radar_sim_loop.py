"""Blender live video source. --motion off|patrol  --camera Camera_Radar_Blue

GUI（持续画面，不要 --background）：
    blender.exe scene/RMUC2026_V2.0.0_radar.blend --python scene/armor_assets/radar_sim_loop.py -- --motion patrol
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scene" / "locate_out"
LIVE_JPG = OUT_DIR / "live_frame.jpg"
LIVE_TMP = OUT_DIR / "live_frame.tmp.jpg"
LIVE_META = OUT_DIR / "live_meta.json"
LIVE_ERR = OUT_DIR / "sim_error.log"
SPEED_MPS = 1.5
MIN_SEP = 1.45
CHEST = 0.4
PATHS: tuple[tuple[tuple[float, float], ...], ...] = (
    ((-8.0, -3.0), (-8.0, 3.0), (4.0, 3.0), (4.0, -3.0)),
    ((-2.0, 2.0), (5.0, 2.0), (5.0, -2.0), (-2.0, -2.0)),
    ((6.0, -1.0), (6.0, 2.0), (-4.0, 2.0), (-4.0, -1.0)),
    ((8.0, 3.0), (8.0, -3.0), (-4.0, -3.0), (-4.0, 3.0)),
    ((2.0, -2.0), (-5.0, -2.0), (-5.0, 2.0), (2.0, 2.0)),
    ((-6.0, 1.0), (-6.0, -2.0), (5.0, -2.0), (5.0, 1.0)),
)


def _argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_opts() -> tuple[str, str, int]:
    motion, camera, max_frames = "off", "Camera_Radar_Blue", 0
    args, i = _argv(), 0
    while i < len(args):
        match args[i]:
            case "--motion":
                motion = args[i + 1]
                i += 2
            case "--camera":
                camera = args[i + 1]
                i += 2
            case "--max-frames":
                max_frames = int(args[i + 1])
                i += 2
            case _:
                i += 1
    if motion not in {"off", "patrol"}:
        raise RuntimeError(f"bad --motion {motion}")
    return motion, camera, max_frames


def point_on_loop(waypoints: tuple[tuple[float, float], ...], t: float) -> tuple[float, float]:
    total = 0.0
    segs: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    n = len(waypoints)
    for i, start in enumerate(waypoints):
        end = waypoints[(i + 1) % n]
        length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        segs.append((start, end, length))
        total += length
    dist = (SPEED_MPS * t) % total
    walked = 0.0
    for start, end, length in segs:
        if dist <= walked + length or length <= 0.0:
            frac = 0.0 if length <= 0.0 else (dist - walked) / length
            return start[0] + frac * (end[0] - start[0]), start[1] + frac * (end[1] - start[1])
        walked += length
    return waypoints[0]


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


def is_robot_mesh(ob: bpy.types.Object | None) -> bool:
    cur = ob
    while cur is not None:
        if cur.name.startswith("Robot_"):
            return True
        cur = cur.parent
    return False


def unhide_tree(ob: bpy.types.Object) -> None:
    ob.hide_viewport = False
    ob.hide_render = False
    ob.hide_set(False)
    for child in ob.children:
        unhide_tree(child)


def set_tree_render(ob: bpy.types.Object, hide: bool) -> None:
    ob.hide_render = hide
    ob.hide_viewport = False
    ob.hide_set(False)
    for child in ob.children:
        set_tree_render(child, hide)


def set_world_xyz(ob: bpy.types.Object, x: float, y: float, z: float) -> None:
    world = ob.matrix_world.copy()
    world.translation = Vector((x, y, z))
    ob.matrix_world = world


def ground_z(x: float, y: float) -> float | None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    origin = Vector((x, y, 30.0))
    down = Vector((0.0, 0.0, -1.0))
    for _ in range(8):
        hit, loc, _n, _i, obj, _m = bpy.context.scene.ray_cast(depsgraph, origin, down)
        if not hit:
            return None
        if is_robot_mesh(obj):
            origin = loc + Vector((0.0, 0.0, -0.03))
            continue
        return float(loc.z)
    return None


def wall_blocks(x0: float, y0: float, x1: float, y1: float, z: float) -> bool:
    delta = Vector((x1 - x0, y1 - y0, 0.0))
    dist = float(delta.length)
    if dist < 1e-4:
        return False
    direction = delta.normalized()
    origin = Vector((x0, y0, z + CHEST))
    remain = dist + 0.2
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for _ in range(8):
        hit, loc, _n, _i, obj, _m = bpy.context.scene.ray_cast(
            depsgraph, origin, direction, distance=remain
        )
        if not hit:
            return False
        if is_robot_mesh(obj):
            origin = loc + direction * 0.08
            remain -= 0.08
            if remain <= 0.0:
                return False
            continue
        return True
    return False


def setup(camera_name: str) -> list[bpy.types.Object]:
    scene = bpy.context.scene
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 75
    eevee = getattr(scene, "eevee", None)
    if eevee is not None:
        if hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = 1
        if hasattr(eevee, "taa_samples"):
            eevee.taa_samples = 1
    scene.camera = bpy.data.objects[camera_name]
    col = bpy.data.collections.get("radar_viz")
    if col is not None:
        col.hide_render = True
    for name in ("VisibilityOverlay_Blue", "VisibilityOverlay_Red"):
        ob = bpy.data.objects.get(name)
        if ob is not None:
            ob.hide_render = True
    robots = root_robots()
    for ob in robots:
        unhide_tree(ob)
    red = [ob for ob in robots if team_of(ob.name) == "red"]
    blue = [ob for ob in robots if team_of(ob.name) == "blue"]
    chosen = red[:3] + blue[:3]
    chosen_set = set(chosen)
    for ob in robots:
        set_tree_render(ob, hide=ob not in chosen_set)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return chosen


def place(chosen: list[bpy.types.Object], motion: str, t: float) -> list[dict[str, str | float]]:
    placed: list[tuple[float, float]] = []
    for i, ob in enumerate(chosen):
        old = ob.matrix_world.to_translation()
        want_x, want_y = point_on_loop(PATHS[i], t) if motion == "patrol" else PATHS[i][0]
        jump = (want_x - old.x) ** 2 + (want_y - old.y) ** 2 > 4.0
        if not jump and wall_blocks(old.x, old.y, want_x, want_y, old.z):
            want_x, want_y = old.x, old.y
        too_close = False
        for px, py in placed:
            if (want_x - px) ** 2 + (want_y - py) ** 2 < MIN_SEP**2:
                too_close = True
                break
        if too_close:
            want_x, want_y = old.x, old.y
        z = ground_z(want_x, want_y)
        if z is None:
            want_x, want_y, z = old.x, old.y, old.z
        set_world_xyz(ob, want_x, want_y, z)
        placed.append((want_x, want_y))
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    records: list[dict[str, str | float]] = []
    for ob in chosen:
        world = ob.evaluated_get(depsgraph).matrix_world.to_translation()
        records.append(
            {
                "name": ob.name,
                "team": team_of(ob.name),
                "x": float(world.x),
                "y": float(world.y),
                "z": float(world.z),
            }
        )
    return records


def _eevee_still() -> None:
    window = bpy.context.window
    scene = bpy.context.scene
    if window is None:
        bpy.ops.render.render(write_still=True)
        return
    with bpy.context.temp_override(window=window, scene=scene):
        bpy.ops.render.render(write_still=True)


def _publish_live() -> None:
    if not LIVE_TMP.is_file():
        return
    last: OSError | None = None
    for _ in range(10):
        try:
            LIVE_TMP.replace(LIVE_JPG)
            return
        except OSError as exc:
            last = exc
            time.sleep(0.02)
    raise RuntimeError(f"publish live frame failed: {last}") from last


def render_live(frame_id: int, camera_name: str, robots: list[dict[str, str | float]]) -> None:
    scene = bpy.context.scene
    scene.render.filepath = str(OUT_DIR / "live_frame.tmp")
    _eevee_still()
    _publish_live()
    LIVE_META.write_text(
        json.dumps({"frame": frame_id, "camera": camera_name, "image": str(LIVE_JPG), "robots": robots}),
        encoding="utf-8",
    )


def _redraw() -> None:
    screen = getattr(bpy.context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        area.tag_redraw()


class RADAR_OT_sim(bpy.types.Operator):
    bl_idname = "wm.radar_sim"
    bl_label = "Radar sim loop"

    def execute(self, context: bpy.types.Context) -> set[str]:
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.04, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        if event.type in {"ESC"}:
            context.window_manager.event_timer_remove(self._timer)
            return {"CANCELLED"}
        if event.type == "TIMER":
            try:
                done = _tick() is None
            except Exception:
                LIVE_ERR.write_text(traceback.format_exc(), encoding="utf-8")
                print(traceback.format_exc(), flush=True)
                context.window_manager.event_timer_remove(self._timer)
                return {"CANCELLED"}
            if done:
                context.window_manager.event_timer_remove(self._timer)
                return {"FINISHED"}
            _redraw()
        return {"PASS_THROUGH"}


_MOTION = "off"
_CAMERA = "Camera_Radar_Blue"
_MAX_FRAMES = 0
_T0 = 0.0
_FRAME = 0
_CHOSEN: list[bpy.types.Object] = []


def _tick() -> float | None:
    global _FRAME
    records = place(_CHOSEN, _MOTION, time.time() - _T0)
    render_live(_FRAME, _CAMERA, records)
    print("frame", _FRAME, flush=True)
    _FRAME += 1
    if _MAX_FRAMES > 0 and _FRAME >= _MAX_FRAMES:
        return None
    return 0.05


def _kick_operator() -> float | None:
    if bpy.context.window is None:
        return 0.25
    bpy.ops.wm.radar_sim()
    return None


def main() -> None:
    global _MOTION, _CAMERA, _MAX_FRAMES, _T0, _FRAME, _CHOSEN
    motion, camera_name, max_frames = parse_opts()
    _MOTION, _CAMERA, _MAX_FRAMES = motion, camera_name, max_frames
    _T0 = time.time()
    _FRAME = 0
    _CHOSEN = setup(camera_name)
    print("sim loop", "motion", motion, "camera", camera_name, "bg", bpy.app.background, flush=True)
    if bpy.app.background:
        try:
            while _tick() is not None:
                pass
        except Exception:
            LIVE_ERR.write_text(traceback.format_exc(), encoding="utf-8")
            raise
        return
    bpy.utils.register_class(RADAR_OT_sim)
    bpy.app.timers.register(_kick_operator, first_interval=0.4)


if __name__ == "__main__":
    main()

