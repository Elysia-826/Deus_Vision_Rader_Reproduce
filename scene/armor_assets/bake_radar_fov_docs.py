"""Bake 1.1-inch radar coverage for 6/8/12 mm into docs/. Does not save the blend."""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Euler, Vector

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SENSOR_W = 14.1312
LENSES = (6.0, 8.0, 12.0)
RESTORE_MM = 8.0
OVERLAY_W = 29.2
OVERLAY_H = 15.6
BAKE_X = 876
BAKE_Y = 468
CAMS = ("Camera_Radar_Blue", "Camera_Radar_Red")


def h_fov_rad(lens_mm: float, sensor_w: float) -> float:
    return 2.0 * math.atan(sensor_w / (2.0 * lens_mm))


def _snapshot_hide() -> dict[str, bool]:
    return {ob.name: ob.hide_render for ob in bpy.data.objects}


def _restore_hide(state: dict[str, bool]) -> None:
    for ob in bpy.data.objects:
        if ob.name in state:
            ob.hide_render = state[ob.name]


def _hide_others_for_bake() -> None:
    keep = {"Radar_Coverage_Spot", "VisibilityBakePlane", "VisibilityBakeCam"}
    for ob in bpy.data.objects:
        ob.hide_render = ob.name not in keep


def _set_lens(lens_mm: float) -> float:
    fov = h_fov_rad(lens_mm, SENSOR_W)
    for name in CAMS:
        cam = bpy.data.objects[name].data
        cam.lens = lens_mm
        cam.sensor_width = SENSOR_W
    bpy.data.objects["Radar_Coverage_Spot"].data.spot_size = fov
    return fov


def bake_one(lens_mm: float) -> None:
    fov = _set_lens(lens_mm)
    print("lens", lens_mm, "fov_h_deg", round(math.degrees(fov), 2))
    scene = bpy.context.scene
    old = (
        scene.render.engine,
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.resolution_percentage,
        scene.render.filepath,
        scene.camera,
        scene.render.film_transparent,
        scene.render.image_settings.file_format,
    )
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = False
    scene.render.resolution_x = BAKE_X
    scene.render.resolution_y = BAKE_Y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    view = scene.view_settings
    old_view = (view.view_transform, view.look, view.exposure)
    view.view_transform = "Standard"
    view.look = "None"
    view.exposure = 0.0

    if "VisibilityBakePlane" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["VisibilityBakePlane"], do_unlink=True)
    if "VisibilityBakeCam" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["VisibilityBakeCam"], do_unlink=True)

    mesh = bpy.data.meshes.new("VisibilityBakePlaneMesh")
    plane = bpy.data.objects.new("VisibilityBakePlane", mesh)
    bpy.context.scene.collection.objects.link(plane)
    hw, hh = OVERLAY_W / 2.0, OVERLAY_H / 2.0
    mesh.from_pydata(
        [(-hw, -hh, 0.0), (hw, -hh, 0.0), (hw, hh, 0.0), (-hw, hh, 0.0)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()
    mat = bpy.data.materials.new("VisibilityBakeMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.08, 0.08, 0.08, 1.0)
        bsdf.inputs["Roughness"].default_value = 1.0
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 0.0
    plane.data.materials.append(mat)

    cam_data = bpy.data.cameras.new("VisibilityBakeCamData")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = OVERLAY_W
    cam_ob = bpy.data.objects.new("VisibilityBakeCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_ob)
    cam_ob.location = Vector((0.0, 0.0, 40.0))
    cam_ob.rotation_euler = Euler((0.0, 0.0, 0.0))
    scene.camera = cam_ob

    hide_state = _snapshot_hide()
    spot = bpy.data.objects["Radar_Coverage_Spot"]
    old_parent = spot.parent
    old_energy = spot.data.energy
    spot.data.energy = max(old_energy, 800.0)
    _hide_others_for_bake()
    plane.hide_render = False
    cam_ob.hide_render = False
    spot.hide_render = False

    for cam_name in CAMS:
        side = "blue" if "Blue" in cam_name else "red"
        path = DOCS / f"radar_fov_1.1inch_{int(lens_mm)}mm_{side}.png"
        cam = bpy.data.objects[cam_name]
        cam.hide_render = False
        world = cam.matrix_world.copy()
        spot.parent = None
        spot.matrix_world = world
        bpy.context.view_layer.update()
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        print("baked", path)

    spot.parent = old_parent
    spot.location = Vector((0.0, 0.0, 0.0))
    spot.rotation_euler = Euler((0.0, 0.0, 0.0))
    spot.data.energy = old_energy
    view.view_transform = old_view[0]
    view.look = old_view[1]
    view.exposure = old_view[2]
    _restore_hide(hide_state)
    for name in ("VisibilityOverlay_Blue", "VisibilityOverlay_Red"):
        ob = bpy.data.objects.get(name)
        if ob is not None:
            ob.hide_set(True)
            ob.hide_render = True

    bpy.data.objects.remove(plane, do_unlink=True)
    bpy.data.objects.remove(cam_ob, do_unlink=True)
    bpy.data.meshes.remove(mesh)
    bpy.data.materials.remove(mat)
    bpy.data.cameras.remove(cam_data)

    scene.render.engine = old[0]
    scene.render.resolution_x = old[1]
    scene.render.resolution_y = old[2]
    scene.render.resolution_percentage = old[3]
    scene.render.filepath = old[4]
    scene.camera = old[5]
    scene.render.film_transparent = old[6]
    scene.render.image_settings.file_format = old[7]


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    for lens_mm in LENSES:
        bake_one(lens_mm)
    restore = _set_lens(RESTORE_MM)
    print("restored lens", RESTORE_MM, "fov_h_deg", round(math.degrees(restore), 2), "no blend save")


if __name__ == "__main__":
    main()
