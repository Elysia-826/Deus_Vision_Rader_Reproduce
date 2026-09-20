"""Set radar cameras to 12mm FOV, match coverage spot, rebake visibility overlays, save blend."""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Euler, Vector

ROOT = Path(__file__).resolve().parents[2]
LENS_MM = 12.0
OVERLAY_W = 29.2
OVERLAY_H = 15.6
BAKE_X = 365
BAKE_Y = 195
CAMS = ("Camera_Radar_Blue", "Camera_Radar_Red")
OUT = {
    "Camera_Radar_Blue": ROOT / "scene" / "materials" / "radar_visibility_blue.png",
    "Camera_Radar_Red": ROOT / "scene" / "materials" / "radar_visibility_red.png",
}
IMG_NAME = {
    "Camera_Radar_Blue": "RadarVisibility_Blue",
    "Camera_Radar_Red": "RadarVisibility_Red",
}


def h_fov_rad(lens_mm: float, sensor_w: float) -> float:
    return 2.0 * math.atan(sensor_w / (2.0 * lens_mm))


def set_cameras() -> float:
    fov = None
    for name in CAMS:
        ob = bpy.data.objects[name]
        cam = ob.data
        cam.lens = LENS_MM
        fov = h_fov_rad(cam.lens, cam.sensor_width)
        print(name, "lens", cam.lens, "sensor_w", cam.sensor_width, "fov_h_deg", round(math.degrees(fov), 2))
    spot = bpy.data.objects["Radar_Coverage_Spot"]
    spot.data.spot_size = fov if fov is not None else h_fov_rad(LENS_MM, 14.1312)
    print("spot_size_deg", round(math.degrees(spot.data.spot_size), 2), "color", list(spot.data.color))
    return spot.data.spot_size


def _snapshot_hide() -> dict[str, bool]:
    return {ob.name: ob.hide_render for ob in bpy.data.objects}


def _restore_hide(state: dict[str, bool]) -> None:
    for ob in bpy.data.objects:
        if ob.name in state:
            ob.hide_render = state[ob.name]


def _hide_others_for_bake() -> None:
    keep = {"Radar_Coverage_Spot", "VisibilityBakePlane", "VisibilityBakeCam"}
    for ob in bpy.data.objects:
        if ob.name in keep:
            ob.hide_render = False
        else:
            ob.hide_render = True


def bake_visibility() -> None:
    scene = bpy.context.scene
    old = (
        scene.render.engine,
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.resolution_percentage,
        scene.render.filepath,
        scene.camera,
        scene.render.film_transparent,
    )
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x = BAKE_X
    scene.render.resolution_y = BAKE_Y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"

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
        bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
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
    _hide_others_for_bake()
    plane.hide_render = False
    cam_ob.hide_render = False
    spot.hide_render = False

    for cam_name, path in OUT.items():
        cam = bpy.data.objects[cam_name]
        spot.parent = cam
        spot.location = Vector((0.0, 0.0, 0.0))
        spot.rotation_euler = Euler((0.0, 0.0, 0.0))
        bpy.context.view_layer.update()
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.get(IMG_NAME[cam_name])
        if img is not None:
            img.filepath = str(path)
            img.reload()
        print("baked", cam_name, path)

    spot.parent = old_parent
    spot.location = Vector((0.0, 0.0, 0.0))
    spot.rotation_euler = Euler((0.0, 0.0, 0.0))
    _restore_hide(hide_state)
    # overlays stay viewport-hidden as before
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


def main() -> None:
    set_cameras()
    bake_visibility()
    bpy.ops.wm.save_mainfile()
    print("saved", bpy.data.filepath)


if __name__ == "__main__":
    main()
