"""Export non-robot field triangles for mesh raycast. Does not save the blend.

    blender.exe scene/RMUC2026_V2.0.0_radar.blend --background --python scene/armor_assets/export_field_mesh.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "locate" / "calib" / "field_mesh.npz"
SKIP_PREFIXES = ("Robot_", "Camera_", "Radar_Coverage", "VisibilityOverlay", "Proto_")


def skip(name: str) -> bool:
    return name.startswith(SKIP_PREFIXES) or "Proto" in name


def main() -> None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam = bpy.data.objects.get("Camera_Radar_Blue")
    if cam is not None:
        euler = cam.rotation_euler
        print(
            "blue cam lens",
            round(cam.data.lens, 3),
            "euler_deg",
            round(np.degrees(euler.x), 2),
            round(np.degrees(euler.y), 2),
            round(np.degrees(euler.z), 2),
            "loc",
            tuple(round(v, 3) for v in cam.matrix_world.to_translation()),
            flush=True,
        )
    chunks_v: list[np.ndarray] = []
    chunks_f: list[np.ndarray] = []
    offset = 0
    for ob in bpy.data.objects:
        if ob.type != "MESH" or skip(ob.name):
            continue
        eval_ob = ob.evaluated_get(depsgraph)
        mesh = eval_ob.to_mesh()
        try:
            mesh.calc_loop_triangles()
            if not mesh.loop_triangles:
                continue
            mw = eval_ob.matrix_world
            local = np.array([vert.co[:] for vert in mesh.vertices], dtype=np.float32)
            world = np.array([(mw @ Vector(row)).to_tuple() for row in local], dtype=np.float32)
            faces = np.array([tri.vertices[:] for tri in mesh.loop_triangles], dtype=np.int32)
            # drop triangles whose centroid is a robot-height outlier or far outside the field
            centroids = world[faces].mean(axis=1)
            keep = (
                (np.abs(centroids[:, 0]) < 16.0)
                & (np.abs(centroids[:, 1]) < 9.0)
                & (centroids[:, 2] > -0.2)
                & (centroids[:, 2] < 2.2)
            )
            faces = faces[keep]
            if len(faces) == 0:
                continue
            used = np.unique(faces)
            remap = np.full(len(world), -1, dtype=np.int32)
            remap[used] = np.arange(len(used), dtype=np.int32)
            chunks_v.append(world[used])
            chunks_f.append(remap[faces] + offset)
            offset += len(used)
            print(ob.name, "tris", len(faces), "verts", len(used), flush=True)
        finally:
            eval_ob.to_mesh_clear()
    if not chunks_v:
        raise SystemExit("no field triangles")
    vertices = np.concatenate(chunks_v, axis=0)
    faces = np.concatenate(chunks_f, axis=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, vertices=vertices, faces=faces)
    print("wrote", OUT, "verts", len(vertices), "tris", len(faces), flush=True)


if __name__ == "__main__":
    main()
