"""Make radar visibility planes toggleable in the viewport and visible on the field."""

from __future__ import annotations

import bpy


OVERLAYS = (
    ("VisibilityOverlay_Blue", "VisibilityOverlay_Blue_Mat", "RadarVisibility_Blue"),
    ("VisibilityOverlay_Red", "VisibilityOverlay_Red_Mat", "RadarVisibility_Red"),
)


def _build_emission_alpha(mat: bpy.types.Material, image: bpy.types.Image) -> None:
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.location = (-500, 0)
    to_bw = tree.nodes.new("ShaderNodeRGBToBW")
    to_bw.location = (-220, -80)
    emit = tree.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 6.0
    emit.location = (-220, 80)
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (-220, -220)
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.location = (40, 0)
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    out.location = (280, 0)
    tree.links.new(tex.outputs["Color"], emit.inputs["Color"])
    tree.links.new(tex.outputs["Color"], to_bw.inputs["Color"])
    tree.links.new(to_bw.outputs["Val"], mix.inputs["Fac"])
    tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    tree.links.new(emit.outputs["Emission"], mix.inputs[2])
    tree.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    mat.blend_method = "BLEND"
    if hasattr(mat, "shadow_method"):
        mat.shadow_method = "NONE"
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = False
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"


def main() -> None:
    for ob_name, mat_name, img_name in OVERLAYS:
        ob = bpy.data.objects[ob_name]
        mat = bpy.data.materials[mat_name]
        img = bpy.data.images[img_name]
        _build_emission_alpha(mat, img)
        ob.location.z = 0.12
        ob.hide_viewport = False
        ob.hide_render = True
        ob.hide_select = False
        ob.display_type = "TEXTURED"
        ob.visible_shadow = False
        ob.hide_set(ob_name.endswith("Red"))
        print(ob_name, "z", ob.location.z, "hide_viewport", ob.hide_viewport, "hide_get", ob.hide_get())
    bpy.ops.wm.save_mainfile()
    print("saved", bpy.data.filepath)


if __name__ == "__main__":
    main()
