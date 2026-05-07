"""
Blender entrypoint: render a trajectory of top-down PNGs from _traj_params.json.

  blender -b --python scripts/render_trajectory_blender.py -- /path/to/_traj_params.json

Imports each asset once (bake_mesh_pose=False), then per frame updates pose and renders.
"""
from __future__ import annotations

import json
import math
import os
import sys
import traceback


def _argv_after_double_dash() -> list[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def _write_error(out_dir: str, message: str) -> None:
    path = os.path.join(out_dir, "_traj_error.json")
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ok": False, "error": message}, f, indent=2)
    except OSError:
        pass


def main() -> None:
    rest = _argv_after_double_dash()
    if not rest:
        print("Usage: blender -b --python .../render_trajectory_blender.py -- /path/to/_traj_params.json", file=sys.stderr)
        sys.exit(1)
    params_path = rest[0]

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
    except Exception as e:
        print(f"Failed to read params: {e}", file=sys.stderr)
        sys.exit(1)

    out_dir = params.get("out_dir") or os.path.dirname(params_path)
    os.makedirs(out_dir, exist_ok=True)

    try:
        import bpy
        import numpy as np

        from utils.blender_render import (
            _compute_room_from_task,
            _import_placed_asset_object,
            _setup_floor_mesh,
        )
        from utils.blender_utils import (
            load_hdri,
            reset_blender,
            set_rendering_settings,
            setup_background,
            setup_camera,
        )

        task = params["task"]
        frames = params["frames"]
        high_res = bool(params.get("high_res", False))
        fov_multiplier = float(params.get("fov_multiplier", 1.1))
        add_hdri = bool(params.get("add_hdri", True))
        floor_material = params.get("floor_material", "Travertine008")
        rotate_90 = bool(params.get("rotate_90", True))
        recenter_mesh = bool(params.get("recenter_mesh", True))
        apply_3dfront_texture = bool(params.get("apply_3dfront_texture", False))
        combine_obj_components = bool(params.get("combine_obj_components", False))

        reset_blender()
        setup_background()
        room = _compute_room_from_task(task)
        _setup_floor_mesh(room, floor_material=floor_material, adjust_top_down_angle=None)

        floor_vertices = room["floor_vertices"]
        floor_center_x = room["floor_center_x"]
        floor_center_y = room["floor_center_y"]
        floor_width = room["floor_width"]
        wall_height = room["wall_height"]

        all_keys: set[str] = set()
        for fr in frames:
            all_keys.update((fr.get("placed_assets") or {}).keys())

        inst_objects: dict[str, object] = {}
        for layout_key in sorted(all_keys):
            if layout_key not in task.get("assets", {}):
                print(f"[render_trajectory_blender] skip unknown asset key: {layout_key}", flush=True)
                continue
            asset = task["assets"][layout_key]
            obj = _import_placed_asset_object(
                layout_key,
                asset,
                {},
                rotate_90=rotate_90,
                recenter_mesh=recenter_mesh,
                apply_3dfront_texture=apply_3dfront_texture,
                combine_obj_components=combine_obj_components,
                bake_mesh_pose=False,
            )
            obj.hide_render = True
            obj.hide_viewport = True
            inst_objects[layout_key] = obj

        if add_hdri:
            load_hdri()
        set_rendering_settings(high_res=high_res)
        # Trajectory GIFs are stitched from PNGs; RGBA + film_transparent causes viewers to
        # composite transparent pixels over the previous frame (ghosting). Force opaque RGB.
        bpy.context.scene.render.film_transparent = False
        bpy.context.scene.render.image_settings.color_mode = "RGB"

        setup_camera(
            floor_center_x,
            floor_center_y,
            floor_width,
            wall_height,
            fov_multiplier=fov_multiplier,
            use_damped_track=False,
        )

        from collections import defaultdict

        group_frames: dict[int, list[str]] = defaultdict(list)

        for fr in frames:
            g = int(fr["group_idx"])
            fi = int(fr["frame_idx"])
            placed = fr.get("placed_assets") or {}

            for o in inst_objects.values():
                o.hide_render = True
                o.hide_viewport = True

            for kid, pdata in placed.items():
                if kid not in inst_objects:
                    continue
                o = inst_objects[kid]
                o.hide_render = False
                o.hide_viewport = False
                pos = list(pdata["position"])
                if len(pos) == 2:
                    meta = task["assets"][kid].get("assetMetadata", {}).get("boundingBox", {})
                    z_half = float(meta.get("z", 0.5)) / 2.0
                    pos = [float(pos[0]), float(pos[1]), z_half]
                o.location = pos
                rot = pdata.get("rotation", [0.0, 0.0, 0.0])
                if isinstance(rot, (int, float)):
                    o.rotation_euler = (0.0, 0.0, math.radians(float(rot)))
                else:
                    r = list(rot)
                    while len(r) < 3:
                        r.append(0.0)
                    o.rotation_euler = (
                        math.radians(float(r[0])),
                        math.radians(float(r[1])),
                        math.radians(float(r[2])),
                    )

            frame_dir = os.path.join(out_dir, f"group_{g}", "frames")
            os.makedirs(frame_dir, exist_ok=True)
            fp = os.path.join(frame_dir, f"frame_{fi:04d}.png")
            bpy.context.scene.render.filepath = fp
            bpy.ops.render.render(write_still=True)
            group_frames[g].append(fp)

        meta_groups = [
            {"group_idx": int(g), "frames": group_frames[g]}
            for g in sorted(group_frames.keys())
        ]
        meta_path = os.path.join(out_dir, "_traj_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"ok": True, "groups": meta_groups}, f, indent=2)
        print(f"render_trajectory_blender: wrote {meta_path}", flush=True)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _write_error(out_dir, msg)
        print(msg, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
