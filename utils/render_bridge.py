import json
import os
import subprocess
import tempfile
from typing import Dict, Tuple

import numpy as np


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _deserialize_coordinate_marks(serialized_marks) -> Dict[Tuple[int, int], Tuple[float, float]]:
    visual_marks = {}
    for item in serialized_marks:
        coord = item["coord"]
        pixel = item["pixel"]
        visual_marks[(int(coord[0]), int(coord[1]))] = (float(pixel[0]), float(pixel[1]))
    return visual_marks


def _run_blender_render(render_kwargs):
    blender_binary = os.environ.get("BLENDER_BIN", "blender")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    worker_script = os.path.join(repo_root, "utils", "blender_subprocess_worker.py")

    with tempfile.TemporaryDirectory(prefix="layoutvlm_blender_") as tmp_dir:
        payload_path = os.path.join(tmp_dir, "payload.json")
        output_path = os.path.join(tmp_dir, "output.json")
        with open(payload_path, "w") as payload_file:
            json.dump({"render_kwargs": render_kwargs}, payload_file, default=_json_default)

        env = os.environ.copy()
        python_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = repo_root if not python_path else f"{repo_root}{os.pathsep}{python_path}"
        command = [
            blender_binary,
            "--background",
            "--python",
            worker_script,
            "--",
            "--payload",
            payload_path,
            "--output",
            output_path,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Unable to find Blender executable. Install Blender and set BLENDER_BIN if needed."
            ) from exc

        if completed.returncode != 0:
            raise RuntimeError(
                "Blender 渲染进程失败。\n"
                f"命令: {' '.join(command)}\n"
                f"返回码: {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        with open(output_path, "r") as output_file:
            return json.load(output_file)


def render_existing_scene(placed_assets, task, save_dir, add_hdri=True, topdown_save_file=None, sideview_save_file=None, add_coordinate_mark=True,
                          annotate_object=True, annotate_wall=True, render_top_down=True, adjust_top_down_angle=None, high_res=False, rotate_90=True,
                          apply_3dfront_texture=False, recenter_mesh=True, fov_multiplier=1.1, default_font_size=None,
                          combine_obj_components=False, side_view_phi=45, side_view_indices=[3], save_blend=False,
                          add_object_bbox=False, ignore_asset_instance_idx=False, floor_material="Travertine008"):
    from utils.plot_utils import annotate_image_with_coordinates

    os.makedirs(save_dir, exist_ok=True)
    render_kwargs = {
        "placed_assets": placed_assets,
        "task": task,
        "save_dir": save_dir,
        "add_hdri": add_hdri,
        "topdown_save_file": topdown_save_file,
        "sideview_save_file": sideview_save_file,
        "add_coordinate_mark": add_coordinate_mark,
        "annotate_object": annotate_object,
        "annotate_wall": annotate_wall,
        "render_top_down": render_top_down,
        "adjust_top_down_angle": adjust_top_down_angle,
        "high_res": high_res,
        "rotate_90": rotate_90,
        "apply_3dfront_texture": apply_3dfront_texture,
        "recenter_mesh": recenter_mesh,
        "fov_multiplier": fov_multiplier,
        "default_font_size": default_font_size,
        "combine_obj_components": combine_obj_components,
        "side_view_phi": side_view_phi,
        "side_view_indices": side_view_indices,
        "save_blend": save_blend,
        "add_object_bbox": add_object_bbox,
        "ignore_asset_instance_idx": ignore_asset_instance_idx,
        "floor_material": floor_material,
        "annotate_in_blender": False,
        "return_annotation_payload": True,
    }
    result = _run_blender_render(render_kwargs)

    annotation_payload = result["annotation_payload"]
    topdown_render_path = result.get("topdown_render_path")
    if topdown_render_path and os.path.exists(topdown_render_path):
        topdown_coordinate_marks = _deserialize_coordinate_marks(annotation_payload["topdown_coordinate_marks"])
        if len(topdown_coordinate_marks) > 0:
            annotate_image_with_coordinates(
                image_path=topdown_render_path,
                visual_marks=topdown_coordinate_marks,
                output_path=topdown_render_path,
                format="coordinate",
            )

        topdown_text_font_size = annotation_payload["topdown_text_font_size"]
        if topdown_text_font_size is not None:
            topdown_text_marks = annotation_payload["topdown_text_marks"]
            annotate_image_with_coordinates(
                image_path=topdown_render_path,
                visual_marks=topdown_text_marks,
                output_path=topdown_render_path,
                format="text",
                default_font_size=topdown_text_font_size,
            )

    for image_path, serialized_marks in annotation_payload["side_coordinate_marks"].items():
        side_visual_marks = _deserialize_coordinate_marks(serialized_marks)
        annotate_image_with_coordinates(
            image_path=image_path,
            visual_marks=side_visual_marks,
            output_path=image_path,
            format="coordinate",
        )

    final_visual_marks = _deserialize_coordinate_marks(result["final_visual_marks"])
    return result["output_images"], final_visual_marks
