"""
Blender entrypoint for intermediate scene renders (invoked by conda via subprocess).

Usage (from repo root, inside Blender's Python):
  blender -b --python scripts/render_intermediate_blender.py -- /path/to/_render_params.json

The JSON payload is written by `utils.blender_subprocess.render_via_blender_subprocess`.
"""
from __future__ import annotations

import json
import os
import sys
import traceback


def _argv_after_double_dash() -> list[str]:
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def _write_error(save_dir: str, message: str) -> None:
    path = os.path.join(save_dir, "_render_error.json")
    try:
        os.makedirs(save_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ok": False, "error": message}, f, indent=2)
    except OSError:
        pass


def _serialize_visual_marks(vm):
    """Match utils.blender_subprocess.serialize_visual_marks (avoid importing parent helpers before sys.path)."""
    if isinstance(vm, dict) and vm:
        first_key = next(iter(vm.keys()))
        if isinstance(first_key, tuple):
            return {
                "__visual_marks_format__": "coord_dict",
                "items": [[[float(a), float(b)], [float(px), float(py)]] for (a, b), (px, py) in vm.items()],
            }
    if isinstance(vm, list):
        out = []
        for item in vm:
            if isinstance(item, dict):
                row = {}
                for k, v in item.items():
                    if isinstance(v, tuple):
                        row[k] = list(v)
                    else:
                        row[k] = v
                out.append(row)
            else:
                out.append(item)
        return out
    return vm


def main() -> None:
    rest = _argv_after_double_dash()
    if not rest:
        print("render_intermediate_blender: missing path to _render_params.json after --", file=sys.stderr)
        sys.exit(2)
    params_path = os.path.abspath(rest[0])
    save_dir_guess = os.path.dirname(params_path)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        with open(params_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        _write_error(save_dir_guess, f"Failed to read params: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    save_dir = payload.get("save_dir") or save_dir_guess
    task = payload.get("task")
    placed_assets = payload.get("placed_assets")
    render_kwargs = payload.get("render_kwargs") or {}

    if not isinstance(task, dict) or not isinstance(placed_assets, dict):
        _write_error(save_dir, "Invalid payload: task and placed_assets must be dicts")
        sys.exit(1)

    os.makedirs(save_dir, exist_ok=True)

    try:
        from utils.annotation_serialize import serialize_annotation_for_json
        from utils.blender_render import render_existing_scene

        output_images, visual_marks, annotations = render_existing_scene(
            placed_assets=placed_assets,
            task=task,
            save_dir=save_dir,
            **render_kwargs,
        )
        result_path = os.path.join(save_dir, "_render_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ok": True,
                    "output_images": list(output_images),
                    "visual_marks": _serialize_visual_marks(visual_marks),
                    "annotations": [serialize_annotation_for_json(a) for a in annotations],
                },
                f,
                indent=2,
            )
        print("render_intermediate_blender: wrote", result_path, flush=True)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _write_error(save_dir, msg)
        print(msg, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
