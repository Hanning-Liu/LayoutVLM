"""
Run Blender-based scene rendering in a separate Blender process (conda-safe).

LayoutVLM's solver runs in conda with PyTorch; `bpy` only exists inside Blender's
Python. When `LAYOUTVLM_BLENDER` points to a `blender` binary, we write a small
JSON payload and invoke `blender -b --python scripts/render_intermediate_blender.py`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_intermediate_blender.py"
PARAMS_NAME = "_render_params.json"
RESULT_NAME = "_render_result.json"
ERROR_NAME = "_render_error.json"


def default_blender_executable() -> str:
    return (
        os.environ.get("LAYOUTVLM_BLENDER", "").strip()
        or (shutil.which("blender") or "")
        or "/home/ubuntu/blender-4.3.0-linux-x64/blender"
    )


def is_blender_subprocess_available() -> bool:
    exe = default_blender_executable()
    return bool(exe) and os.path.isfile(exe) and os.access(exe, os.X_OK) and RENDER_SCRIPT.is_file()


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars / ndarray / tuple keys for JSON."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                sk = "__tuple_key__:" + ",".join(str(x) for x in k)
            else:
                sk = str(k)
            out[sk] = _json_safe(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    return str(obj)


def serialize_visual_marks(vm: Any) -> Any:
    """visual_marks is dict[(x,y)] -> (px,py) or list of dicts with tuple values."""
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
    return _json_safe(vm)


def deserialize_visual_marks(data: Any) -> Any:
    if isinstance(data, dict) and data.get("__visual_marks_format__") == "coord_dict":
        items = data.get("items") or []
        return {(float(a), float(b)): (float(px), float(py)) for (a, b), (px, py) in items}
    if isinstance(data, list):
        restored = []
        for item in data:
            if isinstance(item, dict):
                row = {}
                for k, v in item.items():
                    if k == "pixel" or k == "end_arrow_pixel":
                        if isinstance(v, list) and len(v) == 2:
                            row[k] = (float(v[0]), float(v[1]))
                        else:
                            row[k] = v
                    else:
                        row[k] = v
                restored.append(row)
            else:
                restored.append(item)
        return restored
    return data


def render_via_blender_subprocess(
    placed_assets: Mapping[str, Any],
    task: Dict[str, Any],
    save_dir: Union[str, Path],
    *,
    timeout_sec: Optional[float] = None,
    **render_kwargs: Any,
) -> Tuple[List[str], Any]:
    """
    Invoke Blender to run render_existing_scene; returns same tuple as in-process call.

    `task` must use absolute paths in assets[*]['path'] where applicable.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    blender_bin = default_blender_executable()
    if not blender_bin or not os.path.isfile(blender_bin):
        raise FileNotFoundError(f"LAYOUTVLM_BLENDER / blender not found: {blender_bin!r}")
    if not RENDER_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing render script: {RENDER_SCRIPT}")

    if timeout_sec is None:
        timeout_sec = float(os.environ.get("LAYOUTVLM_BLENDER_TIMEOUT_SEC", "600"))

    params_path = save_dir / PARAMS_NAME
    result_path = save_dir / RESULT_NAME
    error_path = save_dir / ERROR_NAME

    for p in (result_path, error_path):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    payload = {
        "task": _json_safe(task),
        "placed_assets": _json_safe(dict(placed_assets)),
        "save_dir": str(save_dir.resolve()),
        "render_kwargs": _json_safe(dict(render_kwargs)),
    }
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    cmd = [
        blender_bin,
        "-b",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        str(params_path.resolve()),
    ]
    print(f"[render-subprocess] {' '.join(cmd[:5])} ... {params_path.name}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        err_tail = (proc.stderr or "")[-4000:]
        if error_path.is_file():
            try:
                with open(error_path, "r", encoding="utf-8") as ef:
                    err_tail = ef.read()[-4000:] + "\n--- stderr ---\n" + err_tail
            except OSError:
                pass
        raise RuntimeError(
            f"Blender subprocess failed (exit {proc.returncode}). Last output:\n{err_tail}"
        )

    if not result_path.is_file():
        raise FileNotFoundError(f"Blender did not write {result_path}")

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    output_images = result.get("output_images") or []
    visual_marks = deserialize_visual_marks(result.get("visual_marks"))

    from utils.annotation_serialize import deserialize_annotation_from_json
    from utils.image_annotate import apply_annotations

    ann_raw = result.get("annotations") or []
    annotations = [deserialize_annotation_from_json(a) for a in ann_raw]
    try:
        apply_annotations(annotations)
    except ImportError as e:
        raise RuntimeError(
            "Could not apply render annotations in conda (missing Pillow?). "
            "Install pillow in the layoutvlm env: pip install pillow"
        ) from e

    return output_images, visual_marks
