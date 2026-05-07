#!/usr/bin/env python3
"""
Build a 3D trajectory GIF aligned with ``final.gif`` (same frame cadence as ``saved_intermediate_states.json``).

1. Collect ``group_*/temp_*/saved_intermediate_states.json`` under ``--results_dir``.
2. Write ``_traj_params.json`` and invoke Blender once (``render_trajectory_blender.py``).
3. Stitch ``out_blender.gif`` per group and ``final_blender.gif`` (conda: Pillow / imageio).

Example::

  conda run -n layoutvlm python scripts/render_trajectory.py \\
    --scene_json_file out/bench_balcony/balcony_001.json \\
    --results_dir results/out_batch/bench_balcony/balcony_001 \\
    --asset_dir ./objaverse_processed \\
    --out_dir results/out_batch/bench_balcony/balcony_001/trajectory_blender \\
    --blender_bin /path/to/blender
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAJ_PARAMS = "_traj_params.json"
TRAJ_META = "_traj_meta.json"
TRAJ_ERROR = "_traj_error.json"
BLENDER_SCRIPT = REPO_ROOT / "scripts" / "render_trajectory_blender.py"


def _load_prepare_task_assets():
    spec = importlib.util.spec_from_file_location(
        "render_result_blender",
        REPO_ROOT / "scripts" / "render_result_blender.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.prepare_task_assets  # type: ignore[attr-defined]


def default_blender_executable(explicit: str | None) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    return (
        os.environ.get("LAYOUTVLM_BLENDER", "").strip()
        or (shutil.which("blender") or "")
        or ""
    )


def grad_key_to_layout_key(k: str) -> str | None:
    """``ladder_back_chair_0`` -> ``ladder_back_chair-0``; walls skipped."""
    if k.startswith("walls"):
        return None
    parts = k.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return f"{parts[0]}-{parts[1]}"


def _collect_saved_jsons(results_dir: str) -> List[Tuple[int, str]]:
    """Sorted (group_idx, path_to_saved_intermediate_states.json). One file per group (prefer lexicographically first temp_*)."""
    by_group: Dict[int, List[str]] = {}
    pattern = os.path.join(results_dir, "group_*", "temp_*", "saved_intermediate_states.json")
    for p in glob.glob(pattern):
        m = re.search(r"group_(\d+)", p)
        if not m:
            continue
        g = int(m.group(1))
        by_group.setdefault(g, []).append(p)
    out: List[Tuple[int, str]] = []
    for g in sorted(by_group.keys()):
        paths = sorted(by_group[g])
        out.append((g, paths[0]))
    return out


def _build_frames_from_saved(
    task: Dict[str, Any],
    saved_entries: List[Tuple[int, str]],
) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    for group_idx, json_path in saved_entries:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        snaps = data.get("solver_assets") or []
        for frame_idx, snap in enumerate(snaps):
            placed: Dict[str, Any] = {}
            for grad_k, v in snap.items():
                lk = grad_key_to_layout_key(grad_k)
                if lk is None:
                    continue
                if lk not in task.get("assets", {}):
                    print(f"[render_trajectory] skip unknown layout key {lk!r} (from {grad_k!r})", flush=True)
                    continue
                pos = v.get("position") or [0, 0, 0]
                theta = float(v.get("rotation", 0.0))
                deg_z = math.degrees(theta)
                placed[lk] = {
                    "position": [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0],
                    "rotation": [0.0, 0.0, deg_z],
                }
            frames.append(
                {
                    "group_idx": group_idx,
                    "frame_idx": frame_idx,
                    "placed_assets": placed,
                }
            )
    return frames


def _numpy_rgba_to_rgb(arr: Any) -> Any:
    """Flatten RGBA/LA to opaque RGB (white background) for GIF-safe frames."""
    import numpy as np

    a = np.asarray(arr)
    if a.ndim != 3:
        return a
    c = a.shape[2]
    if c == 4:
        alpha = a[:, :, 3:4].astype(np.float32) / 255.0
        rgb = a[:, :, :3].astype(np.float32)
        bg = np.full_like(rgb, 255.0)
        out = (rgb * alpha + bg * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        return out
    if c == 3:
        return a.astype(np.uint8, copy=False) if a.dtype != np.uint8 else a
    return a


def _pil_to_rgb(im: Any) -> Any:
    from PIL import Image

    if im.mode == "P":
        if "transparency" in im.info:
            im = im.convert("RGBA")
        else:
            return im.convert("RGB")
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        if im.mode == "LA":
            bg.paste(im.convert("RGBA"), mask=im.split()[-1])
        else:
            bg.paste(im, mask=im.split()[3])
        return bg
    if im.mode != "RGB":
        return im.convert("RGB")
    return im


def _save_gif(frame_paths: List[str], out_path: str, duration_sec: float = 0.5) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio

        ims = [_numpy_rgba_to_rgb(imageio.imread(p)) for p in frame_paths]
        imageio.mimsave(out_path, ims, duration=duration_sec)
        return
    except ImportError:
        pass
    from PIL import Image

    ims = [_pil_to_rgb(Image.open(p)) for p in frame_paths]
    if not ims:
        return
    ims[0].save(
        out_path,
        save_all=True,
        append_images=ims[1:],
        duration=int(duration_sec * 1000),
        loop=0,
        disposal=2,
        optimize=False,
    )
    for im in ims:
        im.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene_json_file", required=True)
    p.add_argument("--results_dir", required=True, help="Scene output dir containing group_*/temp_*/")
    p.add_argument("--asset_dir", default="./objaverse_processed")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--blender_bin", default="", help="Blender executable (else LAYOUTVLM_BLENDER / PATH)")
    p.add_argument("--high_res", action="store_true")
    p.add_argument("--no_hdri", action="store_true", help="Skip load_hdri in Blender")
    p.add_argument(
        "--gif_duration",
        type=float,
        default=0.5,
        help="Seconds per frame in stitched GIFs (match out.gif default)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    prepare_task_assets = _load_prepare_task_assets()

    with open(args.scene_json_file, "r", encoding="utf-8") as f:
        task = json.load(f)
    task = prepare_task_assets(task, args.asset_dir)

    saved_entries = _collect_saved_jsons(args.results_dir)
    if not saved_entries:
        print(f"No saved_intermediate_states.json under {args.results_dir!r}", file=sys.stderr)
        return 1

    frames = _build_frames_from_saved(task, saved_entries)
    if not frames:
        print("No frames built from solver snapshots.", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    params_path = os.path.join(args.out_dir, TRAJ_PARAMS)
    payload = {
        "task": task,
        "out_dir": os.path.abspath(args.out_dir),
        "frames": frames,
        "high_res": args.high_res,
        "fov_multiplier": 1.1,
        "add_hdri": not args.no_hdri,
        "floor_material": "Travertine008",
        "rotate_90": True,
        "recenter_mesh": True,
        "apply_3dfront_texture": False,
        "combine_obj_components": False,
    }
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    blender_bin = default_blender_executable(args.blender_bin or None)
    if not blender_bin or not os.path.isfile(blender_bin):
        print("Blender executable not found. Set --blender_bin or LAYOUTVLM_BLENDER.", file=sys.stderr)
        return 1
    if not BLENDER_SCRIPT.is_file():
        print(f"Missing {BLENDER_SCRIPT}", file=sys.stderr)
        return 1

    timeout = float(os.environ.get("LAYOUTVLM_TRAJECTORY_TIMEOUT_SEC", "7200"))
    cmd = [
        blender_bin,
        "-b",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        params_path,
    ]
    print("[render_trajectory]", " ".join(cmd[:6]), "...", TRAJ_PARAMS, flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or "")[-8000:]
        ep = os.path.join(args.out_dir, TRAJ_ERROR)
        if os.path.isfile(ep):
            try:
                with open(ep, "r", encoding="utf-8") as ef:
                    err = ef.read()[-8000:] + "\n--- stderr ---\n" + err
            except OSError:
                pass
        print(f"Blender failed (exit {proc.returncode}):\n{err}", file=sys.stderr)
        return proc.returncode or 1

    meta_path = os.path.join(args.out_dir, TRAJ_META)
    if not os.path.isfile(meta_path):
        print(f"Blender did not write {meta_path}", file=sys.stderr)
        return 1
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    groups = meta.get("groups") or []

    all_paths: List[str] = []
    for ginfo in groups:
        g = int(ginfo["group_idx"])
        fpaths = ginfo.get("frames") or []
        all_paths.extend(fpaths)
        gif_out = os.path.join(args.out_dir, f"group_{g}", "out_blender.gif")
        if fpaths:
            _save_gif(fpaths, gif_out, duration_sec=args.gif_duration)
            print("Wrote", gif_out, flush=True)

    final_gif = os.path.join(args.out_dir, "final_blender.gif")
    if all_paths:
        _save_gif(all_paths, final_gif, duration_sec=args.gif_duration)
        print("Wrote", final_gif, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
