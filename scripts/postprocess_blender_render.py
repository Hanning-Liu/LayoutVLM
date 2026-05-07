#!/usr/bin/env python3
"""
After ``render_result_blender.py`` inside Blender: apply 2D overlays per PNG.

Blender's bundled Python usually has no Pillow; ``render_result_blender.py`` writes
``_render_postprocess.json`` in ``--out_dir``. Run with conda:

  conda run -n layoutvlm python scripts/postprocess_blender_render.py --out_dir .../final_blender_render

By default we do **not** merge top-down + side views into one ``render.gif`` (that would
animate unrelated cameras). Pass ``--gif-animate-views`` only if you explicitly want that.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", required=True, help="Same as render_result_blender --out_dir")
    p.add_argument(
        "--no_gif",
        action="store_true",
        help="Only apply annotations; skip render.gif.",
    )
    p.add_argument(
        "--no_annotations",
        action="store_true",
        help="Only build render.gif (e.g. Blender already applied overlays).",
    )
    p.add_argument(
        "--gif-animate-views",
        action="store_true",
        help=(
            "Write render.gif as an animation over every listed PNG. Default off: "
            "top-down and side renders stay as separate stills."
        ),
    )
    args = p.parse_args()
    repo = _repo_root()
    if repo not in sys.path:
        sys.path.insert(0, repo)

    meta_path = os.path.join(args.out_dir, "_render_postprocess.json")
    if not os.path.isfile(meta_path):
        print(f"Missing {meta_path}; run render_result_blender.py first.", file=sys.stderr)
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    from utils.annotation_serialize import deserialize_annotation_from_json
    from utils.image_annotate import apply_annotations

    annotations = [
        deserialize_annotation_from_json(a) for a in (meta.get("annotations") or [])
    ]
    if not args.no_annotations and annotations:
        apply_annotations(annotations)
        print("Applied annotations to", len(annotations), "image(s).")

    if args.no_gif:
        return

    output_images = meta.get("output_images") or []
    paths = [p for p in output_images if isinstance(p, str) and p.lower().endswith((".png", ".jpg", ".jpeg"))]
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("No PNG/JPEG paths in metadata; skipping GIF.")
        return
    if len(paths) == 1:
        print("Single output image; not writing render.gif (open the PNG).")
        return
    if not args.gif_animate_views:
        print(
            "Multiple camera views in metadata; not writing render.gif by default "
            "(each view is its own still). Pass --gif-animate-views to force one animated GIF."
        )
        return

    gif_path = os.path.join(args.out_dir, "render.gif")
    try:
        import imageio.v2 as imageio

        frames = [imageio.imread(p) for p in paths]
        imageio.mimsave(gif_path, frames, duration=0.8)
        print("Saved GIF:", gif_path)
        return
    except ImportError:
        pass

    from PIL import Image

    images = [Image.open(p) for p in paths]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=800,
        loop=0,
    )
    for im in images:
        im.close()
    print("Saved GIF (Pillow):", gif_path)


if __name__ == "__main__":
    main()
