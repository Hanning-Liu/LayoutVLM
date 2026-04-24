#!/usr/bin/env python3
"""
List Objaverse assets under objaverse_processed/ by reading each uid's data.json.

Scans the same layout as main.prepare_task_assets:
  <asset_dir>/<uid>/data.json
  <asset_dir>/test_asset_dir/<uid>/data.json

Run in the project conda env, e.g.:
  conda run -n layoutvlm python scripts/list_objaverse_assets.py --category sofa --on_floor
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


def iter_data_json_paths(asset_dir: str) -> Iterator[Tuple[str, str]]:
    """
    Yields (uid, absolute_path_to_data_json).
    Deduplicate by (uid, path) in case of symlinks; prefer first seen.
    """
    seen: Set[str] = set()
    subroots = [asset_dir, os.path.join(asset_dir, "test_asset_dir")]
    for sub in subroots:
        if not os.path.isdir(sub):
            continue
        for name in os.listdir(sub):
            p = os.path.join(sub, name)
            if not os.path.isdir(p):
                continue
            uid = name
            dj = os.path.join(p, "data.json")
            if not os.path.isfile(dj):
                continue
            key = f"{uid}\0{dj}"
            if key in seen:
                continue
            seen.add(key)
            yield uid, os.path.normpath(dj)


def load_record(uid: str, data_json_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(data_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: skip {data_json_path}: {e}", file=sys.stderr)
        return None
    try:
        ann = data["annotations"]
        _ = data["assetMetadata"]["boundingBox"]
    except (KeyError, TypeError) as e:
        print(f"Warning: skip {uid}: missing fields ({e})", file=sys.stderr)
        return None
    return {
        "uid": uid,
        "data_path": data_json_path,
        "annotations": ann,
        "bbox": data["assetMetadata"]["boundingBox"],
    }


def match_filters(
    rec: Dict[str, Any],
    category_sub: Optional[str],
    keyword_sub: Optional[str],
    on_floor: bool,
) -> bool:
    ann = rec["annotations"]
    cat = str(ann.get("category", "")).lower()
    desc = str(ann.get("description", "")).lower()

    if on_floor and not bool(ann.get("onFloor", False)):
        return False
    if category_sub is not None and category_sub.lower() not in cat:
        return False
    if keyword_sub is not None and keyword_sub.lower() not in desc:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List local Objaverse assets (data.json) with optional filters.",
        epilog="If both --category and --keyword are set, an asset must match both (AND).",
    )
    parser.add_argument(
        "--asset_dir",
        default="./objaverse_processed",
        help="Root with <uid>/data.json and test_asset_dir/<uid>/data.json (default: ./objaverse_processed)",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--all",
        action="store_true",
        help="List every asset (no category/keyword filter). Can be very large.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Case-insensitive substring of annotations.category",
    )
    parser.add_argument(
        "--keyword",
        default=None,
        help="Case-insensitive substring of annotations.description",
    )
    parser.add_argument(
        "--on_floor",
        action="store_true",
        help="Only assets with onFloor==true in annotations",
    )
    parser.add_argument(
        "--require_glb",
        action="store_true",
        help="Only assets that have <uid>.glb next to data.json",
    )
    args = parser.parse_args()

    asset_dir = os.path.abspath(args.asset_dir)
    if not os.path.isdir(asset_dir):
        print(f"Error: --asset_dir is not a directory: {asset_dir}", file=sys.stderr)
        return 1

    if not args.all and args.category is None and args.keyword is None:
        print(
            "Error: specify --category and/or --keyword, or use --all.",
            file=sys.stderr,
        )
        return 1

    category_sub = args.category
    keyword_sub = args.keyword
    if args.all:
        category_sub = None
        keyword_sub = None

    rows: List[Dict[str, Any]] = []
    for uid, dpath in iter_data_json_paths(asset_dir):
        rec = load_record(uid, dpath)
        if rec is None:
            continue
        glb = os.path.join(os.path.dirname(dpath), f"{uid}.glb")
        if args.require_glb and not os.path.isfile(glb):
            continue
        if not match_filters(rec, category_sub, keyword_sub, args.on_floor):
            continue
        rows.append(rec)

    # Stable sort: category, uid
    rows.sort(
        key=lambda r: (
            str(r["annotations"].get("category", "")).lower(),
            r["uid"],
        )
    )

    for r in rows:
        ann = r["annotations"]
        bbox = r["bbox"]
        wx = float(bbox.get("x", 0))
        wy = float(bbox.get("y", 0))
        wz = float(bbox.get("z", 0))
        cat = ann.get("category", "")
        scene_key = f'"{r["uid"]}-0": {{}}'
        print(
            f'uid={r["uid"]}  category={cat!r}  '
            f"bbox_m=(x={wx:.3f},y={wy:.3f},z={wz:.3f})  "
            f"onFloor={ann.get('onFloor')}  onObject={ann.get('onObject')}  "
            f"frontView={ann.get('frontView')}\n"
            f"  scene.json: {scene_key}"
        )
    print(f"Total: {len(rows)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
