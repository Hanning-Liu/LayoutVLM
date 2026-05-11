#!/usr/bin/env python3
"""
Convert LayoutVLM batch outputs (layout.json + bench scene JSON) to Sengine benchmark JSON shape:
top-level { room_info, furniture_data } matching reference/sengine_benchmark_json_format_reference.json.

Path pairing matches scripts/batch_run_layoutvlm.scene_to_save_dir (inverse):
  scene:   <scene_root>/bench_<type>/<stem>.json
  layout:  <layout_root>/bench_<type>/<stem>/layout.json

Furniture geometry follows reference/convert_holodeck_batch_to_sengine.py:
  position [x, z, 0], rotation [-90, 0, yaw_deg], size [dx, dz, height].

Bbox x/y are swapped like main.prepare_task_assets before mapping to LayoutVLM/sengine axes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


def format_asset_uuid(asset_id: str) -> str:
    s = (asset_id or "").strip().replace("-", "").lower()
    if len(s) == 32 and all(c in "0123456789abcdef" for c in s):
        return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:32]}"
    return asset_id


def instance_uid_to_model_uid(instance_key: str) -> str:
    """Strip trailing -<idx> (LayoutVLM instance id)."""
    if "-" in instance_key:
        parts = instance_key.rsplit("-", 1)
        if parts[1].isdigit():
            return parts[0]
    return instance_key


def find_data_json(asset_dir: Path, model_uid: str) -> Optional[Path]:
    for root in (asset_dir, asset_dir / "test_asset_dir"):
        p = root / model_uid / "data.json"
        if p.is_file():
            return p
    return None


def load_bbox_swapped_like_main(data: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """Same x/y swap as main.prepare_task_assets (see main.py)."""
    try:
        bb = data.get("assetMetadata", {}).get("boundingBox", {})
        raw_x = float(bb["x"])
        raw_y = float(bb["y"])
        raw_z = float(bb["z"])
        return raw_y, raw_x, raw_z
    except (KeyError, TypeError, ValueError):
        return None


def layout_rotation_to_yaw_deg(rot: Any) -> float:
    """
    SandBoxEnv.export_layout writes rotation as [np.rad2deg(cos_t), np.rad2deg(sin_t)] (2-vector).
    atan2 is homogeneous: atan2(k*sin, k*cos) == atan2(sin, cos), so yaw = degrees(atan2(r1, r0))
    recovers heading for both raw cos/sin and rad2deg-scaled components.
    If three+ numbers are present, treat the third as explicit yaw (Holodeck-style euler z).
    """
    if not isinstance(rot, (list, tuple)) or len(rot) < 2:
        return 0.0
    r0, r1 = float(rot[0]), float(rot[1])
    if len(rot) >= 3:
        r2 = float(rot[2])
        if abs(r0) <= 1.1 and abs(r1) <= 1.1 and abs(r2) <= 1e-6:
            return math.degrees(math.atan2(r1, r0))
        return r2
    return math.degrees(math.atan2(r1, r0))


def floor_vertices_to_exterior_2d(floor_vertices: Sequence[Sequence[Any]]) -> List[List[float]]:
    """2D ring [x,y] from floor_vertices; closed (first repeated at end) like sengine reference."""
    pts: List[List[float]] = []
    for v in floor_vertices:
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        pts.append([float(v[0]), float(v[1])])
    if len(pts) < 3:
        return pts
    p0, plast = pts[0], pts[-1]
    if abs(p0[0] - plast[0]) < 1e-9 and abs(p0[1] - plast[1]) < 1e-9:
        ring = pts
    else:
        ring = pts + [list(pts[0])]
    return ring


def room_label_from_bench_dir(bench_dir_name: str) -> str:
    if bench_dir_name.startswith("bench_"):
        return bench_dir_name[len("bench_") :]
    return bench_dir_name


def build_room_info(
    scene: Dict[str, Any],
    *,
    bench_dir_name: str,
    scene_stem: str,
) -> Dict[str, Any]:
    boundary = scene.get("boundary") or {}
    floor_vertices = boundary.get("floor_vertices") or []
    exterior = floor_vertices_to_exterior_2d(floor_vertices)
    room_type = room_label_from_bench_dir(bench_dir_name)
    rid = uuid.uuid5(uuid.NAMESPACE_DNS, f"layoutvlm:{room_type}:{scene_stem}")
    return {
        "door": [],
        "room": {
            "label": str(room_type),
            "exterior": exterior,
        },
        "window": [],
        "room_id": 0,
        "uuid": str(rid),
    }


def annotations_to_labels(ann: Any) -> Tuple[str, str]:
    if not isinstance(ann, dict):
        return "object", ""
    cat = str(ann.get("category") or "object")
    sub = ann.get("subcategory") or ann.get("二级分类") or ann.get("second_category")
    sub_s = str(sub) if sub else ""
    return cat, sub_s


def build_furniture_data(
    layout: Dict[str, Any],
    scene: Dict[str, Any],
    asset_dir: Path,
) -> List[Dict[str, Any]]:
    scene_assets = scene.get("assets")
    if not isinstance(scene_assets, dict):
        scene_assets = {}

    name_counts: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []

    for inst_key in sorted(layout.keys()):
        entry = layout.get(inst_key)
        if not isinstance(entry, dict):
            continue
        pos = entry.get("position")
        rot = entry.get("rotation")
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            continue

        model_uid = instance_uid_to_model_uid(str(inst_key))
        model_uuid = format_asset_uuid(model_uid)

        ann: Any = {}
        if isinstance(scene_assets.get(inst_key), dict):
            ann = (scene_assets.get(inst_key) or {}).get("annotations") or {}
        if not ann:
            dj = find_data_json(asset_dir, model_uid)
            if dj is not None:
                try:
                    with open(dj, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    ann = data.get("annotations") or {}
                except (OSError, json.JSONDecodeError):
                    ann = {}

        cat3, cat2 = annotations_to_labels(ann)
        cnt = name_counts.get(cat3, 0)
        name_counts[cat3] = cnt + 1
        scene_uuid = f"{cat3}_{model_uuid}_{cnt}"

        dj_path = find_data_json(asset_dir, model_uid)
        size: List[float] = [0.5, 0.5, 0.5]
        if dj_path is not None:
            try:
                with open(dj_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                bb = load_bbox_swapped_like_main(data)
                if bb is not None:
                    dx, dy, dz = bb
                    size = [dx, dy, dz]
            except (OSError, json.JSONDecodeError):
                pass

        px, py = float(pos[0]), float(pos[1])
        yaw = layout_rotation_to_yaw_deg(rot)

        gidx = len(out)
        out.append(
            {
                "model_uuid": model_uuid,
                "scene_uuid": scene_uuid,
                "global_index": gidx,
                "三级分类": cat3,
                "二级分类": cat2,
                "geometry_data": {
                    "size": size,
                    "scale": [1.0, 1.0, 1.0],
                    "position": [px, py, 0.0],
                    "rotation": [-90.0, 0.0, yaw],
                },
            }
        )
    return out


def iter_layout_tasks(layout_root: Path) -> List[Path]:
    tasks: List[Path] = []
    if not layout_root.is_dir():
        return tasks
    for bench in sorted(layout_root.iterdir()):
        if not bench.is_dir() or not bench.name.startswith("bench_"):
            continue
        for sub in sorted(bench.iterdir()):
            if not sub.is_dir():
                continue
            lp = sub / "layout.json"
            if lp.is_file():
                tasks.append(lp)
    return tasks


def layout_path_to_scene_path(layout_path: Path, scene_root: Path) -> Path:
    stem = layout_path.parent.name
    bench_name = layout_path.parent.parent.name
    return scene_root / bench_name / f"{stem}.json"


def convert_one(
    layout_path: Path,
    scene_root: Path,
    asset_dir: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    scene_path = layout_path_to_scene_path(layout_path, scene_root)
    if not scene_path.is_file():
        return None, f"missing scene {scene_path}"

    try:
        with open(layout_path, "r", encoding="utf-8") as f:
            layout = json.load(f)
        with open(scene_path, "r", encoding="utf-8") as f:
            scene = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None, str(e)

    if not isinstance(layout, dict):
        return None, "layout.json is not an object"
    if not isinstance(scene, dict):
        return None, "scene json is not an object"

    bench_dir = layout_path.parent.parent.name
    stem = layout_path.parent.name
    room_info = build_room_info(scene, bench_dir_name=bench_dir, scene_stem=stem)
    furniture_data = build_furniture_data(layout, scene, asset_dir)
    return {"room_info": room_info, "furniture_data": furniture_data}, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout-root",
        type=str,
        default=str(REPO_ROOT / "results" / "out_batch"),
    )
    parser.add_argument(
        "--scene-root",
        type=str,
        default=str(REPO_ROOT / "out"),
    )
    parser.add_argument(
        "--asset-dir",
        type=str,
        default=str(REPO_ROOT / "objaverse_processed"),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(REPO_ROOT / "results" / "sengine_from_layoutvlm"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    layout_root = Path(args.layout_root).resolve()
    scene_root = Path(args.scene_root).resolve()
    asset_dir = Path(args.asset_dir).resolve()
    output_root = Path(args.output_root).resolve()

    tasks = iter_layout_tasks(layout_root)
    if args.limit is not None:
        tasks = tasks[: max(0, args.limit)]

    ok = 0
    skipped = 0
    for layout_path in tasks:
        doc, err = convert_one(layout_path, scene_root, asset_dir)
        if doc is None:
            print(f"[SKIP] {layout_path}: {err}", file=sys.stderr)
            skipped += 1
            continue

        rel = layout_path.relative_to(layout_root)
        out_path = output_root / rel.parent.parent / rel.parent.name / f"{rel.parent.name}.json"
        if not args.dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
                f.write("\n")
        ok += 1

    print(
        f"done: converted={ok} skipped={skipped} dry_run={args.dry_run} "
        f"output_root={output_root}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
