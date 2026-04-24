#!/usr/bin/env python3
"""
Auto-pick Objaverse asset UIDs by room size and category, then generate a runnable scene JSON.

Why:
- Avoid obviously-too-large / too-small assets for a given boundary.
- Reduce "random UID" mismatch that leads to poor layouts.

Run in conda env `layoutvlm`, e.g.:
  conda run -n layoutvlm python scripts/scene_autopick_by_size.py \
    --template_scene examples/scene_living_room_sofa_tv.json \
    --categories "sofa,coffee table,television" \
    --out_scene examples/scene_living_room_autopick.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Candidate:
    uid: str
    category: str
    description: str
    on_floor: bool
    on_wall: bool
    on_object: bool
    bbox_x: float
    bbox_y: float
    bbox_z: float
    data_json_path: str

    @property
    def area(self) -> float:
        return max(0.0, self.bbox_x) * max(0.0, self.bbox_y)

    @property
    def max_xy(self) -> float:
        return max(self.bbox_x, self.bbox_y)

    @property
    def min_xy(self) -> float:
        return min(self.bbox_x, self.bbox_y)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def iter_data_json_paths(asset_dir: str) -> Iterable[Tuple[str, str]]:
    subroots = [asset_dir, os.path.join(asset_dir, "test_asset_dir")]
    for sub in subroots:
        if not os.path.isdir(sub):
            continue
        for name in os.listdir(sub):
            p = os.path.join(sub, name)
            if not os.path.isdir(p):
                continue
            dj = os.path.join(p, "data.json")
            if os.path.isfile(dj):
                yield name, os.path.normpath(dj)


def load_candidates(asset_dir: str) -> List[Candidate]:
    asset_dir = os.path.abspath(asset_dir)
    out: List[Candidate] = []
    for uid, dj in iter_data_json_paths(asset_dir):
        try:
            with open(dj, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        ann = data.get("annotations") or {}
        bbox = ((data.get("assetMetadata") or {}).get("boundingBox")) or {}
        out.append(
            Candidate(
                uid=uid,
                category=str(ann.get("category", "")),
                description=str(ann.get("description", "")),
                on_floor=bool(ann.get("onFloor", False)),
                on_wall=bool(ann.get("onWall", False)),
                on_object=bool(ann.get("onObject", False)),
                bbox_x=_safe_float(bbox.get("x")),
                bbox_y=_safe_float(bbox.get("y")),
                bbox_z=_safe_float(bbox.get("z")),
                data_json_path=dj,
            )
        )
    return out


def boundary_room_extents(boundary: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    verts = boundary.get("floor_vertices")
    if not isinstance(verts, list) or not verts:
        raise ValueError("boundary.floor_vertices must be a non-empty list")
    xs = [float(v[0]) for v in verts]
    ys = [float(v[1]) for v in verts]
    zs = [float(v[2]) for v in verts]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def room_dims(boundary: Dict[str, Any]) -> Tuple[float, float, float]:
    minx, maxx, miny, maxy, _, _ = boundary_room_extents(boundary)
    w = maxx - minx
    d = maxy - miny
    area = max(0.0, w) * max(0.0, d)
    return w, d, area


def norm_cat(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _default_target_frac(cat: str) -> float:
    c = norm_cat(cat)
    if "sofa" in c:
        return 0.10
    if "coffee table" in c or "table" == c:
        return 0.04
    if "tv stand" in c:
        return 0.04
    if "television" in c or c == "tv":
        return 0.02
    return 0.05


def _placement_ok(cand: Candidate, requested_cat: str) -> bool:
    c = norm_cat(requested_cat)
    # TVs are often wall-mounted in metadata; allow floor or wall.
    if "television" in c or c == "tv":
        return cand.on_floor or cand.on_wall
    return cand.on_floor


def pick_one_for_category(
    requested_cat: str,
    candidates: Sequence[Candidate],
    room_min_dim: float,
    room_area: float,
    max_dim_ratio: float,
    min_dim_ratio: float,
    max_total_area_ratio: float,
    already_picked_area: float,
    topk: int = 5,
) -> Tuple[Candidate, List[Tuple[Candidate, float]]]:
    """
    Returns best candidate and a ranked list with scores (best first).
    """
    req = norm_cat(requested_cat)
    target_frac = _default_target_frac(req)
    target_area = target_frac * room_area

    ranked: List[Tuple[Candidate, float]] = []
    for cand in candidates:
        if req not in norm_cat(cand.category):
            continue
        if not _placement_ok(cand, req):
            continue
        if cand.max_xy <= 0 or cand.min_xy <= 0:
            continue

        if cand.max_xy > room_min_dim * max_dim_ratio:
            continue
        if cand.min_xy < room_min_dim * min_dim_ratio:
            continue
        if already_picked_area + cand.area > room_area * max_total_area_ratio:
            continue

        # Score: prefer area near target; prefer not-too-tall; mild preference for less "onObject".
        area_err = abs(cand.area - target_area) / max(1e-6, target_area)
        tall_penalty = max(0.0, (cand.bbox_z / max(1e-6, room_min_dim)) - 0.8)
        on_object_penalty = 0.3 if cand.on_object else 0.0
        score = -(area_err + 0.3 * tall_penalty + on_object_penalty)
        ranked.append((cand, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    if not ranked:
        raise RuntimeError(f"No candidate fits size constraints for category={requested_cat!r}")
    best = ranked[0][0]
    return best, ranked[: max(1, topk)]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Auto-pick asset UIDs by room size and category, then generate a scene JSON.",
    )
    p.add_argument("--template_scene", required=True, help="Scene JSON with boundary/task_description/layout_criteria")
    p.add_argument("--asset_dir", default="./objaverse_processed", help="Objaverse processed directory")
    p.add_argument(
        "--categories",
        required=True,
        help='Comma-separated requested categories (substring match), e.g. "sofa,coffee table,television"',
    )
    p.add_argument("--out_scene", required=True, help="Output scene JSON path")
    p.add_argument("--seed", type=int, default=0, help="Reserved for future randomized selection (unused for now)")
    p.add_argument("--max_dim_ratio", type=float, default=0.8, help="Reject if max(x,y) > room_min_dim * ratio")
    p.add_argument("--min_dim_ratio", type=float, default=0.05, help="Reject if min(x,y) < room_min_dim * ratio")
    p.add_argument(
        "--max_total_area_ratio",
        type=float,
        default=0.6,
        help="Reject picks if cumulative footprint area exceeds room_area * ratio",
    )
    p.add_argument("--topk", type=int, default=5, help="Show top-k alternatives per category")
    args = p.parse_args()

    with open(args.template_scene, "r", encoding="utf-8") as f:
        tmpl = json.load(f)
    if "boundary" not in tmpl:
        print("Error: template_scene missing boundary", file=sys.stderr)
        return 2

    w, d, area = room_dims(tmpl["boundary"])
    room_min_dim = min(w, d)
    if area <= 0 or room_min_dim <= 0:
        print(f"Error: invalid room dims computed: w={w}, d={d}, area={area}", file=sys.stderr)
        return 2

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    if not cats:
        print("Error: --categories is empty after parsing", file=sys.stderr)
        return 2

    cands = load_candidates(args.asset_dir)
    if not cands:
        print(f"Error: no candidates found under {args.asset_dir}", file=sys.stderr)
        return 2

    picked: List[Candidate] = []
    picked_area = 0.0
    report: List[str] = []
    report.append("== Room precheck ==")
    report.append(f"room_w={w:.3f}m room_d={d:.3f}m room_area={area:.3f}m^2 room_min_dim={room_min_dim:.3f}m")
    report.append(
        f"constraints: max_dim_ratio={args.max_dim_ratio} min_dim_ratio={args.min_dim_ratio} max_total_area_ratio={args.max_total_area_ratio}"
    )
    report.append("")

    for cat in cats:
        best, top = pick_one_for_category(
            requested_cat=cat,
            candidates=cands,
            room_min_dim=room_min_dim,
            room_area=area,
            max_dim_ratio=args.max_dim_ratio,
            min_dim_ratio=args.min_dim_ratio,
            max_total_area_ratio=args.max_total_area_ratio,
            already_picked_area=picked_area,
            topk=args.topk,
        )
        picked.append(best)
        picked_area += best.area

        report.append(f"== Picked for category={cat!r} ==")
        report.append(
            f"uid={best.uid}  meta_category={best.category!r}  bbox=(x={best.bbox_x:.3f},y={best.bbox_y:.3f},z={best.bbox_z:.3f})  "
            f"area={best.area:.3f}  onFloor={best.on_floor}  onWall={best.on_wall}  onObject={best.on_object}"
        )
        report.append("Top alternatives:")
        for cand, score in top:
            report.append(
                f"  score={score:+.3f}  uid={cand.uid}  cat={cand.category!r}  "
                f"bbox=(x={cand.bbox_x:.3f},y={cand.bbox_y:.3f},z={cand.bbox_z:.3f})  area={cand.area:.3f}"
            )
        report.append("")

    report.append(f"picked_total_footprint_area={picked_area:.3f}m^2 ({picked_area/area:.1%} of room)")

    # Build output scene JSON (keep boundary + text; replace assets with uid keys).
    out_scene: Dict[str, Any] = {
        "task_description": tmpl.get("task_description", ""),
        "layout_criteria": tmpl.get("layout_criteria", ""),
        "boundary": tmpl["boundary"],
        "assets": {f"{c.uid}-0": {} for c in picked},
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_scene)) or ".", exist_ok=True)
    with open(args.out_scene, "w", encoding="utf-8") as f:
        json.dump(out_scene, f, indent=2)

    # Write a sibling report file for later inspection.
    report_path = os.path.splitext(args.out_scene)[0] + ".report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nWrote scene: {args.out_scene}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

