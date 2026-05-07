#!/usr/bin/env python3
"""
Build benchmark_tasks-style JSON (bedroom_0 shape) from eval-dataset jsonl + extracted_rooms.
Does not import anything from scripts/.

Run from repo root, e.g.:
  python -m eval_benchmark_gen.build_benchmark_from_eval_dataset \\
    --jsonl-path eval-dataset/bathroom/bathroom.jsonl --room-index 0 --out-dir out/bench
"""
from __future__ import annotations

import argparse
import ast
import base64
import importlib.util
import json
import os
import re
import sys
import time
import signal
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Load DashScope client without importing src.layoutvlm (avoids heavy deps like torch).
_qwen_spec = importlib.util.spec_from_file_location(
    "qwen_dashscope_client_standalone",
    _REPO_ROOT / "src" / "layoutvlm" / "qwen_dashscope_client.py",
)
if _qwen_spec is None or _qwen_spec.loader is None:
    raise RuntimeError("Cannot load qwen_dashscope_client.py")
_qwen_mod = importlib.util.module_from_spec(_qwen_spec)
_qwen_spec.loader.exec_module(_qwen_mod)
get_dashscope_client = _qwen_mod.get_dashscope_client
chat_completions_text = _qwen_mod.chat_completions_text

# ---------------------------------------------------------------------------
# Subdir name (under eval-dataset) -> key in extracted_rooms["rooms"]
# ---------------------------------------------------------------------------
SUBDIR_TO_EXTRACTED_ROOM_KEY: Dict[str, str] = {
    "bathroom": "bathroom",
    "kitchen": "kitchen",
    "livingroom": "livingroom",
    "studio": "studio",
    "balcony": "balcony",
    "bedroom": "bedroom",
    "esports": "esports",
    "childrenroom": "childrenroom",
}


def resolve_room_key(subdir_name: str, override: Optional[str]) -> str:
    if override:
        return override
    key = SUBDIR_TO_EXTRACTED_ROOM_KEY.get(subdir_name)
    if key is None:
        raise ValueError(
            f"Unknown eval subdir {subdir_name!r}. Known: {sorted(SUBDIR_TO_EXTRACTED_ROOM_KEY)} "
            "or pass --extracted-room-key"
        )
    return key


def load_prompts_strings(repo_root: Path) -> Tuple[str, str, str]:
    path = repo_root / "prompts-lhn" / "prompts.py"
    spec = importlib.util.spec_from_file_location("prompts_lhn_prompts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prompts module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    try:
        return (mod.PROMPT_CRITERIA, mod.PROMPT_FUR_LIST, mod.PROMPT_FUR_JUDGE)
    except Exception as e:
        raise RuntimeError(
            f"prompts.py must define PROMPT_CRITERIA / PROMPT_FUR_LIST / PROMPT_FUR_JUDGE: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _dedupe_closed_ring(points: List[List[float]]) -> List[List[float]]:
    if len(points) < 2:
        return points
    p0, plast = points[0], points[-1]
    if abs(p0[0] - plast[0]) < 1e-9 and abs(p0[1] - plast[1]) < 1e-9:
        return points[:-1]
    return points


def shoelace_area_2d(poly: Sequence[Sequence[float]]) -> float:
    """Polygon area (m^2); poly vertices in order (CCW or CW)."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += poly[i][0] * poly[j][1]
        s -= poly[j][0] * poly[i][1]
    return abs(s) * 0.5


def exterior_to_floor_vertices(
    exterior_2d: List[List[float]],
) -> Tuple[List[List[float]], float, float, float]:
    """
    Returns translated floor_vertices [[x,y,0],...], width, depth, polygon_area.
    """
    pts = _dedupe_closed_ring([[float(p[0]), float(p[1])] for p in exterior_2d])
    if len(pts) < 3:
        raise ValueError("exterior must have at least 3 distinct vertices")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, miny = min(xs), min(ys)
    tx = [p[0] - minx for p in pts]
    ty = [p[1] - miny for p in pts]
    poly = list(zip(tx, ty))
    area = shoelace_area_2d(poly)
    width = max(tx) - min(tx)
    depth = max(ty) - min(ty)
    floor_vertices = [[x, y, 0.0] for x, y in poly]
    return floor_vertices, width, depth, area


def boundary_from_extracted(
    exterior_2d: List[List[float]], wall_height: float
) -> Dict[str, Any]:
    floor_vertices, _, _, _ = exterior_to_floor_vertices(exterior_2d)
    return {"floor_vertices": floor_vertices, "wall_height": float(wall_height)}


def room_dims_from_boundary(boundary: Dict[str, Any]) -> Tuple[float, float, float]:
    verts = boundary["floor_vertices"]
    xs = [float(v[0]) for v in verts]
    ys = [float(v[1]) for v in verts]
    w = max(xs) - min(xs)
    d = max(ys) - min(ys)
    area = max(0.0, w) * max(0.0, d)
    return w, d, area


# ---------------------------------------------------------------------------
# Asset candidates (same layout as main.prepare_task_assets)
# ---------------------------------------------------------------------------
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


def iter_data_json_paths(asset_dir: Path) -> Generator[Tuple[str, Path], None, None]:
    seen: Set[str] = set()
    for sub in (asset_dir, asset_dir / "test_asset_dir"):
        if not sub.is_dir():
            continue
        for name in os.listdir(sub):
            p = sub / name
            if not p.is_dir():
                continue
            dj = p / "data.json"
            if not dj.is_file():
                continue
            uid = name
            key = f"{uid}\0{dj.resolve()}"
            if key in seen:
                continue
            seen.add(key)
            yield uid, dj


def load_candidates(asset_dir: str, *, log_every: int = 5000) -> List[Candidate]:
    root = Path(os.path.abspath(asset_dir))
    out: List[Candidate] = []
    t0 = time.time()
    n = 0
    for uid, dj in iter_data_json_paths(root):
        n += 1
        if log_every > 0 and n % log_every == 0:
            dt = time.time() - t0
            print(f"[candidates] scanned={n} kept={len(out)} elapsed={dt:.1f}s ...", file=sys.stderr)
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
                data_json_path=str(dj),
            )
        )
    dt = time.time() - t0
    print(f"[candidates] done scanned={n} kept={len(out)} elapsed={dt:.1f}s", file=sys.stderr)
    return out


def save_candidates_cache(path: Path, candidates: Sequence[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for c in candidates:
        rows.append(
            {
                "uid": c.uid,
                "category": c.category,
                "description": c.description,
                "on_floor": c.on_floor,
                "on_wall": c.on_wall,
                "on_object": c.on_object,
                "bbox_x": c.bbox_x,
                "bbox_y": c.bbox_y,
                "bbox_z": c.bbox_z,
                "data_json_path": c.data_json_path,
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def load_candidates_cache(path: Path) -> List[Candidate]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("candidates cache must be a JSON list")
    out: List[Candidate] = []
    for r in rows:
        out.append(
            Candidate(
                uid=str(r["uid"]),
                category=str(r.get("category", "")),
                description=str(r.get("description", "")),
                on_floor=bool(r.get("on_floor", False)),
                on_wall=bool(r.get("on_wall", False)),
                on_object=bool(r.get("on_object", False)),
                bbox_x=float(r.get("bbox_x", 0.0)),
                bbox_y=float(r.get("bbox_y", 0.0)),
                bbox_z=float(r.get("bbox_z", 0.0)),
                data_json_path=str(r.get("data_json_path", "")),
            )
        )
    return out


def norm_cat(s: str) -> str:
    return " ".join(s.strip().lower().split())


def matches_furniture_query(requested: str, cand: Candidate) -> bool:
    r = norm_cat(requested)
    cat = norm_cat(cand.category)
    desc = norm_cat(cand.description)
    if not r:
        return False
    if r in cat or cat in r:
        return True
    if r in desc:
        return True
    words = [w for w in r.replace("_", " ").split() if len(w) > 2]
    return any(w in cat or w in desc for w in words)


def placement_ok(cand: Candidate, requested: str) -> bool:
    c = norm_cat(requested)
    if "television" in c or c == "tv":
        return cand.on_floor or cand.on_wall
    # Typical floor furniture
    return cand.on_floor


def _default_target_frac(requested: str) -> float:
    c = norm_cat(requested)
    if "sofa" in c:
        return 0.10
    if "coffee table" in c or re.search(r"\btable\b", c):
        return 0.04
    if "tv stand" in c:
        return 0.04
    if "television" in c or c == "tv":
        return 0.02
    return 0.05


def rank_candidates_for_query(
    requested: str,
    candidates: Sequence[Candidate],
    room_min_dim: float,
    room_area: float,
    max_dim_ratio: float,
    min_dim_ratio: float,
    max_total_area_ratio: float,
    already_picked_area: float,
) -> List[Tuple[Candidate, float]]:
    req = norm_cat(requested)
    target_frac = _default_target_frac(requested)
    target_area = target_frac * room_area

    ranked: List[Tuple[Candidate, float]] = []
    for cand in candidates:
        if not matches_furniture_query(requested, cand):
            continue
        if not placement_ok(cand, requested):
            continue
        if cand.max_xy <= 0 or cand.min_xy <= 0:
            continue
        if cand.max_xy > room_min_dim * max_dim_ratio:
            continue
        if cand.min_xy < room_min_dim * min_dim_ratio:
            continue
        if already_picked_area + cand.area > room_area * max_total_area_ratio:
            continue

        area_err = abs(cand.area - target_area) / max(1e-6, target_area)
        tall_penalty = max(0.0, (cand.bbox_z / max(1e-6, room_min_dim)) - 0.8)
        on_object_penalty = 0.3 if cand.on_object else 0.0
        score = -(area_err + 0.3 * tall_penalty + on_object_penalty)
        ranked.append((cand, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def find_preview_images(uid_dir: Path) -> List[Path]:
    names = (
        "thumbnail.jpg",
        "thumbnail.png",
        "preview.jpg",
        "preview.png",
        "front.jpg",
        "render.jpg",
    )
    found: List[Path] = []
    for n in names:
        p = uid_dir / n
        if p.is_file():
            found.append(p)
    if not found:
        for p in sorted(uid_dir.glob("*.jpg")):
            found.append(p)
            break
        for p in sorted(uid_dir.glob("*.png")):
            if p not in found:
                found.append(p)
                break
    return found[:3]


def image_to_data_url(path: Path) -> str:
    suf = path.suffix.lower()
    mime = "image/jpeg" if suf in (".jpg", ".jpeg") else "image/png"
    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
class _TimeoutError(RuntimeError):
    pass


@contextmanager
def _alarm_timeout(seconds: int):
    if seconds <= 0:
        yield
        return

    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(_signum, _frame):
        raise _TimeoutError(f"Timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def call_text_llm(
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int = 2048,
    timeout_s: int = 60,
) -> str:
    with _alarm_timeout(timeout_s):
        return chat_completions_text(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            retries=2,
        )


def call_vision_judge(
    client: Any,
    model: str,
    prompt: str,
    image_paths: Sequence[Path],
    timeout_s: int = 60,
) -> str:
    parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in image_paths:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(p)},
            }
        )
    with _alarm_timeout(timeout_s):
        return chat_completions_text(
            client=client,
            model=model,
            messages=[{"role": "user", "content": parts}],
            max_tokens=64,
            temperature=0.0,
            retries=2,
        )


def extract_dict_text(text: str) -> str:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if m:
        return m.group(0)
    return text


def parse_furniture_dict(response: str) -> Dict[str, List[int]]:
    s = extract_dict_text(response)
    # Fix common LLM mistakes: trailing commas
    s = re.sub(r",\s*}", "}", s)
    try:
        out = ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Could not parse furniture dict: {e}\nRaw:\n{s[:2000]}") from e
    if not isinstance(out, dict):
        raise ValueError("FUR-LIST model output is not a dict")
    parsed: Dict[str, List[int]] = {}
    for k, v in out.items():
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            parsed[str(k)] = [int(v[0]), int(v[1])]
        else:
            raise ValueError(f"Bad entry for {k!r}: {v!r}")
    return parsed


def judge_response_is_true(text: str) -> bool:
    t = text.strip().lower()
    if "true" in t and "false" not in t:
        return True
    if t == "true":
        return True
    if re.match(r"^\s*true\s*$", t):
        return True
    return False


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    dotenv_path = _REPO_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)


def build_room_size_string(width: float, depth: float, poly_area: float) -> str:
    return (
        f"approximately {poly_area:.2f} m^2 polygon floor area "
        f"(axis-aligned bounding box about {width:.2f} m by {depth:.2f} m). "
        f"The floor is a general polygon, not necessarily a rectangle."
    )


def pick_uids_for_object(
    object_phrase: str,
    candidates: Sequence[Candidate],
    boundary: Dict[str, Any],
    *,
    num_distinct_types: int,
    count: int,
    asset_root: Path,
    client: Any,
    text_model: str,
    vision_model: str,
    task_description: str,
    layout_criteria: str,
    judge_no_image: str,
    max_dim_ratio: float,
    min_dim_ratio: float,
    max_total_area_ratio: float,
    picked_footprint_area: float,
    skip_missing: bool,
    prompt_judge: str,
    llm_timeout_s: int,
) -> Tuple[List[str], float]:
    """
    Returns list of `count` uids (with repetition across types) and added footprint area.
    """
    w, d, room_area = room_dims_from_boundary(boundary)
    room_min_dim = min(w, d)
    ranked = rank_candidates_for_object(
        object_phrase,
        candidates,
        room_min_dim,
        room_area,
        max_dim_ratio,
        min_dim_ratio,
        max_total_area_ratio,
        picked_footprint_area,
    )
    if not ranked and skip_missing:
        return [], 0.0
    if not ranked:
        raise RuntimeError(f"No candidate assets match query {object_phrase!r} under size filters")

    # Need num_distinct_types distinct UIDs that pass judge; then assign count instances round-robin.
    approved: List[str] = []
    for cand, _score in ranked:
        if len(approved) >= num_distinct_types:
            break
        if cand.uid in approved:
            continue
        ok = run_furniture_judge(
            cand,
            object_phrase,
            asset_root,
            client,
            text_model,
            vision_model,
            task_description,
            layout_criteria,
            judge_no_image,
            prompt_judge,
            llm_timeout_s=llm_timeout_s,
        )
        if ok:
            approved.append(cand.uid)

    if len(approved) < num_distinct_types:
        if skip_missing:
            num_distinct_types = max(1, len(approved))
        else:
            raise RuntimeError(
                f"Could not approve {num_distinct_types} distinct types for {object_phrase!r}; got {len(approved)}"
            )

    if not approved:
        if skip_missing:
            return [], 0.0
        raise RuntimeError(f"Judge rejected all candidates for {object_phrase!r}")

    # Round-robin count instances across approved types
    uids_for_instances: List[str] = []
    for i in range(count):
        uids_for_instances.append(approved[i % len(approved)])

    added_area = 0.0
    for uid in set(uids_for_instances):
        # approximate footprint contribution
        cnext = next((c for c, _ in ranked if c.uid == uid), None)
        if cnext:
            n = uids_for_instances.count(uid)
            added_area += cnext.area * n

    return uids_for_instances, added_area


def rank_candidates_for_object(
    object_phrase: str,
    candidates: Sequence[Candidate],
    room_min_dim: float,
    room_area: float,
    max_dim_ratio: float,
    min_dim_ratio: float,
    max_total_area_ratio: float,
    picked_footprint_area: float,
) -> List[Tuple[Candidate, float]]:
    return rank_candidates_for_query(
        object_phrase,
        candidates,
        room_min_dim,
        room_area,
        max_dim_ratio,
        min_dim_ratio,
        max_total_area_ratio,
        picked_footprint_area,
    )


def run_furniture_judge(
    cand: Candidate,
    object_looking_for: str,
    asset_root: Path,
    client: Any,
    text_model: str,
    vision_model: str,
    task_description: str,
    layout_criteria: str,
    judge_no_image: str,
    prompt_judge: str,
    llm_timeout_s: int = 60,
) -> bool:
    uid_dir = None
    for sub in (asset_root, asset_root / "test_asset_dir"):
        d = sub / cand.uid
        if d.is_dir():
            uid_dir = d
            break
    imgs: List[Path] = []
    if uid_dir is not None:
        imgs = find_preview_images(uid_dir)

    obj_desc = norm_cat(cand.description) or norm_cat(cand.category)
    pj = (
        prompt_judge.replace("TASK_DESCRIPTION", task_description)
        .replace("LAYOUT_CRITERIA", layout_criteria)
        .replace("OBJECT_DESCRIPTION", obj_desc)
        .replace("OBJECT_LOOKING_FOR", object_looking_for)
    )

    if imgs:
        try:
            text = call_vision_judge(client, vision_model, pj, imgs, timeout_s=llm_timeout_s)
        except Exception:
            text = ""
        if judge_response_is_true(text):
            return True
        # fall through to retry without images if configured
        if judge_no_image != "skip":
            text = call_text_llm(client, text_model, pj, timeout_s=llm_timeout_s)
            return judge_response_is_true(text)
        return False

    if judge_no_image == "skip":
        return False
    text = call_text_llm(client, text_model, pj, timeout_s=llm_timeout_s)
    return judge_response_is_true(text)


def expand_assets_from_furniture_dict(
    fur: Dict[str, List[int]],
    candidates: List[Candidate],
    boundary: Dict[str, Any],
    asset_dir: str,
    client: Any,
    text_model: str,
    vision_model: str,
    task_description: str,
    layout_criteria: str,
    prompt_judge: str,
    judge_no_image: str,
    llm_timeout_s: int,
    max_dim_ratio: float,
    min_dim_ratio: float,
    max_total_area_ratio: float,
    skip_missing: bool,
) -> Dict[str, Dict[str, Any]]:
    asset_root = Path(os.path.abspath(asset_dir))
    assets: Dict[str, Dict[str, Any]] = {}
    picked_area = 0.0
    uid_counters: Dict[str, int] = {}

    for phrase, pair in fur.items():
        count, num_types = int(pair[0]), max(1, int(pair[1]))
        uids_list, added = pick_uids_for_object(
            phrase,
            candidates,
            boundary,
            num_distinct_types=num_types,
            count=count,
            asset_root=asset_root,
            client=client,
            text_model=text_model,
            vision_model=vision_model,
            task_description=task_description,
            layout_criteria=layout_criteria,
            judge_no_image=judge_no_image,
            max_dim_ratio=max_dim_ratio,
            min_dim_ratio=min_dim_ratio,
            max_total_area_ratio=max_total_area_ratio,
            picked_footprint_area=picked_area,
            skip_missing=skip_missing,
            prompt_judge=prompt_judge,
            llm_timeout_s=llm_timeout_s,
        )
        picked_area += added
        for uid in uids_list:
            idx = uid_counters.get(uid, 0)
            key = f"{uid}-{idx}"
            uid_counters[uid] = idx + 1
            assets[key] = {}

    return assets


@dataclass(frozen=True)
class GenTask:
    """One jsonl row to turn into an output JSON (after phase-A scanning)."""

    line_idx: int
    eid: str
    row: Dict[str, Any]
    room_idx: int
    boundary: Dict[str, Any]
    poly_area: float
    width: float
    depth: float
    out_path: Path


def run_single_benchmark_task(
    task: GenTask,
    *,
    prompts: Tuple[str, str, str],
    client: Any,
    candidates: List[Candidate],
    asset_dir: str,
    judge_no_image: str,
    llm_timeout_s: int,
    max_dim_ratio: float,
    min_dim_ratio: float,
    max_total_area_ratio: float,
    skip_missing: bool,
    dry_run: bool,
    text_model: str,
    vision_model: str,
    room_key: str,
) -> None:
    print(f"[rooms] line id={task.eid} rooms[{room_key}][{task.room_idx}]", file=sys.stderr)
    try:
        doc = process_one_line(
            task.row,
            boundary=task.boundary,
            poly_area=task.poly_area,
            width=task.width,
            depth=task.depth,
            prompts=prompts,
            client=client,
            text_model=text_model,
            vision_model=vision_model,
            candidates=candidates,
            asset_dir=asset_dir,
            judge_no_image=judge_no_image,
            llm_timeout_s=llm_timeout_s,
            max_dim_ratio=max_dim_ratio,
            min_dim_ratio=min_dim_ratio,
            max_total_area_ratio=max_total_area_ratio,
            skip_missing=skip_missing,
            dry_run=dry_run,
        )
    except Exception as ex:
        print(f"Error on {task.eid}: {ex}", file=sys.stderr)
        raise

    with open(task.out_path, "w", encoding="utf-8") as wf:
        json.dump(doc, wf, indent=4)
    print(f"Wrote {task.out_path}")


def _run_single_benchmark_task_mp(
    task: GenTask,
    dashscope_api_key: Optional[str],
    dashscope_base_url: Optional[str],
    run_kw: Dict[str, Any],
) -> None:
    """Process-pool entry: pickle-safe; builds OpenAI client in the child process."""
    if run_kw.get("dry_run"):
        client: Any = None
    else:
        client = get_dashscope_client(
            api_key=dashscope_api_key,
            base_url=dashscope_base_url,
        )
    run_single_benchmark_task(task, client=client, **run_kw)


def process_one_line(
    row: Dict[str, Any],
    *,
    boundary: Dict[str, Any],
    poly_area: float,
    width: float,
    depth: float,
    prompts: Tuple[str, str, str],
    client: Any,
    text_model: str,
    vision_model: str,
    candidates: List[Candidate],
    asset_dir: str,
    judge_no_image: str,
    llm_timeout_s: int,
    max_dim_ratio: float,
    min_dim_ratio: float,
    max_total_area_ratio: float,
    skip_missing: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    task_description = str(row["text"])
    pc, pf, pj = prompts
    room_size = build_room_size_string(width, depth, poly_area)

    if dry_run:
        return {
            "task_description": task_description,
            "layout_criteria": "[dry-run]",
            "boundary": boundary,
            "assets": {},
        }

    print(f"[llm] criteria start id={row.get('id','?')}", file=sys.stderr)
    layout_criteria = call_text_llm(
        client,
        text_model,
        pc.replace("TASK_DESCRIPTION", task_description),
        timeout_s=llm_timeout_s,
    ).strip()
    print(f"[llm] criteria done id={row.get('id','?')}", file=sys.stderr)

    fur_prompt = (
        pf.replace("TASK_DESCRIPTION", task_description)
        .replace("LAYOUT_CRITERIA", layout_criteria)
        .replace("ROOM_SIZE", room_size)
    )
    print(f"[llm] fur_list start id={row.get('id','?')}", file=sys.stderr)
    fur_raw = call_text_llm(
        client, text_model, fur_prompt, max_tokens=4096, timeout_s=llm_timeout_s
    )
    print(f"[llm] fur_list done id={row.get('id','?')}", file=sys.stderr)
    fur = parse_furniture_dict(fur_raw)

    assets = expand_assets_from_furniture_dict(
        fur,
        candidates,
        boundary,
        asset_dir,
        client,
        text_model,
        vision_model,
        task_description,
        layout_criteria,
        pj,
        judge_no_image,
        llm_timeout_s,
        max_dim_ratio,
        min_dim_ratio,
        max_total_area_ratio,
        skip_missing,
    )

    return {
        "task_description": task_description,
        "layout_criteria": layout_criteria,
        "boundary": boundary,
        "assets": assets,
    }



def default_jsonl_for_subdir(repo_root: Path, subdir: str) -> Path:
    return repo_root / "eval-dataset" / subdir / f"{subdir}.jsonl"


def load_room_entries(extracted_path: Path, room_key: str) -> List[Any]:
    with open(extracted_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rooms = (data.get("rooms") or {}).get(room_key)
    if not isinstance(rooms, list):
        raise KeyError(f"No rooms list at rooms[{room_key!r}] in {extracted_path}")
    return rooms


def exterior_from_room_entry(entry: Any, room_key: str, room_index: int) -> List[List[float]]:
    exterior = entry.get("room", {}).get("exterior")
    if not exterior:
        raise ValueError(f"Missing exterior for rooms[{room_key}][{room_index}]")
    return exterior


def load_exterior(
    extracted_path: Path, room_key: str, room_index: int
) -> List[List[float]]:
    rooms = load_room_entries(extracted_path, room_key)
    if room_index >= len(rooms):
        raise KeyError(
            f"No room at rooms[{room_key!r}][{room_index}] in {extracted_path}"
        )
    return exterior_from_room_entry(rooms[room_index], room_key, room_index)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate benchmark JSON from eval-dataset jsonl (bedroom_0 format).",
        epilog="Example: python -m eval_benchmark_gen.build_benchmark_from_eval_dataset "
        "--eval-subdir bathroom --room-index 0 --out-dir out/bench",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--jsonl-path",
        type=str,
        help="Path to a .jsonl file (e.g. eval-dataset/bathroom/bathroom.jsonl)",
    )
    g.add_argument(
        "--eval-subdir",
        type=str,
        help="Subdirectory name under eval-dataset (uses eval-dataset/<name>/<name>.jsonl)",
    )
    p.add_argument(
        "--extracted-rooms",
        type=str,
        default=str(_REPO_ROOT / "eval-dataset" / "extracted_rooms.json"),
        help="Path to extracted_rooms.json",
    )
    p.add_argument(
        "--extracted-room-key",
        type=str,
        default=None,
        help="Override key in extracted_rooms['rooms'] (default: map from --eval-subdir)",
    )
    p.add_argument(
        "--room-index",
        type=int,
        default=0,
        help=(
            "Starting room index for the first non-empty jsonl line: line k uses "
            "rooms[room_key][room_index + k]. Requires enough rooms in extracted_rooms."
        ),
    )
    p.add_argument(
        "--asset-dir",
        type=str,
        default=str(_REPO_ROOT / "objaverse_processed"),
        help="Objaverse processed root (same as main.py --asset_dir)",
    )
    p.add_argument(
        "--candidates-cache",
        type=str,
        default=None,
        help="Optional JSON cache for scanned candidates (speeds up repeated runs).",
    )
    p.add_argument("--wall-height", type=float, default=2.5)
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for JSON files")
    p.add_argument(
        "--text-model",
        type=str,
        default=os.getenv("DASHSCOPE_MINI_MODEL") or os.getenv("DASHSCOPE_TEST_MODEL") or "qwen3.6-flash",
        help="Chat model for criteria / fur-list / text-only judge",
    )
    p.add_argument(
        "--vision-model",
        type=str,
        default=os.getenv("DASHSCOPE_VISION_MODEL") or "qwen-vl-max",
        help="Vision model when preview images exist",
    )
    p.add_argument(
        "--judge-no-image",
        choices=["text", "skip"],
        default="text",
        help="If no preview image: use text judge, or skip candidate",
    )
    p.add_argument(
        "--llm-timeout-s",
        type=int,
        default=60,
        help="Per LLM call timeout in seconds (SIGALRM in the process main thread).",
    )
    p.add_argument("--max-dim-ratio", type=float, default=0.8)
    p.add_argument("--min-dim-ratio", type=float, default=0.05)
    p.add_argument("--max-total-area-ratio", type=float, default=0.6)
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip furniture types that cannot be matched or judged (omit from assets)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="If out-dir/<id>.json already exists, skip that row (no LLM); room line index still advances.",
    )
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel worker processes for LLM+I/O (default 1 = serial). Mind API rate limits.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max non-empty jsonl records to process (in file order)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write boundary + task text; no LLM / asset picking",
    )
    p.add_argument(
        "--dashscope-api-key",
        type=str,
        default=None,
        help="Optional; otherwise DASHSCOPE_API_KEY / OPENAI_API_KEY",
    )
    p.add_argument("--dashscope-base-url", type=str, default=None)
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be >= 1")
    return args


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()
    repo_root = _REPO_ROOT

    print("[start] build_benchmark_from_eval_dataset", file=sys.stderr)
    if args.jsonl_path:
        jsonl_path = Path(os.path.abspath(args.jsonl_path))
        subdir_guess = jsonl_path.parent.name
    else:
        subdir_guess = args.eval_subdir  # type: ignore
        jsonl_path = default_jsonl_for_subdir(repo_root, subdir_guess)

    if not jsonl_path.is_file():
        print(f"Error: jsonl not found: {jsonl_path}", file=sys.stderr)
        return 2

    room_key = resolve_room_key(subdir_guess, args.extracted_room_key)
    extracted_path = Path(os.path.abspath(args.extracted_rooms))

    room_entries = load_room_entries(extracted_path, room_key)
    n_rooms = len(room_entries)
    print(
        f"[rooms] key={room_key} count={n_rooms} base_index={args.room_index} from={extracted_path}",
        file=sys.stderr,
    )

    prompts = load_prompts_strings(repo_root)

    candidates: List[Candidate] = []
    if not args.dry_run:
        cache_path = Path(os.path.abspath(args.candidates_cache)) if args.candidates_cache else None
        if cache_path and cache_path.is_file():
            print(f"[candidates] loading cache {cache_path}", file=sys.stderr)
            candidates = load_candidates_cache(cache_path)
            print(f"[candidates] loaded {len(candidates)} from cache", file=sys.stderr)
        else:
            print(f"[candidates] scanning asset_dir={args.asset_dir}", file=sys.stderr)
            candidates = load_candidates(args.asset_dir)
            if cache_path:
                print(f"[candidates] saving cache {cache_path}", file=sys.stderr)
                save_candidates_cache(cache_path, candidates)

    if not candidates and not args.dry_run:
        print(f"Warning: no assets under {args.asset_dir}", file=sys.stderr)

    client: Any
    if args.dry_run:
        client = None
    else:
        client = get_dashscope_client(
            api_key=args.dashscope_api_key,
            base_url=args.dashscope_base_url,
        )

    out_dir = Path(os.path.abspath(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase A: scan jsonl, resolve skip-existing, precompute boundaries (preserves line_idx -> room mapping).
    tasks: List[GenTask] = []
    line_idx = 0
    n_skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if args.limit is not None and line_idx >= args.limit:
                break

            row = json.loads(line)
            eid = str(row.get("id", f"line_{line_idx}"))
            idx = args.room_index + line_idx
            if idx >= n_rooms:
                raise RuntimeError(
                    f"Room index out of range for jsonl id={eid!r}: need rooms[{room_key!r}][{idx}] "
                    f"but extracted_rooms only has {n_rooms} entries for this key (indices 0..{n_rooms - 1}). "
                    f"Either shorten the jsonl, increase rooms in extracted_rooms, or lower --room-index."
                )

            out_path = out_dir / f"{eid}.json"
            if args.skip_existing and out_path.is_file():
                print(f"[resume] skip existing id={eid} -> {out_path}", file=sys.stderr)
                line_idx += 1
                n_skipped += 1
                continue

            exterior = exterior_from_room_entry(room_entries[idx], room_key, idx)
            floor_vertices, width, depth, poly_area = exterior_to_floor_vertices(exterior)
            boundary = {
                "floor_vertices": floor_vertices,
                "wall_height": float(args.wall_height),
            }

            tasks.append(
                GenTask(
                    line_idx=line_idx,
                    eid=eid,
                    row=row,
                    room_idx=idx,
                    boundary=boundary,
                    poly_area=poly_area,
                    width=width,
                    depth=depth,
                    out_path=out_path,
                )
            )
            line_idx += 1

    n_written = 0
    if tasks:
        run_kw = dict(
            prompts=prompts,
            candidates=candidates,
            asset_dir=args.asset_dir,
            judge_no_image=args.judge_no_image,
            llm_timeout_s=args.llm_timeout_s,
            max_dim_ratio=args.max_dim_ratio,
            min_dim_ratio=args.min_dim_ratio,
            max_total_area_ratio=args.max_total_area_ratio,
            skip_missing=args.skip_missing,
            dry_run=args.dry_run,
            text_model=args.text_model,
            vision_model=args.vision_model,
            room_key=room_key,
        )
        if args.workers <= 1:
            for t in tasks:
                run_single_benchmark_task(t, client=client, **run_kw)
                n_written += 1
        else:
            print(
                f"[parallel] processes={args.workers} tasks={len(tasks)}",
                file=sys.stderr,
            )
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        _run_single_benchmark_task_mp,
                        t,
                        args.dashscope_api_key,
                        args.dashscope_base_url,
                        run_kw,
                    )
                    for t in tasks
                ]
                for fut in futures:
                    fut.result()
            n_written = len(tasks)

    print(
        f"[done] non_empty_handled={line_idx} written={n_written} skipped_existing={n_skipped}",
        file=sys.stderr,
    )
    if line_idx == 0:
        print("No lines processed (empty jsonl?)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
