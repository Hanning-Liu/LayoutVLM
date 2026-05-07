#!/usr/bin/env python3
"""
Batch-run LayoutVLM on scene JSON files under out/bench_* with optional multiprocessing.

Run from repository root (recommended):
  conda run -n layoutvlm python scripts/batch_run_layoutvlm.py --out_root ./out --resume

Full pipeline (layout + per-scene Blender final render + trajectory GIF), 750 scenes, 3 workers::

  conda run -n layoutvlm python -u scripts/batch_run_layoutvlm.py \\
    --out_root ./out \\
    --results_root ./results/out_batch \\
    --asset_dir ./objaverse_processed \\
    --model qwen3.6-flash \\
    --workers 3 \\
    --resume \\
    --with_image \\
    --blender_bin /path/to/blender \\
    --post_final_blender \\
    --post_trajectory \\
    --conda_env layoutvlm \\
    --limit 750

If ``conda`` is not on PATH (common under ``conda run -n layoutvlm python ...``), trajectory rendering
falls back to ``sys.executable`` (same interpreter as this batch process). Use ``--conda_exe`` to force
``conda run -n <env> python`` instead.

Credentials: same as main.py — loads repo-root `.env` via `_load_dotenv_if_present()` before resolving keys
(you can still use shell `export` or `--dashscope_api_key`).

Notes:
  - Each scene gets its own save_dir so layout.json files never overwrite each other.
  - Output dirs are created when a worker starts a scene (not all upfront). While running,
    save_dir contains `.batch_in_progress` (removed on success; left if the task errors).
  - Parallel workers increase DashScope QPS and GPU memory use; lower --workers if you
    hit rate limits or CUDA OOM. For --with_image without conda `bpy`, set LAYOUTVLM_BLENDER
    or pass --blender_bin so each worker can spawn Blender subprocess renders.
  - Use --resume for JSON-validated skip + checkpoint JSONL; progress lines print to stderr with flush.
  - First output can be delayed by heavy imports and per-scene work; run with `python -u` if logs lag under conda.
  - With --post_final_blender / --post_trajectory, post-steps run in the same worker after a successful
    ``run_single_layout``; ``--resume`` skips scenes with valid layout.json and does not enqueue post-only work.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TextIO, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Worker pickle payload: base layout job + post-render options (see _worker_mp).
WorkerJob = Tuple[
    str,
    str,
    str,
    str,
    bool,
    bool,
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    bool,
    bool,
    str,
    str,
    str,
    str,
    bool,
]


def discover_scene_jsons(out_root: Path, json_glob: str) -> List[Path]:
    scenes: List[Path] = []
    if not out_root.is_dir():
        return scenes
    for child in sorted(out_root.iterdir()):
        if child.is_dir() and child.name.startswith("bench_"):
            for p in sorted(child.glob(json_glob)):
                if p.is_file() and p.suffix.lower() == ".json":
                    scenes.append(p)
    return scenes


def scene_to_save_dir(scene_path: Path, out_root: Path, results_root: Path) -> Path:
    rel = scene_path.relative_to(out_root.resolve())
    return (results_root / rel.parent / rel.stem).resolve()


def is_valid_done_layout(save_dir: Path) -> bool:
    """True if layout.json exists, parses as JSON, and is a non-empty dict."""
    p = save_dir / "layout.json"
    if not p.is_file():
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return isinstance(data, dict) and len(data) > 0


def append_checkpoint_line(
    fh: TextIO,
    *,
    scene: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    rec = {
        "scene": scene,
        "status": status,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        rec["error"] = error[:500]
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()


def print_progress_line(
    completed: int,
    total: int,
    t0: float,
    scene_path: str,
    ok: bool,
) -> None:
    elapsed = time.monotonic() - t0
    eta_part = ""
    if completed > 0 and total > completed:
        eta = (elapsed / completed) * (total - completed)
        eta_part = f" eta_s~{eta:.0f}"
    status = "ok" if ok else "fail"
    print(
        f"[progress] {completed}/{total} {status} elapsed_s={elapsed:.1f}{eta_part} scene={scene_path}",
        file=sys.stderr,
        flush=True,
    )


def _run_post_renders(
    scene_json_file: str,
    save_dir: str,
    asset_dir: str,
    blender_bin: str,
    *,
    post_final: bool,
    post_traj: bool,
    conda_env: str,
    final_subdir: str,
    traj_subdir: str,
    conda_exe: str,
    traj_via_conda_run: bool,
) -> None:
    """Run Blender final layout render and/or trajectory GIF via subprocess (no torch import)."""
    save_path = Path(save_dir)
    layout_json = save_path / "layout.json"
    if not layout_json.is_file():
        raise RuntimeError(f"PostRenderError: missing {layout_json}")

    if post_final:
        out_final = save_path / final_subdir
        out_final.mkdir(parents=True, exist_ok=True)
        cmd = [
            blender_bin,
            "-b",
            "--python",
            str(REPO_ROOT / "scripts" / "render_result_blender.py"),
            "--",
            "--scene_json_file",
            scene_json_file,
            "--layout_json_file",
            str(layout_json),
            "--asset_dir",
            asset_dir,
            "--out_dir",
            str(out_final),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-4000:]
            raise RuntimeError(
                f"PostRenderError: render_result_blender.py exit={proc.returncode}: {tail}"
            )

    if post_traj:
        out_traj = save_path / traj_subdir
        out_traj.mkdir(parents=True, exist_ok=True)
        script = str(REPO_ROOT / "scripts" / "render_trajectory.py")
        common_tail = [
            script,
            "--scene_json_file",
            scene_json_file,
            "--results_dir",
            str(save_path),
            "--asset_dir",
            asset_dir,
            "--out_dir",
            str(out_traj),
            "--blender_bin",
            blender_bin,
        ]
        if traj_via_conda_run:
            cmd = [conda_exe, "run", "-n", conda_env, "python", "-u", *common_tail]
        else:
            cmd = [conda_exe, "-u", *common_tail]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-4000:]
            raise RuntimeError(
                f"PostRenderError: render_trajectory.py exit={proc.returncode}: {tail}"
            )


def _worker_mp(job: WorkerJob) -> Tuple[str, Optional[str]]:
    (
        scene_json_file,
        save_dir,
        model,
        asset_dir,
        with_image,
        no_image,
        dash_key,
        dash_url,
        openai_deprecated,
        blender_bin,
        post_final,
        post_traj,
        conda_env,
        final_subdir,
        traj_subdir,
        conda_exe,
        traj_via_conda_run,
    ) = job
    try:
        os.chdir(REPO_ROOT)
        if blender_bin:
            os.environ["LAYOUTVLM_BLENDER"] = blender_bin
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        marker = save_path / ".batch_in_progress"
        marker.write_text(
            f"started_utc={datetime.now(timezone.utc).isoformat()}\nscene={scene_json_file}\n",
            encoding="utf-8",
        )
        from main import apply_dashscope_env, run_single_layout

        apply_dashscope_env(dash_key, dash_url, openai_deprecated)
        run_single_layout(
            scene_json_file=scene_json_file,
            save_dir=save_dir,
            model=model,
            asset_dir=asset_dir,
            with_image=with_image,
            no_image=no_image,
            blender_bin=blender_bin,
        )
        if post_final or post_traj:
            if not blender_bin:
                raise RuntimeError("PostRenderError: --blender_bin required for post-render steps")
            _run_post_renders(
                scene_json_file,
                save_dir,
                asset_dir,
                blender_bin,
                post_final=post_final,
                post_traj=post_traj,
                conda_env=conda_env,
                final_subdir=final_subdir,
                traj_subdir=traj_subdir,
                conda_exe=conda_exe,
                traj_via_conda_run=traj_via_conda_run,
            )
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            pass
        return scene_json_file, None
    except Exception as e:
        return scene_json_file, f"{type(e).__name__}: {e}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run LayoutVLM on all JSON scenes under out_root/bench_* in parallel."
    )
    p.add_argument("--out_root", default="./out", help="Root containing bench_* scene folders")
    p.add_argument(
        "--results_root",
        default="./results/out_batch",
        help="Output root; each scene -> results_root/bench_xxx/<stem>/layout.json",
    )
    p.add_argument("--workers", type=int, default=4, help="Process pool size (default 4)")
    p.add_argument(
        "--json_glob",
        default="*.json",
        help="Glob per bench_* directory (default *.json)",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip scenes with a valid non-empty layout.json (JSON-checked)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume run: same skip rules as --skip_existing + append checkpoint JSONL on each task",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Checkpoint JSONL (only with --resume; default: results_root/.layout_batch_checkpoint.jsonl)",
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most this many scenes")
    p.add_argument("--model", default="qwen3.6-plus", help="VLM model name")
    p.add_argument("--asset_dir", default="./objaverse_processed", help="Objaverse processed root")
    img = p.add_mutually_exclusive_group()
    img.add_argument(
        "--with_image",
        action="store_true",
        help="Use Blender for floor-plan images (in-process bpy if available, else subprocess via LAYOUTVLM_BLENDER).",
    )
    img.add_argument(
        "--no_image",
        action="store_true",
        help="Force text-only mode.",
    )
    p.add_argument(
        "--blender_bin",
        default=None,
        metavar="PATH",
        help="Blender executable for --with_image subprocess mode (sets LAYOUTVLM_BLENDER in workers).",
    )
    p.add_argument(
        "--post_final_blender",
        action="store_true",
        help="After successful layout, run Blender render_result_blender.py (final layout stills).",
    )
    p.add_argument(
        "--post_trajectory",
        action="store_true",
        help=(
            "After successful layout, run render_trajectory.py (3D trajectory GIF). "
            "Uses `conda run` when conda is on PATH or `--conda_exe` is set; otherwise "
            "the current `python` (`sys.executable`), e.g. when the batch is started via `conda run`."
        ),
    )
    p.add_argument(
        "--conda_env",
        default="layoutvlm",
        help="Conda env name for --post_trajectory (default: layoutvlm).",
    )
    p.add_argument(
        "--conda_exe",
        default=None,
        metavar="PATH",
        help="conda executable (default: search PATH). Use when conda is not on PATH.",
    )
    p.add_argument(
        "--post_final_out_subdir",
        default="final_blender_render",
        help="Under each save_dir: output for render_result_blender (default: final_blender_render).",
    )
    p.add_argument(
        "--post_traj_out_subdir",
        default="trajectory_blender",
        help="Under each save_dir: output for render_trajectory (default: trajectory_blender).",
    )
    p.add_argument("--dashscope_api_key", default=None, help="Override DASHSCOPE_API_KEY")
    p.add_argument("--dashscope_base_url", default=None, help="Override DASHSCOPE_BASE_URL")
    p.add_argument("--openai_api_key", default=None, help="(deprecated) same as dashscope key")
    return p.parse_args()


def _stderr_linebuf() -> None:
    try:
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    if args.checkpoint and not args.resume:
        print("error: --checkpoint requires --resume", file=sys.stderr)
        return 2

    post_any = bool(args.post_final_blender or args.post_trajectory)
    if post_any and not args.blender_bin:
        print(
            "error: --post_final_blender / --post_trajectory require --blender_bin",
            file=sys.stderr,
        )
        return 2
    if post_any and not Path(args.blender_bin).is_file():
        print(f"error: --blender_bin is not a file: {args.blender_bin}", file=sys.stderr)
        return 2

    conda_exe_resolved = ""
    traj_via_conda_run = False
    if args.post_trajectory:
        if args.conda_exe:
            ce = args.conda_exe.strip()
            if not os.path.isfile(ce):
                print(f"error: --conda_exe is not a file: {ce}", file=sys.stderr)
                return 2
            conda_exe_resolved = ce
            traj_via_conda_run = True
        else:
            w = shutil.which("conda")
            if w and os.path.isfile(w):
                conda_exe_resolved = w
                traj_via_conda_run = True
            else:
                conda_exe_resolved = sys.executable
                traj_via_conda_run = False
                print(
                    "[batch] post_trajectory: conda not on PATH; using sys.executable for render_trajectory.py",
                    file=sys.stderr,
                    flush=True,
                )

    os.chdir(REPO_ROOT)
    if args.blender_bin:
        os.environ["LAYOUTVLM_BLENDER"] = args.blender_bin
    _stderr_linebuf()
    print(
        "[batch] startup: loading `main` (torch / LayoutVLM); first log may take 30–120s…",
        file=sys.stderr,
        flush=True,
    )

    from main import (
        _load_dotenv_if_present,
        apply_dashscope_env,
        resolve_dashscope_credentials,
    )

    _load_dotenv_if_present()

    out_root = Path(args.out_root).resolve()
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path: Optional[Path] = None
    if args.resume:
        checkpoint_path = (
            Path(args.checkpoint).resolve()
            if args.checkpoint
            else (results_root / ".layout_batch_checkpoint.jsonl")
        )

    scenes = discover_scene_jsons(out_root, args.json_glob)
    if args.limit is not None:
        scenes = scenes[: args.limit]

    if not scenes:
        print(f"No scene JSON found under {out_root} (bench_* / {args.json_glob})", file=sys.stderr)
        return 1

    dash_key, dash_url = resolve_dashscope_credentials(
        args.dashscope_api_key,
        args.dashscope_base_url,
        args.openai_api_key,
    )

    use_skip = bool(args.skip_existing or args.resume)

    jobs: List[WorkerJob] = []
    skipped = 0
    n_scenes = len(scenes)
    if use_skip and n_scenes:
        print(
            f"[batch] scanning {n_scenes} scenes for existing layout.json (resume/skip)…",
            file=sys.stderr,
            flush=True,
        )
    for idx, sp in enumerate(scenes, start=1):
        if use_skip and n_scenes and idx % 100 == 0:
            print(
                f"[batch] scan progress {idx}/{n_scenes} (skipped_done so far: {skipped})…",
                file=sys.stderr,
                flush=True,
            )
        save_dir = scene_to_save_dir(sp, out_root, results_root)
        out_json = save_dir / "layout.json"
        if use_skip:
            if is_valid_done_layout(save_dir):
                skipped += 1
                continue
            if out_json.is_file():
                print(
                    f"[resume] invalid or empty layout.json, will rerun: {out_json}",
                    file=sys.stderr,
                    flush=True,
                )
        jobs.append(
            (
                str(sp.resolve()),
                str(save_dir),
                args.model,
                args.asset_dir,
                bool(args.with_image),
                bool(args.no_image),
                dash_key,
                dash_url,
                args.openai_api_key,
                args.blender_bin,
                bool(args.post_final_blender),
                bool(args.post_trajectory),
                str(args.conda_env),
                str(args.post_final_out_subdir),
                str(args.post_traj_out_subdir),
                conda_exe_resolved,
                traj_via_conda_run,
            )
        )

    print(
        f"[batch] scenes_total={len(scenes)} to_run={len(jobs)} skipped_done={skipped} "
        f"workers={args.workers} results_root={results_root} resume={bool(args.resume)} "
        f"checkpoint={checkpoint_path or 'off'} "
        f"post_final_blender={bool(args.post_final_blender)} post_trajectory={bool(args.post_trajectory)} "
        f"traj_via_conda_run={traj_via_conda_run}",
        file=sys.stderr,
        flush=True,
    )

    failures: List[Tuple[str, str]] = []
    total = len(jobs)
    t0 = time.monotonic()
    completed = 0

    checkpoint_fh: Optional[TextIO] = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_fh = open(checkpoint_path, "a", encoding="utf-8")

    try:

        def on_done(scene_path: str, err: Optional[str]) -> None:
            nonlocal completed
            completed += 1
            ok = err is None
            print_progress_line(completed, total, t0, scene_path, ok)
            if checkpoint_fh is not None:
                append_checkpoint_line(
                    checkpoint_fh,
                    scene=scene_path,
                    status="ok" if ok else "fail",
                    error=err,
                )
            if err:
                failures.append((scene_path, err))
                print(f"[fail] {scene_path}\n        {err}", file=sys.stderr, flush=True)

        if args.workers <= 1:
            if jobs:
                print(
                    "[batch] running sequentially; first [progress] may take several minutes…",
                    file=sys.stderr,
                    flush=True,
                )
            apply_dashscope_env(
                args.dashscope_api_key,
                args.dashscope_base_url,
                args.openai_api_key,
            )
            for job in jobs:
                path, err = _worker_mp(job)
                on_done(path, err)
        else:
            print(
                f"[batch] starting {args.workers} worker processes "
                f"(each re-imports stack; first [progress] may take several minutes)…",
                file=sys.stderr,
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(_worker_mp, job) for job in jobs]
                print(
                    f"[batch] submitted {len(futures)} tasks; waiting for completions…",
                    file=sys.stderr,
                    flush=True,
                )
                for fut in as_completed(futures):
                    path, err = fut.result()
                    on_done(path, err)
    finally:
        if checkpoint_fh is not None:
            checkpoint_fh.close()

    ok = len(jobs) - len(failures)
    print(
        f"[done] ok={ok} failed={len(failures)} skipped_done={skipped}",
        file=sys.stderr,
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
