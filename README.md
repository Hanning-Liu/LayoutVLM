# LayoutVLM

<div align="left">
    <a href="https://ai.stanford.edu/~sunfanyun/layoutvlm"><img src="https://img.shields.io/badge/🌐 Website-Visit-orange"></a>
    <a href=""><img src="https://img.shields.io/badge/arXiv-PDF-blue"></a>
</div>

<br>

## Installation

1. Clone this repository
2. Install dependencies (python 3.10):
```bash
pip install -r requirements.txt
```
> Note: `bpy` (the Blender Python API) is **not** installable from PyPI via `pip`.
> If you need Blender functionality, install Blender (3.6+) and run with Blender’s Python
> (or ensure Blender’s bundled Python can import `bpy`).
>
> Quick sanity check (after installing Blender):
>
> ```bash
> blender -b --python-expr "import bpy; print(bpy.app.version_string)"
> ```
3. Install Rotated IOU Loss (https://github.com/lilanxiao/Rotated\_IoU)
```
cd third_party/Rotated_IoU/cuda_op
python setup.py install
````

## Data preprocessing
1. Download the dataset https://drive.google.com/file/d/1WGbj8gWn-f-BRwqPKfoY06budBzgM0pu/view?usp=sharing
2. Unzip it.

Refer to https://github.com/allenai/Holodeck and https://github.com/allenai/objathor for how we preprocess Objaverse assets.

## Usage

Use the project Python environment (recommended: conda env `layoutvlm`, e.g. `conda activate layoutvlm` or `conda run -n layoutvlm python ...`).

**Pick furniture UIDs from local metadata** (avoids random IDs that mismatch your task):

```bash
conda run -n layoutvlm python scripts/list_objaverse_assets.py --category sofa --on_floor
conda run -n layoutvlm python scripts/list_objaverse_assets.py --category "coffee table" --on_floor
conda run -n layoutvlm python scripts/list_objaverse_assets.py --category television --on_floor
```

Copy the printed `"<uid>-0": {}` lines into the `assets` section of your scene JSON. A ready-made example with sofa, coffee table, and television is at `examples/scene_living_room_sofa_tv.json`.

1. Prepare a scene configuration JSON file of Objaverse assets with the following structure:
```json
{
    "task_description": ...,
    "layout_criteria": ...,
    "boundary": {
        "floor_vertices": [[x1, y1, z1], [x2, y2, z2], ...],
        "wall_height": height
    },
    "assets": {
        "asset_id": {
            "path": "path/to/asset.glb",
            "assetMetadata": {
                "boundingBox": {
                    "x": width,
                    "y": depth,
                    "z": height
                }
            }
        }
    }
}
```

2. Run LayoutVLM:
```bash
# DashScope (Aliyun Bailian) Qwen via OpenAI-compatible endpoint
# Optional: put DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL in repo-root .env (loaded automatically via python-dotenv),
# or export them here / pass --dashscope_api_key for CI and one-off runs.
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"  # optional (region-specific)

conda run -n layoutvlm python main.py --scene_json_file examples/scene_living_room_sofa_tv.json --model qwen3.6-plus
# or: python main.py --scene_json_file path/to/scene.json --model qwen3.6-plus
```

### Image-conditioned layout from conda (no `bpy` in conda)

The solver can render intermediate top-down / side-view PNGs for the VLM. In conda, `import bpy` usually fails; LayoutVLM can instead **spawn the real Blender binary** per render via `subprocess` ([utils/blender_subprocess.py](utils/blender_subprocess.py), [scripts/render_intermediate_blender.py](scripts/render_intermediate_blender.py)).

1. Install Blender and point the repo at the executable (or put `blender` on `PATH`):

```bash
export LAYOUTVLM_BLENDER="/path/to/blender"   # e.g. .../blender-4.3.0-linux-x64/blender
# optional: cap one render subprocess (seconds)
export LAYOUTVLM_BLENDER_TIMEOUT_SEC=600
```

2. Run with `--with_image` (or batch `--with_image`). You can also pass `--blender_bin /path/to/blender` on `main.py` / `batch_run_layoutvlm.py` (sets `LAYOUTVLM_BLENDER` for that run).

**Dependencies split:** [utils/blender_render.py](utils/blender_render.py) runs inside Blender and only needs `bpy` + `numpy` (bundled with Blender). You do **not** need to `pip install` matplotlib, scipy, trimesh, or opencv into Blender for this path. The 2D overlays (coordinate grid + object/wall labels) are applied in the **conda** `layoutvlm` process via Pillow ([utils/image_annotate.py](utils/image_annotate.py)); `requirements.txt` already includes `pillow`.

**Known limitation:** [utils/blender_render.py](utils/blender_render.py) uses a hard-coded floor texture path (`/viscam/projects/SceneAug/ambientcg/...`). If that directory does not exist on your machine, Blender subprocess renders may fail until you change that path or add local assets.

**Post-hoc 3D renders** (after `layout.json` exists): run Blender on `scripts/render_result_blender.py` (see script `--help` after `--`). Optional HDRI: set `LAYOUTVLM_HDRI` to an `.exr` file, or add `data/HDRIs/studio_small_08_4k.exr` under the repo. Blender’s Python often lacks Pillow; the script writes `_render_postprocess.json`—then run **`python scripts/postprocess_blender_render.py --out_dir <same dir>`** in conda to draw overlays **on each PNG separately**. We do **not** merge top-down and side views into one `render.gif` by default; pass **`--gif-animate-views`** to Blender or postprocess only if you want that slideshow.

Batch all scene JSON files under `out/bench_*` (default 4 parallel workers). Use **`--resume`** to skip scenes that already have a valid `layout.json`, append a checkpoint JSONL for auditing, and stream **`[progress] N/total ...`** lines to stderr. `--skip_existing` uses the same JSON validation without writing a checkpoint. Per-scene result folders are created only when a worker starts that scene (not all at once); while a scene runs, its folder contains **`.batch_in_progress`** (removed on success).

```bash
# Prefer unbuffered logs under conda: `python -u` or `PYTHONUNBUFFERED=1`.
conda run -n layoutvlm python -u scripts/batch_run_layoutvlm.py --out_root ./out --results_root ./results/out_batch --model qwen3.6-plus --workers 4 --resume
# With image-conditioned layout + explicit Blender binary:
conda run -n layoutvlm python -u scripts/batch_run_layoutvlm.py --out_root ./out --results_root ./results/out_batch --model qwen3.6-plus --workers 1 --resume --with_image --blender_bin /path/to/blender
# Same batch, but after each successful layout run final Blender stills + trajectory GIF (3 workers, first 750 scenes).
# Trajectory step uses `conda run` when `conda` is on PATH; if not (typical under `conda run ... python`), it uses the same Python as this process.
# Optional: `--conda_exe /path/to/conda` to force `conda run -n <env>`.
conda run -n layoutvlm python -u scripts/batch_run_layoutvlm.py \
  --out_root ./out --results_root ./results/out_batch --asset_dir ./objaverse_processed \
  --model qwen3.6-flash --workers 3 --resume --with_image --blender_bin /path/to/blender \
  --post_final_blender --post_trajectory --conda_env layoutvlm --limit 750
```

### 3D trajectory GIF (aligned with `final.gif`)

The 2D `final.gif` is stitched from each group’s `out.gif`, which comes from [src/layoutvlm/grad_solver.py](src/layoutvlm/grad_solver.py) snapshots in `group_<k>/temp_0/saved_intermediate_states.json`. To get a **matching Blender top-down** animation (same frame cadence), run [scripts/render_trajectory.py](scripts/render_trajectory.py) after a solve has produced those JSON files:

```bash
conda run -n layoutvlm python scripts/render_trajectory.py \
  --scene_json_file ./out/bench_balcony/balcony_001.json \
  --results_dir ./results/out_batch/bench_balcony/balcony_001 \
  --asset_dir ./objaverse_processed \
  --out_dir ./results/out_batch/bench_balcony/balcony_001/trajectory_blender \
  --blender_bin /path/to/blender
```

This invokes [scripts/render_trajectory_blender.py](scripts/render_trajectory_blender.py) **once**: each GLB is imported once, then every snapshot frame updates poses and renders. Trajectory renders use **opaque RGB** (`film_transparent=False`) so GIF viewers do not composite transparent PNGs over the previous frame (ghosting). [scripts/render_trajectory.py](scripts/render_trajectory.py) also flattens RGBA to RGB and sets GIF **disposal=2** when using Pillow. Outputs: `group_<k>/frames/frame_XXXX.png`, `group_<k>/out_blender.gif`, and `final_blender.gif`. Optional: `LAYOUTVLM_HDRI` for lighting; `LAYOUTVLM_TRAJECTORY_TIMEOUT_SEC` (default 7200) for long runs. Expect on the order of **several seconds per frame** with default Cycles settings (~40+ frames per group).

Shared helpers live in [utils/blender_render.py](utils/blender_render.py): `_compute_room_from_task`, `_setup_floor_mesh`, `_import_placed_asset_object` (also used by `render_existing_scene`).

## Output
The script will generate a layout.json file in the specified save directory containing the optimized positions and orientations of all assets in the scene.

## BibTeX
```bibtex
@inproceedings{sun2025layoutvlm,
  title={Layoutvlm: Differentiable optimization of 3d layout via vision-language models},
  author={Sun, Fan-Yun and Liu, Weiyu and Gu, Siyi and Lim, Dylan and Bhat, Goutam and Tombari, Federico and Li, Manling and Haber, Nick and Wu, Jiajun},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={29469--29478},
  year={2025}
}
```
