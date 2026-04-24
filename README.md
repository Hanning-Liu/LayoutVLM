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
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"  # optional (region-specific)

conda run -n layoutvlm python main.py --scene_json_file examples/scene_living_room_sofa_tv.json --model qwen3.6-plus
# or: python main.py --scene_json_file path/to/scene.json --model qwen3.6-plus
```

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
