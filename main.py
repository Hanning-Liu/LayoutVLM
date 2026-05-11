from __future__ import annotations

import os
import json
import argparse
import numpy as np
import collections
from pathlib import Path
from typing import Optional, Tuple
from src.layoutvlm.scene import Scene
from src.layoutvlm.layoutvlm import LayoutVLM
from utils.placement_utils import get_random_placement

def _load_dotenv_if_present() -> None:
    """
    Load repo-root .env into process environment if available.
    Python does not automatically read .env files; this mirrors test_dashscope_api.py.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    repo_root = Path(__file__).resolve().parent
    dotenv_path = repo_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_json_file", help="Path to scene JSON file", required=True)
    parser.add_argument("--save_dir", help="Directory to save results", default="./results/test_run")
    # DashScope (Aliyun Bailian) OpenAI-compatible endpoint (Qwen).
    # See `test_dashscope_api.py` for a minimal smoke test.
    parser.add_argument("--model", help="Model to use for layout generation", default="qwen3.6-plus")
    parser.add_argument("--dashscope_api_key", help="DashScope API key (DASHSCOPE_API_KEY).", default=None)
    parser.add_argument(
        "--dashscope_base_url",
        help="DashScope OpenAI-compatible base URL (DASHSCOPE_BASE_URL).",
        default=None,
    )
    # Backward-compatible alias (older README / scripts).
    parser.add_argument("--openai_api_key", help="(deprecated) Use --dashscope_api_key.", default=None)
    parser.add_argument("--asset_dir", help="Directory to load assets from.", default="./objaverse_processed")
    img = parser.add_mutually_exclusive_group()
    img.add_argument(
        "--with_image",
        action="store_true",
        help="Try to render floor plan images with Blender and feed them to the model.",
    )
    img.add_argument(
        "--no_image",
        action="store_true",
        help="Force text-only mode (skip Blender rendering even if available).",
    )
    parser.add_argument(
        "--blender_bin",
        default=None,
        metavar="PATH",
        help="Blender executable for --with_image subprocess rendering (sets LAYOUTVLM_BLENDER).",
    )
    parser.add_argument(
        "--layout_mode",
        default="default",
        choices=("default", "one_shot", "semantic", "finetuned"),
        help=(
            "Layout solving strategy. "
            "'default' matches legacy behavior: one_shot with images, or no_image when --no_image. "
            "'semantic' / 'finetuned' runs LLM asset grouping first (desk+chair, bed+nightstand, …), "
            "then places group by group (writes grouping.json). "
            "'one_shot' places all assets in one stage (often only group_0)."
        ),
    )
    return parser.parse_args()


def resolve_layout_mode(layout_mode: str, *, no_image: bool) -> str:
    """
    Map CLI layout_mode + flags -> LayoutVLM.mode.

    Precedence: --no_image always selects text-only ``no_image`` (still uses semantic grouping in solve()).
    """
    if no_image:
        return "no_image"
    key = (layout_mode or "default").strip().lower()
    if key in ("default", ""):
        return "one_shot"
    if key == "semantic":
        return "finetuned"
    if key in ("one_shot", "finetuned"):
        return key
    raise ValueError(f"unsupported layout_mode: {layout_mode!r}")

def prepare_task_assets(task, asset_dir):
    """
    Prepare assets for the task by processing their metadata and annotations.
    This is a minimal version that assumes assets are already downloaded and processed.
    """
    if "layout_criteria" not in task:
        task["layout_criteria"] = "the layout should follow the task description and adhere to common sense"

    all_data = collections.defaultdict(list)
    for original_uid in task["assets"].keys():
        # Accept either "<uid>-<idx>" (preferred) or bare "<uid>" (assume idx=0).
        if "-" in original_uid:
            uid = "-".join(original_uid.split("-")[:-1])
        else:
            uid = original_uid
        
        # Load asset data
        candidate_roots = [
            asset_dir,
            os.path.join(asset_dir, "test_asset_dir"),
        ]
        data_path = None
        for root in candidate_roots:
            _p = os.path.join(root, uid, "data.json")
            if os.path.exists(_p):
                data_path = _p
                asset_root = root
                break
        if data_path is None or not os.path.exists(data_path):
            print(f"Warning: Asset data not found for {uid}")
            continue
            
        with open(data_path, "r") as f:
            data = json.load(f)
        data["path"] = os.path.join(asset_root, uid, f"{uid}.glb")
        all_data[uid].append(data)

    # Process categories and create asset entries
    category_count = collections.defaultdict(int)
    for uid, duplicated_assets in all_data.items():
        category_var_name = duplicated_assets[0]['annotations']['category']
        category_var_name = category_var_name.replace('-', "_").replace(" ", "_").replace("'", "_").replace("/", "_").replace(",", "_").lower()
        category_count[category_var_name] += 1

    task["assets"] = {}
    category_idx = collections.defaultdict(int)
    
    for uid, duplicated_assets in all_data.items():
        category_var_name = duplicated_assets[0]['annotations']['category']
        category_var_name = category_var_name.replace('-', "_").replace(" ", "_").replace("'", "_").replace("/", "_").replace(",", "_").lower()
        category_idx[category_var_name] += 1
        
        for instance_idx, data in enumerate(duplicated_assets):
            # Create category name with suffix if needed
            category_var_name = f"{category_var_name}_{chr(ord('A') + category_idx[category_var_name]-1)}" if category_count[category_var_name] > 1 else category_var_name
            
            # Create instance name
            var_name = f"{category_var_name}_{instance_idx}" if len(duplicated_assets) > 1 else category_var_name
            
            # Create asset entry
            task["assets"][f"{category_var_name}-{instance_idx}"] = {
                "uid": uid,
                "count": len(duplicated_assets),
                "instance_var_name": var_name,
                "asset_var_name": category_var_name,
                "instance_idx": instance_idx,
                "annotations": data["annotations"],
                "category": data["annotations"]["category"],
                'description': data['annotations']['description'],
                'path': data['path'],
                'onCeiling': data['annotations']['onCeiling'],
                'onFloor': data['annotations']['onFloor'],
                'onWall': data['annotations']['onWall'],
                'onObject': data['annotations']['onObject'],
                'frontView': data['annotations']['frontView'],
                'assetMetadata': {
                    "boundingBox": {
                        "x": float(data['assetMetadata']['boundingBox']['y']),  # SWAP x and y
                        "y": float(data['assetMetadata']['boundingBox']['x']),
                        "z": float(data['assetMetadata']['boundingBox']['z'])
                    },
                }
            }

    return task


def resolve_dashscope_credentials(
    dashscope_api_key: Optional[str] = None,
    dashscope_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Merge CLI-style overrides with environment (same precedence as former main())."""
    key = (
        dashscope_api_key
        or os.getenv("DASHSCOPE_API_KEY")
        or openai_api_key
        or os.getenv("OPENAI_API_KEY")
    )
    url = (
        dashscope_base_url
        or os.getenv("DASHSCOPE_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )
    return key, url


def apply_dashscope_env(
    dashscope_api_key: Optional[str] = None,
    dashscope_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> None:
    """Write DashScope / OpenAI-compatible vars into os.environ for LLM clients."""
    dashscope_api_key, dashscope_base_url = resolve_dashscope_credentials(
        dashscope_api_key, dashscope_base_url, openai_api_key
    )
    if dashscope_api_key:
        os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key
        os.environ["OPENAI_API_KEY"] = dashscope_api_key
    if dashscope_base_url:
        os.environ["DASHSCOPE_BASE_URL"] = dashscope_base_url
        os.environ["OPENAI_BASE_URL"] = dashscope_base_url


def run_single_layout(
    *,
    scene_json_file: str,
    save_dir: str,
    model: str,
    asset_dir: str,
    with_image: bool = False,
    no_image: bool = False,
    blender_bin: Optional[str] = None,
    layout_mode: str = "default",
) -> str:
    """
    Run LayoutVLM for one scene JSON and write layout.json under save_dir.
    Caller should configure credentials (e.g. apply_dashscope_env) beforehand.
    Returns path to layout.json.
    """
    os.makedirs(save_dir, exist_ok=True)

    if blender_bin:
        os.environ["LAYOUTVLM_BLENDER"] = blender_bin

    with open(scene_json_file, "r") as f:
        scene_config = json.load(f)

    scene_config = prepare_task_assets(scene_config, asset_dir)

    mode = resolve_layout_mode(layout_mode, no_image=no_image)
    layout_solver = LayoutVLM(
        mode=mode,
        save_dir=save_dir,
        asset_source="objaverse",
        gpt_4o_model_name=model,
    )
    if with_image and layout_solver.mode == "no_image":
        print(
            "Note: --with_image requested but Blender rendering is not available "
            "(no bpy in this Python and no LAYOUTVLM_BLENDER / blender on PATH); "
            "falling back to no-image mode."
        )

    layout = layout_solver.solve(scene_config)

    output_path = os.path.join(save_dir, "layout.json")
    with open(output_path, "w") as f:
        json.dump(layout, f, indent=2)

    print(f"Layout generated and saved to {output_path}")
    return output_path


def main():
    _load_dotenv_if_present()
    args = parse_args()
    apply_dashscope_env(
        args.dashscope_api_key,
        args.dashscope_base_url,
        args.openai_api_key,
    )

    run_single_layout(
        scene_json_file=args.scene_json_file,
        save_dir=args.save_dir,
        model=args.model,
        asset_dir=args.asset_dir,
        with_image=args.with_image,
        no_image=args.no_image,
        blender_bin=args.blender_bin,
        layout_mode=args.layout_mode,
    )

if __name__ == "__main__":
    main() 