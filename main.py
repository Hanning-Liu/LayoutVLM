import os
import json
import argparse
import numpy as np
import collections
from pathlib import Path
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
    return parser.parse_args()

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

def main():
    _load_dotenv_if_present()
    args = parse_args()
    # Prefer explicit DashScope args; fall back to env; finally allow deprecated flag.
    dashscope_api_key = (
        args.dashscope_api_key
        or os.getenv("DASHSCOPE_API_KEY")
        or args.openai_api_key
        or os.getenv("OPENAI_API_KEY")
    )
    if dashscope_api_key:
        os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key
        # LangChain OpenAI provider reads OPENAI_API_KEY by default.
        os.environ["OPENAI_API_KEY"] = dashscope_api_key

    dashscope_base_url = (
        args.dashscope_base_url
        or os.getenv("DASHSCOPE_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )
    if dashscope_base_url:
        os.environ["DASHSCOPE_BASE_URL"] = dashscope_base_url
        # OpenAI SDK + LangChain honor OPENAI_BASE_URL for compatible endpoints.
        os.environ["OPENAI_BASE_URL"] = dashscope_base_url
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load scene configuration
    with open(args.scene_json_file, 'r') as f:
        scene_config = json.load(f)
    
    # Prepare assets
    scene_config = prepare_task_assets(scene_config, args.asset_dir)
    
    # Initialize constraint solver
    mode = "one_shot"
    if args.no_image:
        mode = "no_image"
    layout_solver = LayoutVLM(
        mode=mode,
        save_dir=args.save_dir,
        asset_source="objaverse",  # Default to objaverse
        gpt_4o_model_name=args.model,
    )
    if args.with_image and layout_solver.mode == "no_image":
        print(
            "Note: --with_image requested but Blender rendering is not available; "
            "falling back to no-image mode."
        )
    
    # Generate layout
    layout = layout_solver.solve(scene_config)
    
    # Save results
    output_path = os.path.join(args.save_dir, 'layout.json')
    with open(output_path, 'w') as f:
        json.dump(layout, f, indent=2)
    
    print(f"Layout generated and saved to {output_path}")

if __name__ == "__main__":
    main() 