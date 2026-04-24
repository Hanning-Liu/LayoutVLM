import argparse
import collections
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_json_file", required=True)
    p.add_argument("--layout_json_file", required=True)
    p.add_argument("--asset_dir", default="./objaverse_processed")
    p.add_argument("--out_dir", default="./results/render")
    p.add_argument("--no_gif", action="store_true", help="Skip GIF stitching.")
    # Blender passes its own flags in sys.argv; user args come after `--`.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return p.parse_args(argv)


def prepare_task_assets(task: dict, asset_dir: str) -> dict:
    """
    Minimal asset materialization for Blender rendering.
    - Accepts either "<uid>-<idx>" (preferred) or bare "<uid>" (assume idx=0).
    - Looks for assets under:
        - <asset_dir>/<uid>/
        - <asset_dir>/test_asset_dir/<uid>/   (common in this repo)
    """
    if "layout_criteria" not in task:
        task["layout_criteria"] = "the layout should follow the task description and adhere to common sense"

    all_data = collections.defaultdict(list)
    for original_uid in task["assets"].keys():
        uid = "-".join(original_uid.split("-")[:-1]) if "-" in original_uid else original_uid

        candidate_roots = [
            asset_dir,
            os.path.join(asset_dir, "test_asset_dir"),
        ]
        data_path = None
        asset_root = None
        for root in candidate_roots:
            p = os.path.join(root, uid, "data.json")
            if os.path.exists(p):
                data_path = p
                asset_root = root
                break

        if data_path is None:
            print(f"Warning: Asset data not found for {uid}")
            continue

        with open(data_path, "r") as f:
            data = json.load(f)
        data["path"] = os.path.join(asset_root, uid, f"{uid}.glb")
        all_data[uid].append(data)

    # Rebuild task["assets"] into LayoutVLM's internal format expected by blender_render.py
    category_count = collections.defaultdict(int)
    for uid, duplicated_assets in all_data.items():
        category_var_name = duplicated_assets[0]["annotations"]["category"]
        category_var_name = (
            category_var_name.replace("-", "_")
            .replace(" ", "_")
            .replace("'", "_")
            .replace("/", "_")
            .replace(",", "_")
            .lower()
        )
        category_count[category_var_name] += 1

    task["assets"] = {}
    category_idx = collections.defaultdict(int)
    for uid, duplicated_assets in all_data.items():
        category_var_name = duplicated_assets[0]["annotations"]["category"]
        category_var_name = (
            category_var_name.replace("-", "_")
            .replace(" ", "_")
            .replace("'", "_")
            .replace("/", "_")
            .replace(",", "_")
            .lower()
        )
        category_idx[category_var_name] += 1

        for instance_idx, data in enumerate(duplicated_assets):
            category_var_name_i = (
                f"{category_var_name}_{chr(ord('A') + category_idx[category_var_name] - 1)}"
                if category_count[category_var_name] > 1
                else category_var_name
            )

            var_name = f"{category_var_name_i}_{instance_idx}" if len(duplicated_assets) > 1 else category_var_name_i
            task["assets"][f"{category_var_name_i}-{instance_idx}"] = {
                "uid": uid,
                "count": len(duplicated_assets),
                "instance_var_name": var_name,
                "asset_var_name": category_var_name_i,
                "instance_idx": instance_idx,
                "annotations": data["annotations"],
                "category": data["annotations"]["category"],
                "description": data["annotations"]["description"],
                "path": data["path"],
                "onCeiling": data["annotations"]["onCeiling"],
                "onFloor": data["annotations"]["onFloor"],
                "onWall": data["annotations"]["onWall"],
                "onObject": data["annotations"]["onObject"],
                "frontView": data["annotations"]["frontView"],
                "assetMetadata": {
                    "boundingBox": {
                        # Keep consistent with main.py (swap x/y)
                        "x": float(data["assetMetadata"]["boundingBox"]["y"]),
                        "y": float(data["assetMetadata"]["boundingBox"]["x"]),
                        "z": float(data["assetMetadata"]["boundingBox"]["z"]),
                    }
                },
            }

    return task


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Ensure repo root is importable when running inside Blender.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    with open(args.scene_json_file, "r") as f:
        task = json.load(f)
    task = prepare_task_assets(task, args.asset_dir)

    with open(args.layout_json_file, "r") as f:
        placed_assets = json.load(f)

    # Import only inside Blender.
    from utils.blender_render import render_existing_scene

    output_images, _ = render_existing_scene(
        placed_assets=placed_assets,
        task=task,
        save_dir=args.out_dir,
        high_res=True,
        render_top_down=True,
        add_coordinate_mark=True,
        annotate_object=True,
        annotate_wall=True,
    )

    print("Rendered:")
    for p in output_images:
        print(" -", p)

    if args.no_gif:
        return

    # Stitch frames if present (best-effort).
    try:
        import imageio.v2 as imageio

        frames = []
        for p in output_images:
            if p.lower().endswith((".png", ".jpg", ".jpeg")) and os.path.exists(p):
                frames.append(imageio.imread(p))
        if frames:
            gif_path = os.path.join(args.out_dir, "render.gif")
            imageio.mimsave(gif_path, frames, duration=0.8)
            print("Saved GIF:", gif_path)
    except Exception as e:
        print("Skipping GIF stitching:", e)


if __name__ == "__main__":
    main()

