import argparse
import json
import sys
import traceback

from utils.blender_render import render_existing_scene


def _parse_args():
    raw_args = sys.argv
    blender_args = raw_args[raw_args.index("--") + 1:] if "--" in raw_args else []
    parser = argparse.ArgumentParser(description="Run LayoutVLM render in Blender subprocess.")
    parser.add_argument("--payload", required=True, help="Path to input payload JSON.")
    parser.add_argument("--output", required=True, help="Path to output JSON.")
    return parser.parse_args(blender_args)


def main():
    args = _parse_args()
    with open(args.payload, "r") as payload_file:
        payload = json.load(payload_file)

    render_kwargs = payload["render_kwargs"]
    try:
        output_images, final_visual_marks, annotation_payload = render_existing_scene(**render_kwargs)
    except Exception:
        traceback.print_exc()
        raise

    result = {
        "output_images": output_images,
        "final_visual_marks": final_visual_marks,
        "annotation_payload": annotation_payload,
        "topdown_render_path": render_kwargs.get("topdown_save_file") or f"{render_kwargs['save_dir']}/top_down_rendering.png",
    }
    with open(args.output, "w") as output_file:
        json.dump(result, output_file)


if __name__ == "__main__":
    main()
