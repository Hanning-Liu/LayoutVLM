"""
Pillow-only image annotation (no matplotlib / cv2).

Used by conda to draw overlays on renders produced by Blender subprocess,
and by in-process bpy paths after render_existing_scene returns raw PNGs.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import ImageDraw, ImageFont


def annotate_image_with_coordinates(
    image_path,
    visual_marks,
    output_path,
    format="coordinate",
    default_font_size=18,
):
    script_path = os.path.dirname(os.path.realpath(__file__))
    img = Image.open(image_path)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)

    to_draw_list = []
    if format == "coordinate":
        assert type(visual_marks) == dict
        for (x, y), (pixel_x, pixel_y) in visual_marks.items():
            to_draw_list.append({"pixel": (pixel_x, pixel_y), "text": f"({x},{y})"})

        default_font_size = 18 * img_width / 1000
        font = ImageFont.truetype(os.path.join(script_path, "Arial.ttf"), default_font_size)

    elif format == "text":
        assert type(visual_marks) == list
        for visual_mark in visual_marks:
            assert type(visual_mark) == dict
            assert "pixel" in visual_mark and "text" in visual_mark
        to_draw_list = visual_marks

        default_font_size = default_font_size * img_width / 1000
        font = ImageFont.truetype(os.path.join(script_path, "Arial.ttf"), default_font_size)

    else:
        raise ValueError("Invalid format. Choose 'coordinate' or 'text'.")

    end_pixel_y = 0.0
    for to_draw_dict in to_draw_list:

        pixel_x, pixel_y = to_draw_dict["pixel"]
        text = to_draw_dict["text"]

        pixel_x = pixel_x * img_width
        pixel_y = pixel_y * img_height

        if "end_arrow_pixel" in to_draw_dict:
            end_pixel_x, end_pixel_y = to_draw_dict["end_arrow_pixel"]
            end_pixel_x = end_pixel_x * img_width
            end_pixel_y = end_pixel_y * img_height
            arrow_length = ((end_pixel_x - pixel_x) ** 2 + (end_pixel_y - pixel_y) ** 2) ** 0.5
            angle = math.atan2(end_pixel_y - pixel_y, end_pixel_x - pixel_x)
            draw.line([pixel_x, pixel_y, end_pixel_x, end_pixel_y], fill="black", width=5)
            arrow_head_length = min(15, arrow_length / 3)
            arrow_head_width = arrow_head_length
            x1 = end_pixel_x - arrow_head_length * math.cos(angle) + arrow_head_width * math.sin(angle)
            y1 = end_pixel_y - arrow_head_length * math.sin(angle) - arrow_head_width * math.cos(angle)
            x2 = end_pixel_x - arrow_head_length * math.cos(angle) - arrow_head_width * math.sin(angle)
            y2 = end_pixel_y - arrow_head_length * math.sin(angle) + arrow_head_width * math.cos(angle)
            draw.polygon([end_pixel_x, end_pixel_y, x1, y1, x2, y2], fill="black")

        dot_radius = 3
        draw.ellipse(
            [
                pixel_x - dot_radius,
                pixel_y - dot_radius,
                pixel_x + dot_radius,
                pixel_y + dot_radius,
            ],
            fill="red",
            outline="red",
        )
        text_bbox = draw.textbbox((pixel_x, pixel_y), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        if text.startswith("("):
            draw.text((pixel_x - text_w / 2, pixel_y + dot_radius - 2), text, font=font, fill="red")
        else:
            font_color = "black"
            if "end_arrow_pixel" in to_draw_dict:
                if end_pixel_y >= pixel_y:
                    if default_font_size > 80:
                        draw.text(
                            (pixel_x - text_w / 2, pixel_y + dot_radius - 2 - text_h * 1.3),
                            text,
                            font=font,
                            fill=font_color,
                        )
                    else:
                        draw.text(
                            (pixel_x - text_w / 2, pixel_y + dot_radius - 2 - text_h * 1.1),
                            text,
                            font=font,
                            fill=font_color,
                        )
                else:
                    draw.text((pixel_x - text_w / 2, pixel_y + dot_radius - 2), text, font=font, fill=font_color)
            else:
                draw.text((pixel_x - text_w / 2, pixel_y + dot_radius - 2), text, font=font, fill=font_color)
    img.save(output_path)
    print(f"Annotated image saved to {output_path}")


def apply_annotations(annotations: Optional[List[Dict[str, Any]]]) -> None:
    """Run annotate_image_with_coordinates for each record (same image path may appear twice)."""
    if not annotations:
        return
    for rec in annotations:
        path = rec["path"]
        fmt = rec["format"]
        items = rec["items"]
        if fmt == "coordinate":
            marks = {}
            for k, v in items.items():
                a, b = k
                px, py = v
                marks[(float(a), float(b))] = (float(px), float(py))
            annotate_image_with_coordinates(path, marks, path, format="coordinate")
        elif fmt == "text":
            norm: List[Dict[str, Any]] = []
            for m in items:
                row = dict(m)
                if "pixel" in row and isinstance(row["pixel"], list):
                    row["pixel"] = (float(row["pixel"][0]), float(row["pixel"][1]))
                if "end_arrow_pixel" in row and isinstance(row["end_arrow_pixel"], list):
                    row["end_arrow_pixel"] = (
                        float(row["end_arrow_pixel"][0]),
                        float(row["end_arrow_pixel"][1]),
                    )
                norm.append(row)
            dfs = rec.get("default_font_size")
            if dfs is None:
                annotate_image_with_coordinates(path, norm, path, format="text")
            else:
                annotate_image_with_coordinates(path, norm, path, format="text", default_font_size=float(dfs))
        else:
            raise ValueError(f"Unknown annotation format: {fmt}")
