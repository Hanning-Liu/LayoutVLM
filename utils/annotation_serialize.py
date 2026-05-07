"""
JSON serialization for render annotation records (stdlib only).

Used by Blender subprocess scripts — must not import Pillow.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _coord_dict_to_json(items: Dict[Tuple[Any, Any], Tuple[float, float]]) -> List[List[Any]]:
    return [[[float(a), float(b)], [float(px), float(py)]] for (a, b), (px, py) in items.items()]


def _coord_dict_from_json(items: List[List[Any]]) -> Dict[Tuple[float, float], Tuple[float, float]]:
    return {(float(a), float(b)): (float(px), float(py)) for (a, b), (px, py) in items}


def serialize_annotation_for_json(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert annotation record to JSON-serializable dict."""
    out: Dict[str, Any] = {"path": rec["path"], "format": rec["format"]}
    if rec.get("default_font_size") is not None:
        out["default_font_size"] = rec["default_font_size"]
    items = rec["items"]
    if rec["format"] == "coordinate":
        out["items"] = _coord_dict_to_json(items)
    else:
        ser = []
        for m in items:
            d = dict(m)
            if "pixel" in d:
                px, py = d["pixel"]
                d["pixel"] = [float(px), float(py)]
            if "end_arrow_pixel" in d:
                ex, ey = d["end_arrow_pixel"]
                d["end_arrow_pixel"] = [float(ex), float(ey)]
            ser.append(d)
        out["items"] = ser
    return out


def deserialize_annotation_from_json(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Restore annotation record for apply_annotations."""
    out = {"path": rec["path"], "format": rec["format"], "items": None}
    if "default_font_size" in rec:
        out["default_font_size"] = rec["default_font_size"]
    fmt = rec["format"]
    raw_items = rec["items"]
    if fmt == "coordinate":
        out["items"] = _coord_dict_from_json(raw_items)
    else:
        out["items"] = raw_items
    return out
