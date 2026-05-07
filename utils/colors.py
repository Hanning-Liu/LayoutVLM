import numpy as np

# `matplotlib` is often unavailable inside Blender's bundled Python.
# Keep rendering working by making it optional and providing small built-in palettes.
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


_TAB10 = [
    (0.1216, 0.4667, 0.7059, 1.0),
    (1.0000, 0.4980, 0.0549, 1.0),
    (0.1725, 0.6275, 0.1725, 1.0),
    (0.8392, 0.1529, 0.1569, 1.0),
    (0.5804, 0.4039, 0.7412, 1.0),
    (0.5490, 0.3373, 0.2941, 1.0),
    (0.8902, 0.4667, 0.7608, 1.0),
    (0.4980, 0.4980, 0.4980, 1.0),
    (0.7373, 0.7412, 0.1333, 1.0),
    (0.0902, 0.7451, 0.8118, 1.0),
]

_TAB20 = [
    (0.1216, 0.4667, 0.7059, 1.0),
    (0.6824, 0.7804, 0.9098, 1.0),
    (1.0000, 0.4980, 0.0549, 1.0),
    (1.0000, 0.7333, 0.4706, 1.0),
    (0.1725, 0.6275, 0.1725, 1.0),
    (0.5961, 0.8745, 0.5412, 1.0),
    (0.8392, 0.1529, 0.1569, 1.0),
    (1.0000, 0.5961, 0.5882, 1.0),
    (0.5804, 0.4039, 0.7412, 1.0),
    (0.7725, 0.6902, 0.8353, 1.0),
    (0.5490, 0.3373, 0.2941, 1.0),
    (0.7686, 0.6118, 0.5804, 1.0),
    (0.8902, 0.4667, 0.7608, 1.0),
    (0.9686, 0.7137, 0.8235, 1.0),
    (0.4980, 0.4980, 0.4980, 1.0),
    (0.7804, 0.7804, 0.7804, 1.0),
    (0.7373, 0.7412, 0.1333, 1.0),
    (0.8588, 0.8588, 0.5529, 1.0),
    (0.0902, 0.7451, 0.8118, 1.0),
    (0.6196, 0.8549, 0.8980, 1.0),
]


def _sample_palette(palette, n: int):
    if n <= len(palette):
        return list(palette[:n])
    # Repeat deterministically if more categories than palette size.
    out = []
    for i in range(n):
        out.append(palette[i % len(palette)])
    return out


def convert_color_range(color, from_range='0-1', to_range='0-255'):
    """
    Convert the color range from [0, 1] to [0, 255] or vice versa
    :param color: list or tuple of RGB values
    :param from_range: '0-1' or '0-255'
    :param to_range: '0-1' or '0-255'
    :return: list of RGB values
    """
    if from_range == '0-1' and to_range == '0-255':
        return [int(c * 255) for c in color]
    elif from_range == '0-255' and to_range == '0-1':
        return [c / 255 for c in color]
    elif from_range == to_range:
        return color
    else:
        raise ValueError(f"Conversion from {from_range} to {to_range} not supported")


def convert_color_format(color, from_format="rgba", to_format="rgb", alpha_value=1.0):
    if from_format == "rgba" and to_format == "rgb":
        return color[:3]
    elif from_format == "rgb" and to_format == "rgba":
        color = list(color)
        color.append(alpha_value)
        return color
    elif from_format == "rgb" and to_format == "bgr":
        return color[::-1]
    elif from_format == "bgr" and to_format == "rgb":
        return color[::-1]
    elif from_format == to_format:
        return color
    else:
        raise ValueError(f"Conversion from {from_format} to {to_format} not supported")


def get_categorical_colors(num_categories: int, colormap_name='tab10', color_range='0-255', color_format='rgb'):

    if plt is None:
        # Minimal built-in palettes for Blender subprocess rendering.
        if colormap_name == "tab10":
            colors = _sample_palette(_TAB10, num_categories)
        elif colormap_name == "tab20":
            colors = _sample_palette(_TAB20, num_categories)
        else:
            # Fall back to tab20-like palette if matplotlib isn't available.
            colors = _sample_palette(_TAB20, num_categories)
    else:
        # Get the colormap
        if colormap_name == 'tab10':
            colormap = plt.cm.tab10
        elif colormap_name == 'tab20':
            colormap = plt.cm.tab20
        elif colormap_name == "viridis":
            colormap = plt.cm.viridis
        elif colormap_name == "jet":
            colormap = plt.cm.jet
        else:
            raise ValueError(f"colormap {colormap_name} not supported")

        # Normalize values to the range [0, 1] for colormap
        norm = plt.Normalize(0, num_categories - 1)

        # Assign colors to each value
        colors = [colormap(norm(value)) for value in range(num_categories)]

    # Convert colors to the desired range
    colors = [convert_color_range(color, from_range='0-1', to_range=color_range) for color in colors]
    colors = [convert_color_format(color, from_format='rgba', to_format=color_format) for color in colors]

    if colormap_name == "tab20" and num_categories == 20:
        colors = colors[::2] + colors[1::2]

    return colors
