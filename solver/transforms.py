"""All 18 known ARC-AGI-2 transformations, reimplemented from Hone source.

Each function takes (grid, params) and returns a new grid.
This is the exact set of transforms used by the Hone subnet generator.
"""

from typing import List, Dict, Optional, Tuple, Set
from solver.grid_utils import Grid, deep_copy, dims, colors_in


# ============= GEOMETRIC =============

def rotate_90(grid: Grid, params: Optional[Dict] = None) -> Grid:
    if not grid:
        return grid
    h, w = dims(grid)
    result = [[0] * h for _ in range(w)]
    for i in range(h):
        for j in range(w):
            result[j][h - 1 - i] = grid[i][j]
    return result


def rotate_180(grid: Grid, params: Optional[Dict] = None) -> Grid:
    if not grid:
        return grid
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: Grid, params: Optional[Dict] = None) -> Grid:
    return rotate_90(rotate_180(grid))


def flip_horizontal(grid: Grid, params: Optional[Dict] = None) -> Grid:
    return [row[::-1] for row in grid]


def flip_vertical(grid: Grid, params: Optional[Dict] = None) -> Grid:
    return grid[::-1]


def transpose(grid: Grid, params: Optional[Dict] = None) -> Grid:
    if not grid:
        return grid
    h, w = dims(grid)
    return [[grid[i][j] for i in range(h)] for j in range(w)]


def flip_diagonal(grid: Grid, params: Optional[Dict] = None) -> Grid:
    return transpose(grid)


def flip_antidiagonal(grid: Grid, params: Optional[Dict] = None) -> Grid:
    return rotate_90(flip_vertical(grid))


# ============= SPATIAL =============

def shift(grid: Grid, params: Optional[Dict] = None) -> Grid:
    h, w = dims(grid)
    if params is None:
        return grid
    direction = params.get("direction", "right")
    amount = int(params.get("amount", 1))
    amount = max(0, amount)
    result = [[0] * w for _ in range(h)]

    if direction == "up":
        if amount < h:
            for r in range(amount, h):
                result[r - amount] = grid[r][:]
    elif direction == "down":
        if amount < h:
            for r in range(h - amount):
                result[r + amount] = grid[r][:]
    elif direction == "left":
        if amount < w:
            for r in range(h):
                result[r][:w - amount] = grid[r][amount:]
    elif direction == "right":
        if amount < w:
            for r in range(h):
                result[r][amount:] = grid[r][:w - amount]
    return result


def recenter(grid: Grid, params: Optional[Dict] = None) -> Grid:
    h, w = dims(grid)
    min_r, max_r, min_c, max_c = h, -1, w, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if max_r < 0:
        return grid
    content_h = max_r - min_r + 1
    content_w = max_c - min_c + 1
    result = [[0] * w for _ in range(h)]
    start_r = (h - content_h) // 2
    start_c = (w - content_w) // 2
    for r in range(content_h):
        for c in range(content_w):
            if start_r + r < h and start_c + c < w:
                result[start_r + r][start_c + c] = grid[min_r + r][min_c + c]
    return result


# ============= SCALE =============

def zoom_2x(grid: Grid, params: Optional[Dict] = None) -> Grid:
    h, w = dims(grid)
    result = [[0] * (w * 2) for _ in range(h * 2)]
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            result[r * 2][c * 2] = color
            result[r * 2][c * 2 + 1] = color
            result[r * 2 + 1][c * 2] = color
            result[r * 2 + 1][c * 2 + 1] = color
    return result


def zoom_3x(grid: Grid, params: Optional[Dict] = None) -> Grid:
    h, w = dims(grid)
    result = [[0] * (w * 3) for _ in range(h * 3)]
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            for dr in range(3):
                for dc in range(3):
                    result[r * 3 + dr][c * 3 + dc] = color
    return result


def downsample_2x(grid: Grid, params: Optional[Dict] = None) -> Grid:
    h, w = dims(grid)
    if h < 2 or w < 2:
        return grid
    return [
        [grid[r * 2][c * 2] for c in range(w // 2)]
        for r in range(h // 2) if r * 2 < h
    ]


# ============= COLOR =============

def swap_colors(grid: Grid, params: Optional[Dict] = None) -> Grid:
    if params is None:
        return grid
    c1 = params.get("color1")
    c2 = params.get("color2")
    if c1 is None or c2 is None:
        return grid
    result = deep_copy(grid)
    for r in range(len(result)):
        for c in range(len(result[0])):
            if result[r][c] == c1:
                result[r][c] = c2
            elif result[r][c] == c2:
                result[r][c] = c1
    return result


def remove_color(grid: Grid, params: Optional[Dict] = None) -> Grid:
    if params is None:
        return grid
    color = params.get("color")
    if color is None:
        return grid
    return [[0 if v == color else v for v in row] for row in grid]


def highlight_color(grid: Grid, params: Optional[Dict] = None) -> Grid:
    if params is None:
        return grid
    hl = params.get("color")
    if hl is None:
        return grid
    return [[v if v == hl or v == 0 else 5 for v in row] for row in grid]


# ============= GRAVITY =============

def gravity_down(grid: Grid, params: Optional[Dict] = None) -> Grid:
    h, w = dims(grid)
    result = [[0] * w for _ in range(h)]
    for c in range(w):
        wp = h - 1
        for r in range(h - 1, -1, -1):
            if grid[r][c] != 0:
                result[wp][c] = grid[r][c]
                wp -= 1
    return result


def gravity_up(grid: Grid, params: Optional[Dict] = None) -> Grid:
    h, w = dims(grid)
    result = [[0] * w for _ in range(h)]
    for c in range(w):
        wp = 0
        for r in range(h):
            if grid[r][c] != 0:
                result[wp][c] = grid[r][c]
                wp += 1
    return result


def gravity_left(grid: Grid, params: Optional[Dict] = None) -> Grid:
    h, w = dims(grid)
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        wp = 0
        for c in range(w):
            if grid[r][c] != 0:
                result[r][wp] = grid[r][c]
                wp += 1
    return result


def gravity_right(grid: Grid, params: Optional[Dict] = None) -> Grid:
    h, w = dims(grid)
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        wp = w - 1
        for c in range(w - 1, -1, -1):
            if grid[r][c] != 0:
                result[r][wp] = grid[r][c]
                wp -= 1
    return result


# ============= REGISTRY =============

# All transforms used by the Hone generator, keyed by their exact names
TRANSFORMS = {
    "rotate_180": rotate_180,
    "rotate_270": rotate_270,
    "transpose": transpose,
    "flip_diagonal": flip_diagonal,
    "flip_antidiagonal": flip_antidiagonal,
    "shift": shift,
    "recenter": recenter,
    "zoom_2x": zoom_2x,
    "zoom_3x": zoom_3x,
    "downsample_2x": downsample_2x,
    "swap_colors": swap_colors,
    "remove_color": remove_color,
    "highlight_color": highlight_color,
    "gravity_down": gravity_down,
    "gravity_up": gravity_up,
    "gravity_left": gravity_left,
    "gravity_right": gravity_right,
}

# Additional transforms not in the generator registry but used internally
EXTRA_TRANSFORMS = {
    "rotate_90": rotate_90,
    "flip_horizontal": flip_horizontal,
    "flip_vertical": flip_vertical,
}

ALL_TRANSFORMS = {**TRANSFORMS, **EXTRA_TRANSFORMS}

# Parameterless transforms (no params needed to apply)
PARAMETERLESS = {
    "rotate_90", "rotate_180", "rotate_270",
    "flip_horizontal", "flip_vertical",
    "transpose", "flip_diagonal", "flip_antidiagonal",
    "recenter",
    "zoom_2x", "zoom_3x", "downsample_2x",
    "gravity_down", "gravity_up", "gravity_left", "gravity_right",
}

# Transforms that preserve grid dimensions
SIZE_PRESERVING = {
    "rotate_180",
    "shift", "recenter",
    "swap_colors", "remove_color", "highlight_color",
    "gravity_down", "gravity_up", "gravity_left", "gravity_right",
}

# Transforms that change dimensions
SIZE_CHANGING = {
    "rotate_270", "transpose", "flip_diagonal", "flip_antidiagonal",
    "zoom_2x", "zoom_3x", "downsample_2x",
    "rotate_90", "flip_horizontal", "flip_vertical",
}
