"""Grid utility functions for ARC-AGI-2 solving."""

from typing import List, Tuple, Set


Grid = List[List[int]]


def deep_copy(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def dims(grid: Grid) -> Tuple[int, int]:
    if not grid:
        return 0, 0
    return len(grid), len(grid[0])


def is_valid(grid: Grid) -> bool:
    if not grid or not grid[0]:
        return False
    w = len(grid[0])
    for row in grid:
        if len(row) != w:
            return False
        for v in row:
            if not isinstance(v, int) or v < 0 or v > 9:
                return False
    h = len(grid)
    return 1 <= h <= 30 and 1 <= w <= 30


def colors_in(grid: Grid) -> Set[int]:
    c = set()
    for row in grid:
        c.update(row)
    return c


def non_black_count(grid: Grid) -> int:
    return sum(1 for row in grid for v in row if v != 0)


def grids_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if ra != rb:
            return False
    return True


def flatten(grid: Grid) -> List[int]:
    return [v for row in grid for v in row]


def density(grid: Grid) -> float:
    h, w = dims(grid)
    if h == 0 or w == 0:
        return 0.0
    return non_black_count(grid) / (h * w)


def bounding_box(grid: Grid) -> Tuple[int, int, int, int]:
    """Return (min_r, min_c, max_r, max_c) of non-black pixels, or (-1,-1,-1,-1) if all black."""
    h, w = dims(grid)
    min_r, min_c = h, w
    max_r, max_c = -1, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                min_r = min(min_r, r)
                min_c = min(min_c, c)
                max_r = max(max_r, r)
                max_c = max(max_c, c)
    if max_r == -1:
        return -1, -1, -1, -1
    return min_r, min_c, max_r, max_c
