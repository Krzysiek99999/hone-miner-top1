"""Detect and strip known transformation chains from ARC-AGI-2 outputs.

Strategy: analyze train outputs for structural patterns that reveal
which known transforms were applied. Then strip them to simplify
the problem for the LLM, or apply them to the LLM's answer.

The Hone generator applies a chain of transforms to base_task outputs:
  final_output = Tn(...T2(T1(base_task_output)))

We detect transforms from the END of the chain backwards (outermost first),
because those are most visible in the final output structure.
"""

from typing import List, Dict, Optional, Tuple, Any
from solver.grid_utils import Grid, dims, grids_equal, colors_in, deep_copy, is_valid, non_black_count
from solver import transforms as T


# ============= PUBLIC API =============

def apply_chain(grid: Grid, chain: List[Dict[str, Any]]) -> Grid:
    """Apply a transformation chain to a grid."""
    result = deep_copy(grid)
    for step in chain:
        func = T.ALL_TRANSFORMS.get(step["name"])
        if func is None:
            return result
        result = func(result, step.get("params"))
        if not is_valid(result):
            return grid
    return result


def try_direct_transforms(
    train_inputs: List[Grid],
    train_outputs: List[Grid],
    test_input: Grid,
) -> Optional[Grid]:
    """
    Check if the FULL input→output mapping is a single known transform
    or a short chain of parameterless transforms.

    Only returns a result if confident (verified on ALL training examples).
    """
    if not train_inputs or not train_outputs:
        return None

    # Require at least 2 training examples for confidence
    if len(train_inputs) < 2:
        return None

    # Try identity: output == input
    if all(grids_equal(inp, out) for inp, out in zip(train_inputs, train_outputs)):
        return deep_copy(test_input)

    # Try single transforms (parameterless + parameterized)
    result = _try_single_transforms(train_inputs, train_outputs, test_input)
    if result is not None:
        return result

    # Try pairs of parameterless transforms (dimension-filtered)
    result = _try_double_transforms(train_inputs, train_outputs, test_input)
    if result is not None:
        return result

    # Try parameterless + parameterized combos
    result = _try_parameterless_then_parameterized(train_inputs, train_outputs, test_input)
    if result is not None:
        return result

    # Try chains of 3-5 parameterless transforms (dimension-filtered, fast)
    result = _try_chain_n(train_inputs, train_outputs, test_input, max_depth=5)
    if result is not None:
        return result

    return None


def detect_output_chain(
    train_inputs: List[Grid],
    train_outputs: List[Grid],
) -> List[Dict[str, Any]]:
    """
    Detect transforms applied to outputs by structural analysis.
    Only returns verified detections.
    """
    chain = []
    current_outputs = [deep_copy(o) for o in train_outputs]

    found = _detect_output_transform_validated(train_inputs, current_outputs)
    if found is not None:
        name, params, stripped = found
        chain.insert(0, {"name": name, "params": params})

    return chain


def try_zoom_wrapped_transforms(
    train_inputs: List[Grid],
    train_outputs: List[Grid],
    test_input: Grid,
) -> Optional[Grid]:
    """
    Try to solve the problem by stripping known transforms from outputs.

    Strategy layers (each verified on ALL training examples):
    1. Zoom stripping: strip zoom_2x/zoom_3x, solve inner with direct transforms
    2. Zoom + parameterized: zoom stripped + swap_colors/remove_color/etc
    3. Self-inverse wrapping: strip a self-inverse transform (flip, rotate_180, etc)
    4. Inverse-pair wrapping: strip rotate_90 via rotate_270, etc
    5. Double wrapping: strip TWO transforms from outputs (outer then inner)
    6. Forward-compose with non-invertible: try outer(direct(input)) == output
    7. Parameterized outer wrappers: detect swap_colors/remove_color/highlight_color
       as outer wrapper, strip from outputs, solve inner, re-apply
    """
    # === 1. Zoom stripping ===
    for zoom_name, strip_fn, apply_fn in [
        ("zoom_3x", _try_strip_zoom_3x, T.zoom_3x),
        ("zoom_2x", _try_strip_zoom_2x, T.zoom_2x),
    ]:
        stripped = strip_fn(train_outputs)
        if stripped is None:
            continue

        inner_result = try_direct_transforms(train_inputs, stripped, test_input)
        if inner_result is not None:
            final = apply_fn(inner_result)
            if is_valid(final):
                return final

        # Also try double-zoom: zoom_2x(zoom_2x(...))
        if zoom_name == "zoom_2x":
            stripped2 = _try_strip_zoom_2x(stripped)
            if stripped2 is not None:
                inner = try_direct_transforms(train_inputs, stripped2, test_input)
                if inner is not None:
                    final = apply_fn(apply_fn(inner))
                    if is_valid(final):
                        return final

    # === 1b. Downsample stripping ===
    # If outputs are half the size of inputs, maybe downsample_2x was the last step
    if train_inputs and train_outputs:
        ih0, iw0 = dims(train_inputs[0])
        oh0, ow0 = dims(train_outputs[0])
        if oh0 * 2 == ih0 and ow0 * 2 == iw0:
            # Check if downsample_2x(something) == output
            # We can't "undo" downsample, but we can try forward:
            # Does downsample_2x(direct(input)) == output for all examples?
            for name in T.PARAMETERLESS:
                if name in ("downsample_2x", "zoom_2x", "zoom_3x"):
                    continue  # Skip size-changing transforms
                func = T.ALL_TRANSFORMS[name]
                try:
                    if all(
                        grids_equal(T.downsample_2x(func(inp)), out)
                        for inp, out in zip(train_inputs, train_outputs)
                    ):
                        result = T.downsample_2x(func(test_input))
                        if is_valid(result):
                            return result
                except Exception:
                    continue

    # === 2. Zoom + parameterized transforms ===
    for zoom_name, strip_fn, apply_fn in [
        ("zoom_3x", _try_strip_zoom_3x, T.zoom_3x),
        ("zoom_2x", _try_strip_zoom_2x, T.zoom_2x),
    ]:
        stripped = strip_fn(train_outputs)
        if stripped is None:
            continue
        for try_fn in [_try_swap_colors, _try_remove_color, _try_highlight_color, _try_shift]:
            result = try_fn(train_inputs, stripped, test_input)
            if result is not None:
                final = apply_fn(result)
                if is_valid(final):
                    return final

    # === 3 & 4. Single wrapping: self-inverse and inverse-pair transforms ===
    self_inverse = [
        ("rotate_180", T.rotate_180, T.rotate_180),
        ("flip_horizontal", T.flip_horizontal, T.flip_horizontal),
        ("flip_vertical", T.flip_vertical, T.flip_vertical),
        ("transpose", T.transpose, T.transpose),
        ("flip_diagonal", T.flip_diagonal, T.flip_diagonal),
        ("flip_antidiagonal", T.flip_antidiagonal, T.flip_antidiagonal),
    ]
    inverse_pairs = [
        ("rotate_90", T.rotate_270, T.rotate_90),    # strip by rotate_270
        ("rotate_270", T.rotate_90, T.rotate_270),    # strip by rotate_90
        ("gravity_down", T.gravity_down, T.gravity_down),
        ("gravity_up", T.gravity_up, T.gravity_up),
        ("gravity_left", T.gravity_left, T.gravity_left),
        ("gravity_right", T.gravity_right, T.gravity_right),
    ]

    all_single_wrappers = self_inverse + inverse_pairs

    for wrap_name, strip_fn, apply_fn in all_single_wrappers:
        stripped = [strip_fn(out) for out in train_outputs]
        if not all(is_valid(s) for s in stripped):
            continue

        inner = try_direct_transforms(train_inputs, stripped, test_input)
        if inner is not None:
            final = apply_fn(inner)
            if is_valid(final):
                return final

    # === 5. Double wrapping: strip TWO transforms from outputs ===
    # If output = T_outer(T_inner(direct(input))), we strip T_outer first, then T_inner.
    # We then solve: inputs -> double_stripped = direct transform
    # And re-apply: T_inner(T_outer(direct(test_input)))... wait, order matters.
    # output = T_outer(T_inner(direct(input)))
    # strip_outer(output) = T_inner(direct(input))
    # strip_inner(strip_outer(output)) = direct(input)
    # So: result = T_outer(T_inner(direct(test_input)))

    # Use a focused set: self-inverse transforms and inverse pairs for outer/inner
    # to keep runtime manageable
    _double_wrap_candidates = [
        ("rotate_180", T.rotate_180, T.rotate_180),
        ("flip_horizontal", T.flip_horizontal, T.flip_horizontal),
        ("flip_vertical", T.flip_vertical, T.flip_vertical),
        ("transpose", T.transpose, T.transpose),
        ("rotate_90", T.rotate_270, T.rotate_90),
        ("rotate_270", T.rotate_90, T.rotate_270),
        ("gravity_down", T.gravity_down, T.gravity_down),
        ("gravity_up", T.gravity_up, T.gravity_up),
        ("gravity_left", T.gravity_left, T.gravity_left),
        ("gravity_right", T.gravity_right, T.gravity_right),
    ]

    for outer_name, outer_strip, outer_apply in _double_wrap_candidates:
        outer_stripped = [outer_strip(out) for out in train_outputs]
        if not all(is_valid(s) for s in outer_stripped):
            continue

        for inner_name, inner_strip, inner_apply in _double_wrap_candidates:
            # Skip if both are the same self-inverse (would cancel out to identity)
            if outer_name == inner_name and outer_strip == inner_apply:
                continue

            inner_stripped = [inner_strip(s) for s in outer_stripped]
            if not all(is_valid(s) for s in inner_stripped):
                continue

            inner = try_direct_transforms(train_inputs, inner_stripped, test_input)
            if inner is not None:
                # Re-apply: inner first, then outer
                mid = inner_apply(inner)
                if is_valid(mid):
                    final = outer_apply(mid)
                    if is_valid(final):
                        return final

    # === 6. Forward-compose with non-invertible transforms ===
    # For transforms that cannot be inverted (gravity, recenter), we try the
    # forward approach: does outer(direct(input)) == output for all examples?
    result = _try_forward_compose(train_inputs, train_outputs, test_input)
    if result is not None:
        return result

    # === 7. Parameterized outer wrappers ===
    # Detect if a parameterized transform (swap_colors, remove_color,
    # highlight_color) was applied as the outermost wrapper, strip it from
    # outputs, solve the inner problem, then re-apply.
    result = _try_parameterized_outer_wrapper(train_inputs, train_outputs, test_input)
    if result is not None:
        return result

    return None


def _try_forward_compose(
    train_inputs: List[Grid],
    train_outputs: List[Grid],
    test_input: Grid,
) -> Optional[Grid]:
    """
    Try: does outer_transform(direct_transform(input)) == output for all examples?
    """
    if not train_inputs or not train_outputs:
        return None

    outer_transforms = [
        ("gravity_down", T.gravity_down),
        ("gravity_up", T.gravity_up),
        ("gravity_left", T.gravity_left),
        ("gravity_right", T.gravity_right),
        ("recenter", T.recenter),
    ]

    # Try outer(single_parameterless(input)) == output
    for outer_name, outer_fn in outer_transforms:
        for inner_name in T.PARAMETERLESS:
            inner_fn = T.ALL_TRANSFORMS[inner_name]
            # Skip if outer == inner (already tried in single/double transforms)
            if outer_name == inner_name:
                continue
            try:
                # Quick check on first example
                mid0 = inner_fn(train_inputs[0])
                cand0 = outer_fn(mid0)
                if not grids_equal(cand0, train_outputs[0]):
                    continue

                # Verify on ALL examples
                if all(
                    grids_equal(outer_fn(inner_fn(inp)), out)
                    for inp, out in zip(train_inputs, train_outputs)
                ):
                    result = outer_fn(inner_fn(test_input))
                    if is_valid(result):
                        return result
            except Exception:
                continue

    # NOTE: outer(parameterized(input)) == output is not attempted here
    # because outer is non-invertible, so we cannot derive parameter candidates
    # by comparing stripped outputs to inputs. The parameterized outer wrapper
    # approach (section 7) handles the invertible parameterized case instead.

    return None


def _try_parameterized_outer_wrapper(
    train_inputs: List[Grid],
    train_outputs: List[Grid],
    test_input: Grid,
) -> Optional[Grid]:
    """
    Detect if a parameterized transform was applied as the outermost wrapper.

    For each candidate parameterized wrapper:
    1. Detect its parameters from training data (comparing outputs pairwise)
    2. Strip it from all outputs (apply inverse)
    3. Try to solve the inner problem with direct transforms
    4. Re-apply the wrapper to the solution

    Handles: swap_colors (self-inverse), remove_color (detectable but not
    cleanly invertible), highlight_color (detectable but not invertible).
    """
    if not train_inputs or not train_outputs:
        return None

    # --- swap_colors as outer wrapper ---
    result = _try_swap_colors_outer_wrapper(train_inputs, train_outputs, test_input)
    if result is not None:
        return result

    # --- highlight_color as outer wrapper ---
    result = _try_highlight_color_outer_wrapper(train_inputs, train_outputs, test_input)
    if result is not None:
        return result

    return None


def _try_swap_colors_outer_wrapper(
    train_inputs: List[Grid],
    train_outputs: List[Grid],
    test_input: Grid,
) -> Optional[Grid]:
    """
    Detect swap_colors as outermost wrapper by comparing output color sets.

    If the outputs have exactly two colors swapped compared to what some
    direct transform would produce, we detect and handle it.

    Since swap_colors is self-inverse, stripping = re-applying with same params.
    We try all possible 2-color swaps from the output color palette.
    """
    if not train_inputs or not train_outputs:
        return None
    # Collect candidate color pairs from first output
    out_colors = colors_in(train_outputs[0]) - {0}
    if len(out_colors) < 2:
        return None

    color_list = sorted(out_colors)
    # Try all pairs
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            c1, c2 = color_list[i], color_list[j]
            params = {"color1": c1, "color2": c2}

            # Strip swap from all outputs
            stripped = [T.swap_colors(out, params) for out in train_outputs]
            if not all(is_valid(s) for s in stripped):
                continue

            # Try to solve the inner problem
            inner = try_direct_transforms(train_inputs, stripped, test_input)
            if inner is not None:
                # Re-apply swap_colors (self-inverse)
                final = T.swap_colors(inner, params)
                if is_valid(final):
                    return final

    return None


def _try_highlight_color_outer_wrapper(
    train_inputs: List[Grid],
    train_outputs: List[Grid],
    test_input: Grid,
) -> Optional[Grid]:
    """
    Detect highlight_color as outermost wrapper.
    """
    if not train_inputs or not train_outputs:
        return None
    # Detect candidate highlight color from outputs
    for out in train_outputs:
        out_colors = colors_in(out) - {0}
        non_gray = out_colors - {5}
        if len(non_gray) != 1:
            return None
        if 5 not in out_colors:
            # No gray means highlight wasn't applied (or no non-highlight colors)
            return None

    hl_color = (colors_in(train_outputs[0]) - {0, 5}).pop()
    hl_params = {"color": hl_color}

    # Try: highlight_color(direct_parameterless(input)) == output
    for name in T.PARAMETERLESS:
        func = T.ALL_TRANSFORMS[name]
        try:
            # Quick check on first example
            mid0 = func(train_inputs[0])
            cand0 = T.highlight_color(mid0, hl_params)
            if not grids_equal(cand0, train_outputs[0]):
                continue

            if all(
                grids_equal(T.highlight_color(func(inp), hl_params), out)
                for inp, out in zip(train_inputs, train_outputs)
            ):
                result = T.highlight_color(func(test_input), hl_params)
                if is_valid(result):
                    return result
        except Exception:
            continue

    # Try: highlight_color(swap_colors(input, swap_params)) == output
    # Detect swap_colors params from first example by comparing input colors
    # to pre-highlight output colors. This is too speculative without more
    # info, so skip for now.

    return None


def _try_parameterless_then_parameterized(inputs, outputs, test_input):
    """Try parameterless_transform → parameterized_transform chains.

    For each parameterless P and parameterized Q:
      check if Q(P(input)) == output for all examples
    """
    for pname in T.PARAMETERLESS:
        pfunc = T.ALL_TRANSFORMS[pname]
        try:
            mid_inputs = [pfunc(inp) for inp in inputs]
            if not all(is_valid(m) for m in mid_inputs):
                continue
        except Exception:
            continue

        # Try swap_colors after parameterless
        result = _try_swap_colors(mid_inputs, outputs, pfunc(test_input))
        if result is not None:
            return result

        # Try remove_color after parameterless
        result = _try_remove_color(mid_inputs, outputs, pfunc(test_input))
        if result is not None:
            return result

        # Try highlight_color after parameterless
        result = _try_highlight_color(mid_inputs, outputs, pfunc(test_input))
        if result is not None:
            return result

        # Try shift after parameterless
        result = _try_shift(mid_inputs, outputs, pfunc(test_input))
        if result is not None:
            return result

    return None


def _try_strip_zoom_2x(outputs: List[Grid]) -> Optional[List[Grid]]:
    """Try to strip zoom_2x from all outputs. Returns stripped or None."""
    stripped = []
    for out in outputs:
        h, w = dims(out)
        if h % 2 != 0 or w % 2 != 0 or h < 4 or w < 4:
            return None
        half_h, half_w = h // 2, w // 2
        result = [[0] * half_w for _ in range(half_h)]
        for r in range(half_h):
            for c in range(half_w):
                vals = {
                    out[r * 2][c * 2], out[r * 2][c * 2 + 1],
                    out[r * 2 + 1][c * 2], out[r * 2 + 1][c * 2 + 1],
                }
                if len(vals) > 1:
                    return None
                result[r][c] = out[r * 2][c * 2]
        stripped.append(result)
    return stripped


def _try_strip_zoom_3x(outputs: List[Grid]) -> Optional[List[Grid]]:
    """Try to strip zoom_3x from all outputs. Returns stripped or None."""
    stripped = []
    for out in outputs:
        h, w = dims(out)
        if h % 3 != 0 or w % 3 != 0 or h < 6 or w < 6:
            return None
        third_h, third_w = h // 3, w // 3
        result = [[0] * third_w for _ in range(third_h)]
        for r in range(third_h):
            for c in range(third_w):
                vals = set()
                for dr in range(3):
                    for dc in range(3):
                        vals.add(out[r * 3 + dr][c * 3 + dc])
                if len(vals) > 1:
                    return None
                result[r][c] = out[r * 3][c * 3]
        stripped.append(result)
    return stripped


# ============= SINGLE TRANSFORM DETECTION =============

def _try_single_transforms(inputs, outputs, test_input):
    """Try every single transform."""
    # Parameterless transforms
    for name in T.PARAMETERLESS:
        func = T.ALL_TRANSFORMS[name]
        try:
            if all(grids_equal(func(inp), out) for inp, out in zip(inputs, outputs)):
                result = func(test_input)
                if is_valid(result):
                    return result
        except Exception:
            continue

    # Parameterized: swap_colors
    result = _try_swap_colors(inputs, outputs, test_input)
    if result is not None:
        return result

    # Parameterized: remove_color
    result = _try_remove_color(inputs, outputs, test_input)
    if result is not None:
        return result

    # Parameterized: highlight_color
    result = _try_highlight_color(inputs, outputs, test_input)
    if result is not None:
        return result

    # Parameterized: shift
    result = _try_shift(inputs, outputs, test_input)
    if result is not None:
        return result

    return None


def _get_dims_after(h, w, name):
    """Get output dimensions for a transform without applying it."""
    if name in ("rotate_90", "rotate_270", "transpose",
                "flip_diagonal", "flip_antidiagonal"):
        return (w, h)
    if name == "zoom_2x":
        return (h * 2, w * 2)
    if name == "zoom_3x":
        return (h * 3, w * 3)
    if name == "downsample_2x":
        return (h // 2, w // 2)
    return (h, w)



# Only transforms that actually appear in Hone chains (TRANSFORMATIONS registry).
# Excludes: rotate_90, flip_horizontal, flip_vertical (not in registry, helper-only)
# Excludes: flip_diagonal (identical to transpose — flip_diagonal() just calls transpose())
_CHAIN_TRANSFORMS = sorted([
    "rotate_180", "rotate_270", "transpose",
    "flip_antidiagonal", "recenter",
    "zoom_2x", "zoom_3x", "downsample_2x",
    "gravity_down", "gravity_up", "gravity_left", "gravity_right",
])


def _try_chain_n(inputs, outputs, test_input, max_depth=5):
    """Try chains of N parameterless transforms with dimension filtering.

    Uses dimension pre-filtering to eliminate >95% of combinations instantly.
    Only computes actual transforms for dimension-compatible chains.
    Accumulates transform results through DFS to avoid re-applying full chain at leaves.
    """
    if not inputs or not outputs:
        return None

    inp0, out0 = inputs[0], outputs[0]
    ih, iw = dims(inp0)
    oh, ow = dims(out0)

    names = _CHAIN_TRANSFORMS
    funcs = {n: T.ALL_TRANSFORMS[n] for n in names}

    # Build dimension-filtered chains using DFS with accumulation
    def search(depth, target_depth, cur_h, cur_w, chain, accumulated):
        if depth == target_depth:
            if (cur_h, cur_w) != (oh, ow):
                return None
            # accumulated already has chain applied to inp0
            if not grids_equal(accumulated, out0):
                return None
            # Verify on ALL other examples
            for inp, out in zip(inputs[1:], outputs[1:]):
                r = inp
                for n in chain:
                    r = funcs[n](r)
                if not grids_equal(r, out):
                    return None
            # Apply to test
            r = test_input
            for n in chain:
                r = funcs[n](r)
            if is_valid(r):
                return r
            return None

        for n in names:
            nh, nw = _get_dims_after(cur_h, cur_w, n)
            # Prune: remaining steps can at most change dims by known factors
            # Skip if dimensions are already too large
            if nh > oh * 3 or nw > ow * 3:
                continue
            if nh < 1 or nw < 1:
                continue
            chain.append(n)
            new_acc = funcs[n](accumulated)
            result = search(depth + 1, target_depth, nh, nw, chain, new_acc)
            if result is not None:
                return result
            chain.pop()
        return None

    # Try depth 3, then 4, then 5
    for depth in range(3, max_depth + 1):
        result = search(0, depth, ih, iw, [], inp0)
        if result is not None:
            return result

    return None


def _try_double_transforms(inputs, outputs, test_input):
    """Try pairs of parameterless transforms with dimension filtering."""
    inp0, out0 = inputs[0], outputs[0]
    ih, iw = dims(inp0)
    oh, ow = dims(out0)

    parameterless_list = sorted(T.PARAMETERLESS)

    for n1 in parameterless_list:
        f1 = T.ALL_TRANSFORMS[n1]
        try:
            mid1 = f1(inp0)
        except Exception:
            continue

        for n2 in parameterless_list:
            f2 = T.ALL_TRANSFORMS[n2]
            try:
                mid2 = f2(mid1)
            except Exception:
                continue

            # Quick dimension check
            if dims(mid2) != (oh, ow):
                continue

            # Quick content check on first example
            if not grids_equal(mid2, out0):
                continue

            # Verify on ALL examples
            try:
                if all(grids_equal(f2(f1(inp)), out) for inp, out in zip(inputs, outputs)):
                    result = f2(f1(test_input))
                    if is_valid(result):
                        return result
            except Exception:
                continue

    return None


# ============= PARAMETERIZED TRANSFORM DETECTION =============

def _try_swap_colors(inputs, outputs, test_input):
    """Try to find a consistent color swap across all examples."""
    if not inputs or not outputs:
        return None
    for inp, out in zip(inputs, outputs):
        if dims(inp) != dims(out):
            return None

    # Find swap mapping from first example
    inp0, out0 = inputs[0], outputs[0]
    mapping = {}
    for r in range(len(inp0)):
        for c in range(len(inp0[0])):
            iv, ov = inp0[r][c], out0[r][c]
            if iv != ov:
                if iv in mapping:
                    if mapping[iv] != ov:
                        return None
                mapping[iv] = ov

    if len(mapping) != 2:
        return None

    # Check it's a swap (a→b AND b→a)
    items = list(mapping.items())
    (a, b), (c, d) = items
    if not (a == d and b == c):
        return None

    params = {"color1": a, "color2": b}

    # Verify on ALL examples
    for inp, out in zip(inputs, outputs):
        if not grids_equal(T.swap_colors(inp, params), out):
            return None

    result = T.swap_colors(test_input, params)
    return result if is_valid(result) else None


def _try_remove_color(inputs, outputs, test_input):
    if not inputs or not outputs:
        return None
    for inp, out in zip(inputs, outputs):
        if dims(inp) != dims(out):
            return None

    in_colors = colors_in(inputs[0])
    out_colors = colors_in(outputs[0])
    removed = in_colors - out_colors - {0}

    if len(removed) != 1:
        return None

    color = removed.pop()
    params = {"color": color}

    for inp, out in zip(inputs, outputs):
        if not grids_equal(T.remove_color(inp, params), out):
            return None

    result = T.remove_color(test_input, params)
    return result if is_valid(result) else None


def _try_highlight_color(inputs, outputs, test_input):
    if not inputs or not outputs:
        return None
    for inp, out in zip(inputs, outputs):
        if dims(inp) != dims(out):
            return None

    out_colors = colors_in(outputs[0])
    non_zero_non_gray = out_colors - {0, 5}
    if len(non_zero_non_gray) != 1:
        return None

    hl_color = non_zero_non_gray.pop()
    params = {"color": hl_color}

    for inp, out in zip(inputs, outputs):
        if not grids_equal(T.highlight_color(inp, params), out):
            return None

    result = T.highlight_color(test_input, params)
    return result if is_valid(result) else None


def _try_shift(inputs, outputs, test_input):
    if not inputs or not outputs:
        return None
    for inp, out in zip(inputs, outputs):
        if dims(inp) != dims(out):
            return None

    for direction in ["up", "down", "left", "right"]:
        for amount in range(1, 4):
            params = {"direction": direction, "amount": amount}
            if all(grids_equal(T.shift(inp, params), out) for inp, out in zip(inputs, outputs)):
                result = T.shift(test_input, params)
                return result if is_valid(result) else None
    return None


# ============= OUTPUT CHAIN DETECTION (for stripping) =============

def _detect_output_transform_validated(
    inputs: List[Grid],
    outputs: List[Grid],
) -> Optional[Tuple[str, Optional[Dict], List[Grid]]]:
    """Detect the last transform on outputs with input-based cross-validation.

    Key insight: if zoom was the LAST transform in the chain, then stripping it
    should bring output dimensions CLOSER to input dimensions (or to some
    consistent ratio with inputs).
    """
    # Try zoom_3x first (more specific)
    r = _detect_output_zoom_3x(outputs)
    if r and _zoom_makes_sense(inputs, outputs, r[2], 3):
        return r

    r = _detect_output_zoom_2x(outputs)
    if r and _zoom_makes_sense(inputs, outputs, r[2], 2):
        return r

    return None


def _zoom_makes_sense(
    inputs: List[Grid],
    outputs: List[Grid],
    stripped: List[Grid],
    zoom_factor: int,
) -> bool:
    """Validate zoom: stripped output dims must exactly match input dims.

    This is the strongest possible validation. If zoom_Nx was the last
    transform, then stripping it gives us base_task output (or an
    intermediate). For many base tasks, output dims == input dims,
    so stripped dims should match input dims.
    """
    # All stripped outputs must match their corresponding input dims
    for inp, s in zip(inputs, stripped):
        ih, iw = dims(inp)
        sh, sw = dims(s)
        if (sh, sw) != (ih, iw):
            return False
    return True


def _is_valid_zoom_source(grid: Grid) -> bool:
    """Strict check: is this grid a plausible pre-zoom source?
    Rejects grids where zoom detection is likely coincidental.
    """
    h, w = dims(grid)
    total = h * w
    if total < 9:  # At least 3x3
        return False

    # Count color frequencies
    freq = {}
    for row in grid:
        for v in row:
            freq[v] = freq.get(v, 0) + 1

    # Require at least 3 distinct colors (including black)
    if len(freq) < 3:
        return False

    # Non-black cells must be at least 25% of total
    non_black = total - freq.get(0, 0)
    if non_black < total * 0.25:
        return False

    # No single color can dominate more than 60%
    max_freq = max(freq.values())
    if max_freq > total * 0.60:
        return False

    # Count spatial transitions (edges between different-colored cells)
    edges = 0
    for r in range(h):
        for c in range(w):
            if c + 1 < w and grid[r][c] != grid[r][c + 1]:
                edges += 1
            if r + 1 < h and grid[r][c] != grid[r + 1][c]:
                edges += 1

    max_edges = (h - 1) * w + h * (w - 1)
    if max_edges > 0 and edges / max_edges < 0.15:
        return False

    return True


def _detect_output_zoom_2x(outputs):
    """Detect if ALL outputs are zoom_2x of some intermediate grid."""
    stripped = []
    for out in outputs:
        h, w = dims(out)
        if h % 2 != 0 or w % 2 != 0 or h < 6 or w < 6:
            return None
        half_h, half_w = h // 2, w // 2
        if half_h < 3 or half_w < 3:
            return None
        result = [[0] * half_w for _ in range(half_h)]
        for r in range(half_h):
            for c in range(half_w):
                vals = {
                    out[r * 2][c * 2], out[r * 2][c * 2 + 1],
                    out[r * 2 + 1][c * 2], out[r * 2 + 1][c * 2 + 1],
                }
                if len(vals) > 1:
                    return None
                result[r][c] = out[r * 2][c * 2]
        stripped.append(result)

    # All stripped results must have spatial variety (reject trivial uniformity)
    if not all(_is_valid_zoom_source(s) for s in stripped):
        return None

    return ("zoom_2x", None, stripped)


def _detect_output_zoom_3x(outputs):
    """Detect if ALL outputs are zoom_3x of some intermediate grid."""
    stripped = []
    for out in outputs:
        h, w = dims(out)
        if h % 3 != 0 or w % 3 != 0 or h < 9 or w < 9:
            return None
        third_h, third_w = h // 3, w // 3
        if third_h < 3 or third_w < 3:
            return None
        result = [[0] * third_w for _ in range(third_h)]
        for r in range(third_h):
            for c in range(third_w):
                vals = set()
                for dr in range(3):
                    for dc in range(3):
                        vals.add(out[r * 3 + dr][c * 3 + dc])
                if len(vals) > 1:
                    return None
                result[r][c] = out[r * 3][c * 3]
        stripped.append(result)

    if not all(_is_valid_zoom_source(s) for s in stripped):
        return None

    return ("zoom_3x", None, stripped)
