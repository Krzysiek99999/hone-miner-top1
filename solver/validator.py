"""Self-check validation for predicted outputs.

Validates that predicted outputs are structurally consistent
with patterns observed in training examples.
"""

from typing import List, Dict, Optional, Set, Tuple
from solver.grid_utils import Grid, dims, is_valid, colors_in, density, non_black_count


def validate_prediction(
    train_examples: List[Dict],
    test_input: Grid,
    predicted: Grid,
) -> bool:
    """Multi-level validation of a predicted output."""
    if predicted is None:
        return False
    if not is_valid(predicted):
        return False
    if not _check_dimensions(train_examples, test_input, predicted):
        return False
    if not _check_colors(train_examples, test_input, predicted):
        return False
    return True


def _check_dimensions(
    train_examples: List[Dict],
    test_input: Grid,
    predicted: Grid,
) -> bool:
    """Check if predicted dimensions match the pattern from train examples."""
    ph, pw = dims(predicted)

    # Gather output dimensions from training
    out_dims = set()
    for ex in train_examples:
        oh, ow = dims(ex["output"])
        out_dims.add((oh, ow))

    # If all training outputs have same dimensions, predicted should too
    if len(out_dims) == 1:
        expected = out_dims.pop()
        if (ph, pw) != expected:
            return False
        return True

    # Check dimension ratios (input → output)
    ratios = set()
    for ex in train_examples:
        ih, iw = dims(ex["input"])
        oh, ow = dims(ex["output"])
        if ih > 0 and iw > 0:
            ratios.add((round(oh / ih, 3), round(ow / iw, 3)))

    if len(ratios) == 1:
        rh, rw = ratios.pop()
        th, tw = dims(test_input)
        exp_h = round(th * rh)
        exp_w = round(tw * rw)
        if ph != exp_h or pw != exp_w:
            return False

    return True


def _check_colors(
    train_examples: List[Dict],
    test_input: Grid,
    predicted: Grid,
) -> bool:
    """Check if predicted colors are consistent with training patterns."""
    pred_colors = colors_in(predicted)

    # Collect all colors that appear in train outputs and test input
    allowed = set()
    for ex in train_examples:
        allowed |= colors_in(ex["output"])
        allowed |= colors_in(ex["input"])
    allowed |= colors_in(test_input)

    # Predicted shouldn't introduce completely new colors
    novel = pred_colors - allowed
    if novel:
        return False

    return True


def infer_output_dims(
    train_examples: List[Dict],
    test_input: Grid,
) -> Optional[Tuple[int, int]]:
    """Infer expected output dimensions from train examples."""
    out_dims = [(dims(ex["output"])) for ex in train_examples]

    # All same size
    if len(set(out_dims)) == 1:
        return out_dims[0]

    # Consistent ratio
    ratios = []
    for ex in train_examples:
        ih, iw = dims(ex["input"])
        oh, ow = dims(ex["output"])
        if ih > 0 and iw > 0:
            ratios.append((oh / ih, ow / iw))

    if ratios and len(set((round(r, 3), round(c, 3)) for r, c in ratios)) == 1:
        rh, rw = ratios[0]
        th, tw = dims(test_input)
        return (round(th * rh), round(tw * rw))

    return None
