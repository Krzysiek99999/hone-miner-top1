"""LLM-based ARC solver using vLLM backend.

Features:
- Multi-representation prompting (JSON + visual)
- Structural analysis hints (dimensions, colors, symmetry)
- Chain-of-thought reasoning
- Best-of-N voting
- Program synthesis (LLM generates Python code) — multi-attempt
- Robust JSON parsing from LLM output
"""

import json
import os
import re
import time
from collections import Counter
from typing import List, Dict, Optional, Any

from solver.grid_utils import Grid, dims, is_valid, colors_in, density


# Color emoji map for visual grid representation
COLOR_EMOJI = {
    0: ".", 1: "B", 2: "R", 3: "G", 4: "Y",
    5: "X", 6: "P", 7: "O", 8: "A", 9: "M",
}

COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "azure", 9: "maroon",
}


class LLMEngine:
    def __init__(self):
        self.client = None
        self.model_name = None
        self.available = False
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            from config import VLLM_API_BASE
            api_base = VLLM_API_BASE
            self.client = OpenAI(base_url=f"{api_base}/v1", api_key="dummy")
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            if model_ids:
                self.model_name = model_ids[0]
                self.available = True
                print(f"[LLM] Connected to vLLM: {self.model_name}")
            else:
                print("[LLM] No models available on vLLM server")
        except Exception as e:
            print(f"[LLM] vLLM not available: {e}")

    def solve(
        self,
        train_examples: List[Dict],
        test_input: Grid,
        time_budget: float = 30.0,
        chain_hints: Optional[List[str]] = None,
    ) -> Optional[Grid]:
        """Solve using LLM with voting + program synthesis."""
        if not self.available:
            return None

        start = time.time()

        # Allocate budget: 55% voting, 45% program synthesis
        vote_budget = time_budget * 0.55
        prog_budget = time_budget * 0.45

        n_attempts = max(1, min(5, int(vote_budget / 7)))

        # Strategy 1: Direct solving with voting
        result = self._solve_with_voting(
            train_examples, test_input, n_attempts, vote_budget,
            chain_hints=chain_hints,
        )
        if result is not None:
            return result

        remaining = time_budget - (time.time() - start)

        # Strategy 2: Program synthesis (multiple attempts with different prompts)
        if remaining > 10:
            result = self._solve_with_program(train_examples, test_input, remaining)
            if result is not None:
                return result

        return None

    def _solve_with_voting(
        self,
        train_examples: List[Dict],
        test_input: Grid,
        n_attempts: int,
        time_budget: float,
        chain_hints: Optional[List[str]] = None,
    ) -> Optional[Grid]:
        """Generate N solutions, pick most common (dimension-filtered)."""
        from solver.validator import infer_output_dims
        start = time.time()
        candidates = []

        analysis = _analyze_problem(train_examples, test_input)
        expected_dims = infer_output_dims(train_examples, test_input)

        for attempt in range(n_attempts):
            if time.time() - start > time_budget * 0.7:
                break

            temp = 0.3 + (attempt * 0.15)
            prompt = _build_prompt(train_examples, test_input, analysis, chain_hints)

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                    max_tokens=6000,
                )
                content = response.choices[0].message.content
                grid = _parse_grid_response(content)
                if grid is not None and is_valid(grid):
                    candidates.append(grid)
            except Exception as e:
                print(f"    [LLM] Attempt {attempt + 1} failed: {e}")

        if not candidates:
            return None

        # Filter by expected dimensions (if we know them)
        if expected_dims is not None:
            dim_filtered = [g for g in candidates if dims(g) == expected_dims]
            if dim_filtered:
                candidates = dim_filtered

        # Majority voting — pick the most common answer
        grid_strs = [json.dumps(g) for g in candidates]
        counter = Counter(grid_strs)
        most_common_str, most_common_count = counter.most_common(1)[0]

        # If there's a clear winner (>1 vote), use it
        if most_common_count > 1:
            return json.loads(most_common_str)

        # No consensus — return first candidate (lowest temperature = most confident)
        return candidates[0]

    def _solve_with_program(
        self,
        train_examples: List[Dict],
        test_input: Grid,
        time_budget: float,
    ) -> Optional[Grid]:
        """Ask LLM to write a Python function, with multiple attempts."""
        start = time.time()
        n_attempts = max(1, min(3, int(time_budget / 12)))

        for attempt in range(n_attempts):
            if time.time() - start > time_budget * 0.85:
                break

            temp = 0.1 + (attempt * 0.2)
            # Alternate between standard and detailed prompting
            if attempt % 2 == 0:
                prompt = _build_program_synthesis_prompt(train_examples, test_input)
            else:
                prompt = _build_program_synthesis_prompt_detailed(train_examples, test_input)

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": PROGRAM_SYNTHESIS_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                    max_tokens=6000,
                )
                content = response.choices[0].message.content
                code = _extract_code(content)
                if not code:
                    continue

                result = _execute_solver_code(code, train_examples, test_input)
                if result is not None and is_valid(result):
                    return result
            except Exception as e:
                print(f"    [LLM] Program synthesis attempt {attempt+1} failed: {e}")

        return None

    def solve_simplified(
        self,
        train_examples: List[Dict],
        test_input: Grid,
        chain_description: str,
        time_budget: float = 25.0,
    ) -> Optional[Grid]:
        """Solve a simplified problem (after chain stripping)."""
        if not self.available:
            return None

        prompt = _build_simplified_prompt(train_examples, test_input, chain_description)
        n_attempts = max(1, min(3, int(time_budget / 8)))
        candidates = []
        start = time.time()

        for attempt in range(n_attempts):
            if time.time() - start > time_budget * 0.7:
                break
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3 + attempt * 0.15,
                    max_tokens=4000,
                )
                grid = _parse_grid_response(response.choices[0].message.content)
                if grid is not None and is_valid(grid):
                    candidates.append(grid)
            except Exception:
                pass

        if not candidates:
            return None

        grid_strs = [json.dumps(g) for g in candidates]
        return json.loads(Counter(grid_strs).most_common(1)[0][0])


# ============= PROMPTS =============

SYSTEM_PROMPT = """You are a world-class ARC-AGI-2 puzzle solver. You find EXACT transformation rules from examples.

ARC-AGI-2 puzzles have colored grids (0-9). Training examples show input→output pairs that share ONE consistent rule. Apply that rule to the test input.

REASONING STRATEGY:
1. Compare dimensions: Does the output size relate to input? (same, doubled, halved, transposed?)
2. Compare colors: Which colors appear/disappear? Any color mappings?
3. Check geometry: Is the output a rotation, flip, or zoom of the input?
4. Look for objects: Are shapes moved, copied, filtered, or combined?
5. Check for composites: Could the rule be TWO simpler operations chained?

KNOWN TRANSFORM TYPES (the puzzle may combine several):
- Geometric: rotate 90°/180°/270°, flip horizontal/vertical, transpose
- Scale: zoom 2x/3x (each pixel becomes 2x2/3x3 block), downsample
- Color: swap two colors, remove a color (→black), highlight one color (others→gray)
- Gravity: push non-black cells down/up/left/right
- Spatial: shift grid, recenter content

CRITICAL RULES:
1. Output must be PIXEL-PERFECT — every cell matters.
2. The SAME rule applies to ALL training examples.
3. Think step-by-step, then output the grid.
4. Return the output as a JSON array of arrays at the end.

Colors: 0=black 1=blue 2=red 3=green 4=yellow 5=gray 6=magenta 7=orange 8=azure 9=maroon"""


PROGRAM_SYNTHESIS_SYSTEM = """You are an expert programmer solving ARC-AGI-2 puzzles by writing Python code.

Write a function `solve(input_grid)` that takes a 2D list of ints (0-9) and returns the output 2D list.

APPROACH:
1. Study ALL examples carefully. The same rule applies to each.
2. Describe the pattern in a comment before coding.
3. Consider: size changes, color mappings, geometric transforms, object manipulation.
4. Write clean, generalizable code — no hardcoded values from specific examples.

RULES:
- Only standard Python (no imports). You may use: len, range, list, set, dict, min, max, sum, sorted, enumerate, zip, any, all, abs, reversed.
- Return a valid 2D list of ints 0-9.
- Wrap code in ```python ... ```."""


def _analyze_problem(train_examples: List[Dict], test_input: Grid) -> str:
    """Generate structural analysis hints for the LLM."""
    hints = []

    # Dimension analysis
    dim_pairs = []
    for ex in train_examples:
        ih, iw = dims(ex["input"])
        oh, ow = dims(ex["output"])
        dim_pairs.append((ih, iw, oh, ow))

    if dim_pairs:
        ih0, iw0, oh0, ow0 = dim_pairs[0]
        if all(oh == ih and ow == iw for ih, iw, oh, ow in dim_pairs):
            hints.append("Size: output is SAME size as input.")
        elif all(oh == ih * 2 and ow == iw * 2 for ih, iw, oh, ow in dim_pairs):
            hints.append("Size: output is 2x larger (zoom 2x?).")
        elif all(oh == ih * 3 and ow == iw * 3 for ih, iw, oh, ow in dim_pairs):
            hints.append("Size: output is 3x larger (zoom 3x?).")
        elif all(oh * 2 == ih and ow * 2 == iw for ih, iw, oh, ow in dim_pairs):
            hints.append("Size: output is half the input (downsample?).")
        elif all(oh == iw and ow == ih for ih, iw, oh, ow in dim_pairs):
            hints.append("Size: dimensions are swapped (rotate 90/270 or transpose?).")
        elif len(set((oh, ow) for _, _, oh, ow in dim_pairs)) == 1:
            hints.append(f"Size: all outputs are {oh0}x{ow0} (fixed output size).")
        else:
            hints.append(f"Size: output dimensions vary (first: {ih0}x{iw0} → {oh0}x{ow0}).")

    # Color analysis
    all_in_colors = set()
    all_out_colors = set()
    for ex in train_examples:
        all_in_colors |= colors_in(ex["input"])
        all_out_colors |= colors_in(ex["output"])

    new_colors = all_out_colors - all_in_colors
    lost_colors = all_in_colors - all_out_colors
    if new_colors:
        names = [COLOR_NAMES.get(c, str(c)) for c in sorted(new_colors)]
        hints.append(f"Colors: outputs introduce {', '.join(names)}.")
    if lost_colors:
        names = [COLOR_NAMES.get(c, str(c)) for c in sorted(lost_colors)]
        hints.append(f"Colors: {', '.join(names)} disappear in outputs.")

    # Check for highlight_color pattern (outputs have only 1 non-black non-gray color)
    for ex in train_examples:
        oc = colors_in(ex["output"]) - {0}
        non_gray = oc - {5}
        if 5 in oc and len(non_gray) == 1:
            hints.append(f"Pattern: outputs keep only {COLOR_NAMES.get(non_gray.pop(), '?')} + gray (highlight?).")
            break

    # Density analysis
    for ex in train_examples:
        id_ = density(ex["input"])
        od_ = density(ex["output"])
        if abs(id_ - od_) > 0.3:
            hints.append(f"Density changes significantly ({id_:.0%} → {od_:.0%}).")
            break

    return " ".join(hints) if hints else ""


def _grid_to_visual(grid: Grid) -> str:
    """Convert grid to compact visual representation."""
    lines = []
    for row in grid:
        lines.append(" ".join(COLOR_EMOJI.get(v, "?") for v in row))
    return "\n".join(lines)


def _build_prompt(
    train_examples: List[Dict],
    test_input: Grid,
    analysis: str = "",
    chain_hints: Optional[List[str]] = None,
) -> str:
    parts = []
    parts.append("# ARC-AGI-2 Puzzle\n")

    # Add structural analysis
    if analysis:
        parts.append(f"## Observations\n{analysis}\n")

    if chain_hints:
        parts.append(f"## Hints\nThese transforms may be involved: {', '.join(chain_hints)}\n")

    for i, ex in enumerate(train_examples, 1):
        inp = ex["input"]
        out = ex["output"]
        ih, iw = dims(inp)
        oh, ow = dims(out)
        parts.append(f"## Training Example {i}")
        parts.append(f"Input ({ih}x{iw}):")
        parts.append(f"```\n{_grid_to_visual(inp)}\n```")
        parts.append(f"JSON: {json.dumps(inp)}")
        parts.append(f"Output ({oh}x{ow}):")
        parts.append(f"```\n{_grid_to_visual(out)}\n```")
        parts.append(f"JSON: {json.dumps(out)}\n")

    th, tw = dims(test_input)
    parts.append(f"## Test Input ({th}x{tw})")
    parts.append(f"```\n{_grid_to_visual(test_input)}\n```")
    parts.append(f"JSON: {json.dumps(test_input)}\n")
    parts.append("## Your Answer")
    parts.append("1. Describe the transformation rule in one sentence.")
    parts.append("2. Apply it to the test input.")
    parts.append("3. Return the output grid as a JSON array of arrays.")

    return "\n".join(parts)


def _build_simplified_prompt(
    train_examples: List[Dict],
    test_input: Grid,
    chain_description: str,
) -> str:
    """Build prompt for a simplified problem (chain already stripped)."""
    parts = [_build_prompt(train_examples, test_input)]
    parts.append(f"\nNote: The outputs have been simplified. {chain_description}")
    return "\n".join(parts)


def _build_program_synthesis_prompt(
    train_examples: List[Dict],
    test_input: Grid,
) -> str:
    parts = []
    parts.append("Write a Python function `solve(input_grid)` that transforms the input grid to produce the output grid.\n")
    parts.append("Here are the training examples:\n")

    for i, ex in enumerate(train_examples, 1):
        ih, iw = dims(ex["input"])
        oh, ow = dims(ex["output"])
        parts.append(f"Example {i} ({ih}x{iw} → {oh}x{ow}):")
        parts.append(f"  Input:  {json.dumps(ex['input'])}")
        parts.append(f"  Output: {json.dumps(ex['output'])}\n")

    th, tw = dims(test_input)
    parts.append(f"Test input ({th}x{tw}): {json.dumps(test_input)}\n")
    parts.append("Write the `solve(input_grid)` function that works for ALL examples above.")
    parts.append("The function should take a list of lists of ints and return a list of lists of ints.")

    return "\n".join(parts)


def _build_program_synthesis_prompt_detailed(
    train_examples: List[Dict],
    test_input: Grid,
) -> str:
    """Detailed prompt with visual grids and analysis hints."""
    parts = []
    analysis = _analyze_problem(train_examples, test_input)

    parts.append("# Task: Write `solve(input_grid)` for this ARC puzzle\n")
    if analysis:
        parts.append(f"Observations: {analysis}\n")

    for i, ex in enumerate(train_examples, 1):
        inp = ex["input"]
        out = ex["output"]
        ih, iw = dims(inp)
        oh, ow = dims(out)
        parts.append(f"## Example {i} ({ih}x{iw} → {oh}x{ow})")
        parts.append(f"Input:\n```\n{_grid_to_visual(inp)}\n```")
        parts.append(f"Output:\n```\n{_grid_to_visual(out)}\n```")
        parts.append(f"Input JSON: {json.dumps(inp)}")
        parts.append(f"Output JSON: {json.dumps(out)}\n")

    th, tw = dims(test_input)
    parts.append(f"## Test Input ({th}x{tw})")
    parts.append(f"```\n{_grid_to_visual(test_input)}\n```")
    parts.append(f"JSON: {json.dumps(test_input)}\n")

    parts.append("First describe the transformation in a comment, then write `solve(input_grid)` that works for ALL examples.")
    parts.append("Common patterns: rotation, flipping, zooming, color swaps, gravity, object extraction.")

    return "\n".join(parts)


# ============= PARSING =============

def _parse_grid_response(content: str) -> Optional[Grid]:
    """Parse LLM response to extract a grid. Handles various formats.

    Handles thinking tokens (<think>...</think>), markdown code blocks,
    and raw JSON arrays anywhere in the text.
    """
    if content is None:
        return None

    content = content.strip()

    # Strip thinking tokens (Qwen3, DeepSeek-R1, etc.)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL).strip()

    # Try to find JSON array in code blocks
    json_str = None

    # Pattern 1: ```json ... ```
    match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if match:
        json_str = match.group(1)

    # Pattern 2: ``` ... ```  (generic code block)
    if json_str is None:
        match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            if candidate.startswith('['):
                json_str = candidate

    # Pattern 3: Find the last JSON array in the text
    if json_str is None:
        # Find all potential JSON arrays
        bracket_positions = []
        depth = 0
        start = -1
        for i, ch in enumerate(content):
            if ch == '[':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0 and start >= 0:
                    bracket_positions.append((start, i + 1))
                    start = -1

        # Try the last (usually the answer), then others
        for s, e in reversed(bracket_positions):
            candidate = content[s:e]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
                    return _validate_parsed_grid(parsed)
            except json.JSONDecodeError:
                continue

    if json_str is not None:
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
                return _validate_parsed_grid(parsed)
        except json.JSONDecodeError:
            pass

    return None


def _validate_parsed_grid(parsed: Any) -> Optional[Grid]:
    """Validate and clean a parsed grid."""
    if not isinstance(parsed, list) or not parsed:
        return None

    grid = []
    width = None
    for row in parsed:
        if not isinstance(row, list):
            return None
        int_row = []
        for v in row:
            try:
                iv = int(v)
                if iv < 0 or iv > 9:
                    return None
                int_row.append(iv)
            except (TypeError, ValueError):
                return None
        if width is None:
            width = len(int_row)
        elif len(int_row) != width:
            return None
        grid.append(int_row)

    if not grid or not grid[0]:
        return None
    if len(grid) > 30 or len(grid[0]) > 30:
        return None

    return grid


def _extract_code(content: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    # Strip thinking tokens
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL).strip()

    # Find ```python ... ``` block
    match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
    if match:
        return match.group(1)

    # Find ``` ... ``` block that looks like Python
    match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
    if match:
        code = match.group(1)
        if 'def solve' in code:
            return code

    # Find inline def solve
    match = re.search(r'(def solve.*?)(?=\n\S|\Z)', content, re.DOTALL)
    if match:
        return match.group(1)

    return None


def _execute_solver_code(
    code: str,
    train_examples: List[Dict],
    test_input: Grid,
    timeout_sec: float = 5.0,
) -> Optional[Grid]:
    """Safely execute LLM-generated solver code with timeout."""
    import signal

    # Basic safety checks
    dangerous = ['import os', 'import sys', 'subprocess', 'eval(', 'exec(',
                 '__import__', 'open(', 'file(', 'input(', 'globals(']
    for d in dangerous:
        if d in code:
            return None

    # Allow safe imports by rewriting them
    safe_imports = [
        'from copy import deepcopy',
        'import copy',
        'from collections import defaultdict',
        'from collections import Counter',
        'import collections',
        'from itertools import product',
        'import itertools',
        'from typing import List, Tuple, Dict, Set, Optional, Any',
        'from typing import *',
    ]
    clean_code = code
    for imp_str in safe_imports:
        clean_code = clean_code.replace(imp_str, '')

    namespace = {}
    builtins_dict = _safe_builtins()

    def _timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    try:
        exec(clean_code, {"__builtins__": builtins_dict}, namespace)
    except Exception:
        return None

    solve_fn = namespace.get('solve')
    if solve_fn is None:
        return None

    # Set timeout for execution
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        signal.alarm(int(timeout_sec) + 1)

        # Verify on training examples first
        for ex in train_examples:
            try:
                result = solve_fn(ex["input"])
                if result != ex["output"]:
                    return None
            except TimeoutError:
                return None
            except Exception:
                return None

        # Apply to test input
        try:
            result = solve_fn(test_input)
            if isinstance(result, list) and result and isinstance(result[0], list):
                return result
        except Exception:
            pass

        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _safe_builtins():
    """Restricted builtins for code execution."""
    import builtins
    import copy
    from collections import defaultdict, Counter
    from itertools import product

    safe = {}
    allowed = [
        'abs', 'all', 'any', 'bool', 'chr', 'dict', 'enumerate', 'filter',
        'float', 'frozenset', 'hasattr', 'hash', 'int', 'isinstance',
        'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min', 'next',
        'ord', 'pow', 'print', 'range', 'repr', 'reversed', 'round', 'set',
        'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip',
        'True', 'False', 'None',
    ]
    for name in allowed:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)

    # Common modules that LLMs tend to import
    safe['copy'] = copy
    safe['deepcopy'] = copy.deepcopy
    safe['defaultdict'] = defaultdict
    safe['Counter'] = Counter
    safe['product'] = product
    return safe
