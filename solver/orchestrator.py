"""Main orchestrator — three-layer ARC-AGI-2 solving pipeline.

Layer 1: Direct transform detection — check if input→output is a known transform chain
Layer 2: Output chain stripping — detect transforms on outputs, simplify for LLM
Layer 3: Full LLM solving — multi-representation prompting with voting

Each layer validates its answer before returning.
"""

import time
from typing import List, Dict, Optional, Any

from solver.grid_utils import Grid, dims, is_valid, deep_copy, grids_equal, colors_in
from solver.chain_detector import (
    try_direct_transforms,
    try_zoom_wrapped_transforms,
    detect_output_chain,
    apply_chain,
)
from solver.llm_engine import LLMEngine
from solver.validator import validate_prediction, infer_output_dims
from solver.time_budget import TimeBudget


class Orchestrator:
    def __init__(self):
        from config import TOTAL_TIME_BUDGET
        print("[Orchestrator] Initializing...")
        self.llm = LLMEngine()
        self.budget = TimeBudget(total_seconds=TOTAL_TIME_BUDGET)
        print(f"[Orchestrator] Ready. LLM available: {self.llm.available}")

    def solve_all(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Solve all tasks and return predictions."""
        total = len(tasks)
        results = {}

        print(f"\n[Orchestrator] Solving {total} tasks...")
        print(f"[Orchestrator] LLM: {self.llm.available}")

        # PASS 1: Fast path (Layer 1 — direct transforms) for ALL tasks
        for i, task in enumerate(tasks):
            if "train_examples" not in task:
                continue
            t0 = time.time()
            result = self._try_fast_path(task)
            dt = time.time() - t0
            if result is not None:
                results[i] = result
                self.budget.record_fast()
                print(f"  [{i+1}/{total}] FAST solved in {dt:.2f}s")

        # Sort remaining tasks by estimated difficulty (easier first)
        unsolved = [i for i in range(total) if i not in results and "train_examples" in tasks[i]]
        unsolved.sort(key=lambda i: self._difficulty_score(tasks[i]))

        # PASS 2: LLM path (Layer 2 + 3) — sorted by difficulty
        for idx, i in enumerate(unsolved):
            task = tasks[i]
            tasks_remaining = len(unsolved) - idx
            task_budget = self.budget.budget_for_task(tasks_remaining)

            if task_budget < 3:
                # Not enough time — use fallback for remaining
                for j in unsolved[idx:]:
                    if j not in results:
                        results[j] = self._fallback(tasks[j])
                        self.budget.record_skip()
                break

            t0 = time.time()
            result = self._try_llm_path(task, task_budget)
            dt = time.time() - t0

            if result is not None:
                results[i] = result
                self.budget.record_llm()
                print(f"  [{i+1}/{total}] LLM solved in {dt:.2f}s")
            else:
                result = self._fallback(task)
                results[i] = result
                self.budget.record_skip()
                print(f"  [{i+1}/{total}] FALLBACK in {dt:.2f}s")

        # Compile predictions in original order
        predictions = []
        for i, task in enumerate(tasks):
            pred = results.get(i)
            if pred is None and "train_examples" in task:
                pred = self._fallback(task)
            predictions.append({
                "problem_index": i,
                "task_hash": task.get("task_hash"),
                "predicted_output": pred,
                "metadata": task.get("metadata", {}),
            })

        print(f"\n[Orchestrator] {self.budget.summary()}")
        return predictions

    def _difficulty_score(self, task: Dict) -> float:
        """Estimate task difficulty for prioritization (lower = easier)."""
        train = task.get("train_examples", [])
        if not train:
            return 100.0

        score = 0.0
        # Smaller grids are generally easier
        for ex in train:
            ih, iw = dims(ex["input"])
            oh, ow = dims(ex["output"])
            score += (ih * iw + oh * ow) / 2

        score /= len(train)

        # Same-size transformations are often simpler
        all_same_dims = all(
            dims(ex["input"]) == dims(ex["output"]) for ex in train
        )
        if all_same_dims:
            score *= 0.7

        return score

    def _try_fast_path(self, task: Dict) -> Optional[Grid]:
        """Layer 1: Try to solve without LLM using known transforms."""
        train = task["train_examples"]
        test_input = task["test_input"]

        inputs = [ex["input"] for ex in train]
        outputs = [ex["output"] for ex in train]

        # Try direct transforms (input → output as known transform chain)
        result = try_direct_transforms(inputs, outputs, test_input)
        if result is not None:
            if validate_prediction(train, test_input, result):
                return result

        # Try zoom-wrapped: strip zoom from outputs, solve simplified, re-zoom
        result = try_zoom_wrapped_transforms(inputs, outputs, test_input)
        if result is not None:
            if validate_prediction(train, test_input, result):
                return result

        return None

    def _try_llm_path(self, task: Dict, time_budget: float) -> Optional[Grid]:
        """Layer 2 + 3: Chain stripping + LLM solving."""
        if not self.llm.available:
            return None

        train = task["train_examples"]
        test_input = task["test_input"]
        inputs = [ex["input"] for ex in train]
        outputs = [ex["output"] for ex in train]

        start = time.time()

        # Gather chain hints from structural analysis
        chain_hints = self._detect_chain_hints(inputs, outputs)

        # Layer 2: Try to detect output chain and simplify
        output_chain = detect_output_chain(inputs, outputs)
        if output_chain:
            stripped_outputs = outputs
            for step in reversed(output_chain):
                new_stripped = []
                for out in stripped_outputs:
                    if step["name"] == "zoom_2x":
                        from solver.transforms import downsample_2x
                        new_stripped.append(downsample_2x(out))
                    elif step["name"] == "zoom_3x":
                        h, w = dims(out)
                        new_stripped.append(
                            [[out[r*3][c*3] for c in range(w//3)] for r in range(h//3)]
                        )
                    else:
                        new_stripped = None
                        break
                if new_stripped is None:
                    break
                stripped_outputs = new_stripped

            if stripped_outputs is not None and stripped_outputs != outputs:
                simplified_train = [
                    {"input": ex["input"], "output": so}
                    for ex, so in zip(train, stripped_outputs)
                ]
                chain_desc = f"Original outputs had these transforms applied: {[s['name'] for s in output_chain]}"

                remaining = time_budget - (time.time() - start)
                llm_result = self.llm.solve_simplified(
                    simplified_train, test_input, chain_desc,
                    time_budget=remaining * 0.8,
                )
                if llm_result is not None:
                    final = apply_chain(llm_result, output_chain)
                    if validate_prediction(train, test_input, final):
                        return final

        # Layer 3: Full LLM solving with chain hints
        remaining = time_budget - (time.time() - start)
        if remaining < 5:
            return None

        result = self.llm.solve(
            train, test_input, time_budget=remaining,
            chain_hints=chain_hints if chain_hints else None,
        )
        if result is not None:
            if validate_prediction(train, test_input, result):
                return result
            if is_valid(result):
                return result

        return None

    def _detect_chain_hints(self, inputs, outputs):
        """Detect structural hints about what transforms may be involved."""
        hints = []

        if not inputs or not outputs:
            return hints

        ih0, iw0 = dims(inputs[0])
        oh0, ow0 = dims(outputs[0])

        # Size-based hints
        if all(dims(out) == (dims(inp)[0] * 2, dims(inp)[1] * 2) for inp, out in zip(inputs, outputs)):
            hints.append("zoom_2x")
        elif all(dims(out) == (dims(inp)[0] * 3, dims(inp)[1] * 3) for inp, out in zip(inputs, outputs)):
            hints.append("zoom_3x")
        elif oh0 == iw0 and ow0 == ih0:
            hints.append("transpose_or_rotate_90_270")

        # Color-based hints
        for out in outputs:
            oc = colors_in(out) - {0}
            non_gray = oc - {5}
            if 5 in oc and len(non_gray) == 1:
                hints.append("highlight_color")
                break

        # Gravity detection: check if non-black cells are packed to one side
        for out in outputs:
            h, w = dims(out)
            if h < 3 or w < 3:
                continue
            # Check gravity_down: non-black cells should be at bottom
            cols_packed_down = 0
            for c in range(w):
                found_gap = False
                for r in range(h):
                    if out[r][c] != 0:
                        if found_gap:
                            break
                    else:
                        if any(out[rr][c] != 0 for rr in range(r)):
                            found_gap = True
                else:
                    cols_packed_down += 1
            if cols_packed_down > w * 0.8:
                hints.append("gravity")
                break

        return hints

    def _fallback(self, task: Dict) -> Optional[Grid]:
        """Last resort heuristic fallback.

        Strategy: Try to return something with correct dimensions.
        Even a wrong answer with right dimensions scores better than nothing
        if partial-credit were ever added.
        """
        train = task["train_examples"]
        test_input = task["test_input"]

        expected_dims = infer_output_dims(train, test_input)

        if expected_dims is not None:
            eh, ew = expected_dims
            th, tw = dims(test_input)

            # If same dims, return input copy (sometimes the transform is identity-like)
            if eh == th and ew == tw:
                return deep_copy(test_input)

            # If output is 2x input, try zoom_2x
            if eh == th * 2 and ew == tw * 2:
                from solver.transforms import zoom_2x
                return zoom_2x(test_input)

            # If output is 3x input, try zoom_3x
            if eh == th * 3 and ew == tw * 3:
                from solver.transforms import zoom_3x
                return zoom_3x(test_input)

            # Otherwise return black grid of expected size
            return [[0] * ew for _ in range(eh)]

        # Ultimate fallback: return input
        return deep_copy(test_input)
