"""
ARC-AGI-2 INFERENCE PHASE — Solve problems using 3-layer pipeline.

Internet is BLOCKED during this phase.
Uses pre-downloaded models and transform detection.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

from arc_utils import load_input_data, save_output_data
from solver.orchestrator import Orchestrator


def run_inference_phase(input_dir: Path, output_dir: Path) -> None:
    import os
    print("\n" + "=" * 60)
    print("INFERENCE PHASE — Solving ARC-AGI-2 Problems")
    print("=" * 60)

    # Diagnostic info for debugging sandbox issues
    print(f"\n[ENV] VLLM_API_BASE = {os.environ.get('VLLM_API_BASE', '(not set)')}")
    print(f"[ENV] SOLVER_MODEL  = {os.environ.get('SOLVER_MODEL', '(not set)')}")
    print(f"[ENV] HF_HOME       = {os.environ.get('HF_HOME', '(not set)')}")
    print(f"[ENV] Python version = {sys.version}")
    print(f"[ENV] Working dir    = {os.getcwd()}")

    try:
        print(f"\n[1/3] Loading input data from {input_dir}...")
        data = load_input_data(input_dir)
        problems: List[Dict[str, Any]] = data["tasks"]
        print(f"       Loaded {len(problems)} problems")

        print("[2/3] Initializing solver pipeline (connecting to vLLM)...")
        orchestrator = Orchestrator()

        print("[3/3] Solving problems...")
        predictions = orchestrator.solve_all(problems)

        # Save results
        num_solved = sum(
            1 for p in predictions if p.get("predicted_output") is not None
        )
        results = {
            "phase": "inference",
            "status": "success",
            "num_problems_solved": num_solved,
            "total_problems": len(problems),
            "vllm_available": orchestrator.llm.available,
            "predictions": predictions,
        }
        save_output_data(results, output_dir)

        print("\n" + "=" * 60)
        print(f"INFERENCE PHASE COMPLETED — {num_solved}/{len(problems)} solved")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: Inference phase failed: {e}")
        import traceback
        traceback.print_exc()

        results = {
            "phase": "inference",
            "status": "failed",
            "error": str(e),
            "predictions": [],
        }
        save_output_data(results, output_dir)

        print("\n" + "=" * 60)
        print("INFERENCE PHASE COMPLETED — Status: failed")
        print("=" * 60)
        sys.exit(1)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Inference Phase")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    print(f"\nPhase: inference | Input: {args.input} | Output: {args.output}")
    run_inference_phase(Path(args.input), Path(args.output))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_cli())
    except Exception as e:
        print(f"\nERROR (inference): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
