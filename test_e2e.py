#!/usr/bin/env python3
"""End-to-end test: generate problems, save as dataset, run inference."""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add Hone repo to path
HONE_PATH = os.path.join(os.path.dirname(__file__), "..", "hone", "sandbox_runner")
sys.path.insert(0, HONE_PATH)
sys.path.insert(0, os.path.dirname(__file__))

from synthetics.arc_agi2_generator import ARC2Generator


def generate_dataset(count=20, seed=123):
    """Generate a test dataset in miner_current_dataset.json format."""
    gen = ARC2Generator(max_chain_length=5, max_grid_size=30, seed=seed)
    tasks = []
    expected = []

    for i in range(count * 3):
        if len(tasks) >= count:
            break
        try:
            ps = gen.generate_problem_set(num_train=3, num_test=1)
            task = {
                "task_hash": f"test_{len(tasks):04d}",
                "train_examples": ps["train_examples"],
                "test_input": ps["test_input"],
                "metadata": ps.get("metadata", {}),
            }
            tasks.append(task)
            expected.append(ps["test_output"])
        except Exception:
            continue

    return {"tasks": tasks}, expected


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    print(f"Generating {count} problems...")
    dataset, expected_outputs = generate_dataset(count)
    print(f"Generated {len(dataset['tasks'])} problems")

    # Save to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        dataset_path = input_dir / "miner_current_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        print(f"\nRunning inference pipeline...")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}\n")

        # Run inference
        from arc_inference_phase import run_inference_phase
        run_inference_phase(input_dir, output_dir)

        # Load results
        results_path = output_dir / "results.json"
        with open(results_path) as f:
            results = json.load(f)

        # Score results
        predictions = results.get("predictions", [])
        exact_matches = 0
        total_with_output = 0

        for pred, expected in zip(predictions, expected_outputs):
            predicted = pred.get("predicted_output")
            if predicted is not None:
                total_with_output += 1
                if predicted == expected:
                    exact_matches += 1

        total = len(expected_outputs)
        print(f"\n{'='*50}")
        print(f"RESULTS")
        print(f"{'='*50}")
        print(f"Total problems: {total}")
        print(f"Predictions made: {total_with_output}")
        print(f"Exact matches: {exact_matches}")
        print(f"Exact match rate: {exact_matches/total*100:.1f}%")
        print(f"Status: {results.get('status')}")
        print(f"vLLM available: {results.get('vllm_available')}")


if __name__ == "__main__":
    main()
