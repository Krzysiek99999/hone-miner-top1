"""
Shared utilities for ARC-AGI-2 Sandbox Runner

- `load_input_data(input_dir)`: reads `miner_current_dataset.json`
- `save_output_data(results, output_dir)`: writes `results.json`

DO NOT MODIFY — validator expects these exact behaviors.
"""

import json
from pathlib import Path
from typing import Dict, Any


def save_output_data(results: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[IO] Saved output to: {output_path}")


def load_input_data(input_dir: Path) -> Dict[str, Any]:
    all_files = list(input_dir.glob("*"))
    print("[IO] Files in input dir:", all_files)

    dataset_file = input_dir / "miner_current_dataset.json"
    if dataset_file.exists():
        print(f"[IO] Loading dataset: {dataset_file}")
        with open(dataset_file, "r") as f:
            data = json.load(f)
        return data

    raise FileNotFoundError(f"No input data found in {input_dir}")
