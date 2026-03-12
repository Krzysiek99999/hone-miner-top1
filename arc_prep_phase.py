"""
ARC-AGI-2 PREP PHASE — Download models and assets.

Internet access is available during this phase.
Downloads the LLM model for inference phase.
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config import SOLVER_MODEL, MODEL_CACHE_DIR

MODEL_NAME = SOLVER_MODEL
CACHE_DIR = Path(MODEL_CACHE_DIR)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
def download_model_with_retry(repo_id: str, cache_dir: str, local_dir: str) -> str:
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
    )


def run_prep_phase(cache_dir: Path = CACHE_DIR) -> None:
    print("\n" + "=" * 60)
    print("PREP PHASE — Downloading Models / Assets")
    print("=" * 60)

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = cache_dir / MODEL_NAME.replace("/", "--")

    print(f"\n  Model: {MODEL_NAME}")
    print(f"  Cache: {cache_dir}")
    print(f"  Local: {local_dir}")

    # Check if already downloaded
    if local_dir.exists() and any(local_dir.iterdir()):
        files_count = len(list(local_dir.glob("*")))
        if files_count >= 10:
            print(f"\n  Model already cached ({files_count} files), skipping download")
            print("\n" + "=" * 60)
            print("PREP PHASE COMPLETED — Status: success")
            print("=" * 60)
            return
        else:
            print(f"\n  Partial download detected ({files_count} files), resuming...")

    try:
        print("\n  Downloading from Hugging Face...")
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = download_model_with_retry(
            repo_id=MODEL_NAME,
            cache_dir=str(cache_dir),
            local_dir=str(local_dir),
        )
        files_count = len(list(Path(downloaded_path).glob("*")))
        print(f"  Downloaded to: {downloaded_path} ({files_count} files)")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("PREP PHASE COMPLETED — Status: failed")
        print("=" * 60)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("PREP PHASE COMPLETED — Status: success")
    print("=" * 60)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="ARC-AGI-2 Prep Phase")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    print(f"\nPhase: prep | Input: {args.input} | Output: {args.output}")
    run_prep_phase()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_cli())
    except Exception as e:
        print(f"\nERROR (prep): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
