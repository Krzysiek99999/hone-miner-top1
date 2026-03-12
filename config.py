"""Solver configuration — all tunable parameters in one place.

Override via environment variables.
"""

import os


# ============= MODEL =============
# Model to download and serve via vLLM
# Good options:
#   1xH200: "Qwen/Qwen2.5-32B-Instruct" (~64GB FP16)
#   2xH200: "Qwen/Qwen2.5-72B-Instruct" (~144GB FP16)
#   4xH200: "Qwen/Qwen3-235B-A22B" (~470GB FP16, MoE)
SOLVER_MODEL = os.environ.get("SOLVER_MODEL", "Qwen/Qwen2.5-72B-Instruct")

# vLLM connection
VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://vllm-container:8000")

# Model cache directory
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/app/models")

# ============= vLLM CONFIG (for /info endpoint) =============
VLLM_DTYPE = os.environ.get("VLLM_DTYPE", "auto")
VLLM_GPU_MEMORY_UTIL = float(os.environ.get("VLLM_GPU_MEMORY_UTIL", "0.90"))
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384"))
WEIGHT_CLASS = os.environ.get("WEIGHT_CLASS", "2xH200")

# ============= SOLVER =============
# Total time budget for inference phase (seconds)
TOTAL_TIME_BUDGET = float(os.environ.get("TOTAL_TIME_BUDGET", "3500"))

# Max time per task for LLM (seconds)
MAX_TASK_TIME = float(os.environ.get("MAX_TASK_TIME", "60"))

# Number of voting attempts for direct LLM solving
VOTE_ATTEMPTS = int(os.environ.get("VOTE_ATTEMPTS", "5"))

# Number of program synthesis attempts
PROGRAM_ATTEMPTS = int(os.environ.get("PROGRAM_ATTEMPTS", "3"))

# Max tokens for LLM response
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))

# LLM temperature range for voting (start, step)
TEMP_START = float(os.environ.get("TEMP_START", "0.3"))
TEMP_STEP = float(os.environ.get("TEMP_STEP", "0.15"))


def vllm_config_dict():
    """Return vLLM config dict for /info endpoint."""
    return {
        "model": SOLVER_MODEL,
        "dtype": VLLM_DTYPE,
        "gpu_memory_utilization": VLLM_GPU_MEMORY_UTIL,
        "max_model_len": VLLM_MAX_MODEL_LEN,
    }


def info_dict(repo_url: str, repo_branch: str = "main"):
    """Return full /info response dict."""
    return {
        "repo_url": repo_url,
        "repo_branch": repo_branch,
        "weight_class": WEIGHT_CLASS,
        "use_vllm": True,
        "vllm_config": vllm_config_dict(),
        "version": "1.0.0",
    }
