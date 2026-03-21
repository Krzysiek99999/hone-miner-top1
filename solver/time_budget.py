"""Time budget manager for inference phase."""

import time


class TimeBudget:
    def __init__(self, total_seconds: float = 3500.0):
        self.total = total_seconds
        self.start = time.time()
        self.task_count = 0
        self.solved_fast = 0
        self.solved_llm = 0

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return max(0.0, self.total - self.elapsed())

    def budget_for_task(self, tasks_remaining: int) -> float:
        if tasks_remaining <= 0:
            return 0.0
        remaining = self.remaining()
        # Reserve minimum 2s per remaining task
        min_reserved = max(0, (tasks_remaining - 1) * 2)
        available = remaining - min_reserved
        # Cap at 60s per task — 8 program synthesis attempts need more time
        return max(3.0, min(60.0, available))

    def should_use_llm(self, tasks_remaining: int) -> bool:
        return self.budget_for_task(tasks_remaining) >= 8.0

    def record_fast(self):
        self.solved_fast += 1
        self.task_count += 1

    def record_llm(self):
        self.solved_llm += 1
        self.task_count += 1

    def record_skip(self):
        self.task_count += 1

    def summary(self) -> str:
        return (
            f"Time: {self.elapsed():.1f}s / {self.total:.0f}s | "
            f"Fast: {self.solved_fast} | LLM: {self.solved_llm} | "
            f"Total: {self.task_count}"
        )
