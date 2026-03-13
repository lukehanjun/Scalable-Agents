from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    role: str
    base_token_cost: int
    base_latency_ms: int
    robustness_bonus: float
    quality_bonus: float
    default_tools: tuple[str, ...] = ()


OPERATOR_LIBRARY: dict[str, OperatorSpec] = {
    "planner": OperatorSpec("planner", "Task planner", 220, 700, 0.03, 0.12),
    "decomposer": OperatorSpec("decomposer", "Subtask decomposer", 160, 520, 0.02, 0.08),
    "retriever": OperatorSpec("retriever", "Context retriever", 260, 800, 0.05, 0.1, ("search",)),
    "worker": OperatorSpec("worker", "Execution worker", 300, 1_050, 0.01, 0.09, ("editor", "tests")),
    "critic": OperatorSpec("critic", "Draft critic", 180, 580, 0.08, 0.06),
    "verifier": OperatorSpec("verifier", "Evidence verifier", 210, 760, 0.11, 0.07, ("tests",)),
    "judge": OperatorSpec("judge", "Final arbiter", 150, 560, 0.12, 0.05),
    "synthesizer": OperatorSpec("synthesizer", "Final response composer", 170, 500, 0.03, 0.07),
    "tool-user": OperatorSpec("tool-user", "Tool-focused specialist", 240, 840, 0.04, 0.08, ("shell", "git")),
}


def model_mix_factor(model_mix: str) -> tuple[float, float, float]:
    if model_mix == "quality":
        return 1.25, 1.2, 1.18
    if model_mix == "cheap":
        return 0.82, 0.85, 0.88
    return 1.0, 1.0, 1.0
