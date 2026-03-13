from __future__ import annotations

from .models import TaskProfile
from .utils import stable_hash


class TaskProfiler:
    SOFTWARE_HINTS = {
        "bug", "repo", "issue", "patch", "test", "failing", "python", "typescript",
        "refactor", "compiler", "api", "docker", "build", "swe-bench", "benchmark",
    }
    RETRIEVAL_HINTS = {"research", "docs", "compare", "survey", "benchmark", "retrieve", "search"}
    VERIFICATION_HINTS = {"verify", "robust", "test", "check", "critic", "judge", "prove"}
    PARALLEL_HINTS = {"multi", "many", "batch", "parallel", "benchmark", "suite", "shard"}

    @classmethod
    def profile(cls, description: str) -> TaskProfile:
        lowered = description.lower()
        keywords = sorted({word.strip(".,:;!?") for word in lowered.split() if len(word) > 3})
        matched_software = cls.SOFTWARE_HINTS.intersection(keywords)
        matched_retrieval = cls.RETRIEVAL_HINTS.intersection(keywords)
        matched_verification = cls.VERIFICATION_HINTS.intersection(keywords)
        matched_parallel = cls.PARALLEL_HINTS.intersection(keywords)

        domain = "software_engineering" if matched_software else "general"
        complexity = 2
        if len(description) > 150:
            complexity += 1
        if any(term in lowered for term in ("benchmark", "multi-agent", "hierarchy", "evolution", "swe-bench")):
            complexity += 1
        if any(term in lowered for term in ("scalable", "distributed", "gpu", "parallel")):
            complexity += 1
        complexity = min(complexity, 5)

        return TaskProfile(
            task_id=f"task-{stable_hash(description) % 10_000_000}",
            description=description,
            domain=domain,
            complexity=complexity,
            need_retrieval=bool(matched_retrieval or "swe-bench" in lowered),
            need_verification=bool(matched_verification or domain == "software_engineering"),
            parallelism_potential=min(1.0, 0.3 + 0.15 * len(matched_parallel) + 0.1 * (complexity - 2)),
            robustness_sensitivity=min(1.0, 0.35 + 0.15 * len(matched_verification) + 0.1 * ("benchmark" in lowered)),
            benchmark_family="swebench" if "swe-bench" in lowered else "generic",
            keywords=keywords,
        )
