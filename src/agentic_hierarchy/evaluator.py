from __future__ import annotations

import threading
from statistics import pvariance

from .models import AgentGraph, BudgetConstraints, CandidateResult, EvaluationMetrics, TaskProfile
from .operators import OPERATOR_LIBRARY, model_mix_factor
from .utils import clamp, safe_mean, stable_float


class GraphEvaluator:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, int, int, float, int], EvaluationMetrics] = {}
        self._lock = threading.Lock()

    def evaluate(
        self,
        graph: AgentGraph,
        profile: TaskProfile,
        budget: BudgetConstraints,
        trials: int = 3,
        device_id: str = "cpu:0",
    ) -> CandidateResult:
        cache_key = (
            graph.graph_id,
            profile.task_id,
            budget.max_tokens,
            budget.max_latency_ms,
            budget.max_cost,
            trials,
        )
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return CandidateResult(graph=graph, metrics=cached, device_id=device_id)

        quality_values = []
        token_estimate = self._estimate_tokens(graph)
        latency_estimate = self._estimate_latency(graph)
        robustness = self._estimate_robustness(graph, profile)
        for trial_index in range(trials):
            quality_values.append(
                self._estimate_quality(
                    graph=graph,
                    profile=profile,
                    budget=budget,
                    token_estimate=token_estimate,
                    latency_estimate=latency_estimate,
                    robustness=robustness,
                    trial_index=trial_index,
                )
            )

        quality = safe_mean(quality_values)
        variance = pvariance(quality_values) if len(quality_values) > 1 else 0.0
        cost = round(token_estimate / 1_000 * {"cheap": 0.5, "balanced": 0.85, "quality": 1.25}[graph.metadata["model_mix"]], 3)
        feasible = (
            token_estimate <= budget.max_tokens
            and latency_estimate <= budget.max_latency_ms
            and cost <= budget.max_cost
        )
        scalar_score = self._scalarize(
            quality=quality,
            tokens=token_estimate,
            latency_ms=latency_estimate,
            robustness=robustness,
            variance=variance,
            budget=budget,
            feasible=feasible,
            cost=cost,
        )
        metrics = EvaluationMetrics(
            quality=round(quality, 4),
            tokens=token_estimate,
            latency_ms=latency_estimate,
            cost=cost,
            variance=round(variance, 5),
            robustness=round(robustness, 4),
            feasible=feasible,
            scalar_score=round(scalar_score, 4),
            breakdown={
                "budget_pressure": round(token_estimate / max(1, budget.max_tokens), 4),
                "latency_pressure": round(latency_estimate / max(1, budget.max_latency_ms), 4),
                "edge_density": round(graph.edge_density(), 4),
            },
        )
        with self._lock:
            self._cache[cache_key] = metrics
        return CandidateResult(graph=graph, metrics=metrics, device_id=device_id)

    def _estimate_tokens(self, graph: AgentGraph) -> int:
        token_factor, _, _ = model_mix_factor(graph.metadata["model_mix"])
        node_tokens = sum(int(OPERATOR_LIBRARY[node.operator].base_token_cost * token_factor) for node in graph.nodes)
        edge_tokens = sum(45 if edge.channel == "summary" else 90 for edge in graph.edges)
        review_overhead = 110 if graph.has_operator("judge") and graph.has_operator("verifier") else 0
        return int(node_tokens + edge_tokens + review_overhead)

    def _estimate_latency(self, graph: AgentGraph) -> int:
        _, latency_factor, _ = model_mix_factor(graph.metadata["model_mix"])
        layer_costs: dict[int, list[int]] = {}
        for node in graph.nodes:
            layer_costs.setdefault(node.layer, []).append(int(OPERATOR_LIBRARY[node.operator].base_latency_ms * latency_factor))
        coordination_overhead = int(70 * len(graph.edges) + 55 * graph.worker_count())
        layer_latency = sum(max(costs) for costs in layer_costs.values())
        return int(layer_latency + coordination_overhead)

    def _estimate_robustness(self, graph: AgentGraph, profile: TaskProfile) -> float:
        _, _, quality_factor = model_mix_factor(graph.metadata["model_mix"])
        robustness = 0.34 + 0.06 * quality_factor
        if graph.has_operator("critic"):
            robustness += 0.08
        if graph.has_operator("verifier"):
            robustness += 0.11
        if graph.has_operator("judge"):
            robustness += 0.08
        if graph.metadata["template"] in {"tree", "layered_dag"}:
            robustness += 0.05
        if graph.metadata["summarized_ratio"] >= 0.25:
            robustness += 0.03
        robustness += 0.06 * profile.robustness_sensitivity
        return clamp(robustness, 0.0, 1.0)

    def _estimate_quality(
        self,
        graph: AgentGraph,
        profile: TaskProfile,
        budget: BudgetConstraints,
        token_estimate: int,
        latency_estimate: int,
        robustness: float,
        trial_index: int,
    ) -> float:
        _, _, quality_factor = model_mix_factor(graph.metadata["model_mix"])
        quality = 0.34
        quality += 0.08 * quality_factor
        quality += 0.08 if graph.has_operator("planner") else -0.12
        quality += 0.05 if graph.has_operator("decomposer") and profile.complexity >= 3 else 0.0
        quality += 0.11 if profile.need_retrieval and graph.has_operator("retriever") else (-0.12 if profile.need_retrieval else 0.0)
        quality += 0.13 if profile.need_verification and (graph.has_operator("critic") or graph.has_operator("verifier") or graph.has_operator("judge")) else (-0.11 if profile.need_verification else 0.0)
        quality += 0.09 if profile.domain == "software_engineering" and graph.has_operator("tool-user") else 0.0
        quality += 0.05 if graph.metadata["template"] == "layered_dag" and profile.parallelism_potential > 0.55 else 0.0
        quality += 0.06 if graph.metadata["template"] == "tree" and graph.worker_count() >= 2 else 0.0
        quality -= 0.05 if graph.metadata["template"] == "chain" and graph.worker_count() > 2 else 0.0
        quality += 0.07 if graph.worker_count() >= 3 and profile.parallelism_potential > 0.6 else 0.0
        quality += 0.04 if 0.2 <= graph.metadata["summarized_ratio"] <= 0.65 and budget.max_tokens <= 3_000 else 0.0
        quality -= 0.05 if token_estimate > budget.max_tokens else 0.0
        quality -= 0.04 if latency_estimate > budget.max_latency_ms else 0.0
        quality += 0.08 * robustness

        noise = stable_float(f"{graph.graph_id}-{profile.task_id}-{trial_index}", -1.0, 1.0)
        brittleness = max(0.02, 0.16 - 0.08 * robustness)
        quality += noise * brittleness
        return clamp(quality, 0.0, 1.0)

    def _scalarize(
        self,
        quality: float,
        tokens: int,
        latency_ms: int,
        robustness: float,
        variance: float,
        budget: BudgetConstraints,
        feasible: bool,
        cost: float,
    ) -> float:
        token_penalty = max(0.0, tokens / max(1, budget.max_tokens) - 1.0) * 0.25
        latency_penalty = max(0.0, latency_ms / max(1, budget.max_latency_ms) - 1.0) * 0.2
        cost_penalty = max(0.0, cost / max(0.01, budget.max_cost) - 1.0) * 0.12
        score = quality + 0.14 * robustness - 0.18 * variance - token_penalty - latency_penalty - cost_penalty
        score += 0.08 if feasible else -0.08
        return score
