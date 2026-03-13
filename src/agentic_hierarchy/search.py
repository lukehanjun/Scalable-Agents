from __future__ import annotations

import random

from .evaluator import GraphEvaluator
from .grammar import GraphFactory
from .models import (
    BudgetConstraints,
    CandidateResult,
    GenerationSummary,
    HardwareProfile,
    MutationEvent,
    SearchConfig,
    SearchRun,
)
from .parallel import ParallelEvaluationRuntime
from .profiler import TaskProfiler
from .utils import safe_mean


class EvolutionarySearch:
    def __init__(
        self,
        factory: GraphFactory | None = None,
        evaluator: GraphEvaluator | None = None,
        runtime: ParallelEvaluationRuntime | None = None,
    ) -> None:
        self.factory = factory or GraphFactory()
        self.evaluator = evaluator or GraphEvaluator()
        self.runtime = runtime or ParallelEvaluationRuntime()

    def run(
        self,
        task_description: str,
        budget: BudgetConstraints,
        config: SearchConfig | None = None,
        hardware: HardwareProfile | None = None,
    ) -> SearchRun:
        config = config or SearchConfig()
        hardware = hardware or HardwareProfile()
        rng = random.Random(config.random_seed)
        task_profile = TaskProfiler.profile(task_description)
        execution_plan = self.runtime.make_plan(hardware, config, config.population_size)

        baseline_graph = self.factory.single_agent_baseline()
        baseline_candidate = self.evaluator.evaluate(baseline_graph, task_profile, budget, config.trials)
        population = self.factory.seed_population(task_profile, config.population_size)

        mutation_events: list[MutationEvent] = []
        generation_summaries: list[GenerationSummary] = []
        pareto_front: list[CandidateResult] = []
        best_candidate = baseline_candidate
        event_index = 0

        for generation in range(config.generations):
            evaluated = self.runtime.evaluate_population(
                population=population,
                evaluator=self.evaluator,
                profile=task_profile,
                budget=budget,
                config=config,
                hardware=hardware,
                plan=execution_plan,
            )
            ranked = sorted(
                evaluated,
                key=lambda candidate: (
                    candidate.metrics.feasible,
                    candidate.metrics.scalar_score,
                    candidate.metrics.quality,
                    candidate.metrics.robustness,
                ),
                reverse=True,
            )
            if ranked and self._better(ranked[0], best_candidate):
                best_candidate = ranked[0]

            pareto_front = self._pareto_front(pareto_front + ranked)
            generation_summaries.append(
                GenerationSummary(
                    generation=generation,
                    best_graph_id=ranked[0].graph.graph_id,
                    best_score=ranked[0].metrics.scalar_score,
                    feasible_count=sum(1 for candidate in ranked if candidate.metrics.feasible),
                    average_score=round(safe_mean([candidate.metrics.scalar_score for candidate in ranked]), 4),
                    frontier_size=len(pareto_front),
                )
            )

            elites = [candidate.graph for candidate in ranked[: config.elitism]]
            next_population = list(elites)
            while len(next_population) < config.population_size:
                if rng.random() < config.crossover_rate and len(ranked) >= 2:
                    parent_a = self._tournament(ranked, config.tournament_size, rng)
                    parent_b = self._tournament(ranked, config.tournament_size, rng)
                    child, description = self.factory.crossover(parent_a.graph, parent_b.graph)
                    operation = "crossover"
                    parent_ids = [parent_a.graph.graph_id, parent_b.graph.graph_id]
                else:
                    parent = self._tournament(ranked, config.tournament_size, rng)
                    child, operation, description = self.factory.mutate(parent.graph)
                    parent_ids = [parent.graph.graph_id]
                mutation_events.append(
                    MutationEvent(
                        generation=generation,
                        event_index=event_index,
                        operation=operation,
                        description=description,
                        parent_ids=parent_ids,
                        child_graph=child,
                    )
                )
                event_index += 1
                next_population.append(child)
            population = next_population

        final_ranked = sorted(
            self.runtime.evaluate_population(
                population=population,
                evaluator=self.evaluator,
                profile=task_profile,
                budget=budget,
                config=config,
                hardware=hardware,
                plan=execution_plan,
            ),
            key=lambda candidate: (
                candidate.metrics.feasible,
                candidate.metrics.scalar_score,
                candidate.metrics.quality,
                candidate.metrics.robustness,
            ),
            reverse=True,
        )
        if final_ranked and self._better(final_ranked[0], best_candidate):
            best_candidate = final_ranked[0]
        pareto_front = self._pareto_front(pareto_front + final_ranked)
        for event in mutation_events:
            matched = next((candidate for candidate in final_ranked if candidate.graph.graph_id == event.child_graph.graph_id), None)
            if matched is not None:
                event.score_hint = matched.metrics.scalar_score

        return SearchRun(
            task_profile=task_profile,
            budget=budget,
            hardware=hardware,
            execution_plan=execution_plan,
            best_candidate=best_candidate,
            baseline_candidate=baseline_candidate,
            pareto_front=pareto_front,
            generation_summaries=generation_summaries,
            mutation_events=mutation_events,
            condensed_events=self._condense_events(mutation_events),
        )

    @staticmethod
    def _tournament(candidates: list[CandidateResult], size: int, rng: random.Random) -> CandidateResult:
        pool = rng.sample(candidates, k=min(size, len(candidates)))
        return max(pool, key=lambda candidate: (candidate.metrics.feasible, candidate.metrics.scalar_score))

    @staticmethod
    def _better(left: CandidateResult, right: CandidateResult) -> bool:
        return (
            left.metrics.feasible,
            left.metrics.scalar_score,
            left.metrics.quality,
            left.metrics.robustness,
        ) > (
            right.metrics.feasible,
            right.metrics.scalar_score,
            right.metrics.quality,
            right.metrics.robustness,
        )

    @staticmethod
    def _dominates(left: CandidateResult, right: CandidateResult) -> bool:
        if left.metrics.feasible and not right.metrics.feasible:
            return True
        if right.metrics.feasible and not left.metrics.feasible:
            return False
        no_worse = (
            left.metrics.quality >= right.metrics.quality
            and left.metrics.robustness >= right.metrics.robustness
            and left.metrics.tokens <= right.metrics.tokens
            and left.metrics.latency_ms <= right.metrics.latency_ms
            and left.metrics.variance <= right.metrics.variance
        )
        strictly_better = (
            left.metrics.quality > right.metrics.quality
            or left.metrics.robustness > right.metrics.robustness
            or left.metrics.tokens < right.metrics.tokens
            or left.metrics.latency_ms < right.metrics.latency_ms
            or left.metrics.variance < right.metrics.variance
        )
        return no_worse and strictly_better

    def _pareto_front(self, candidates: list[CandidateResult]) -> list[CandidateResult]:
        frontier: list[CandidateResult] = []
        deduped = list({candidate.graph.graph_id: candidate for candidate in candidates}.values())
        for candidate in deduped:
            if any(self._dominates(other, candidate) for other in deduped if other.graph.graph_id != candidate.graph.graph_id):
                continue
            frontier.append(candidate)
        return sorted(frontier, key=lambda candidate: candidate.metrics.scalar_score, reverse=True)[:12]

    @staticmethod
    def _condense_events(events: list[MutationEvent], target_count: int = 24) -> list[MutationEvent]:
        if len(events) <= target_count:
            return events
        selected_indexes = sorted({round(index * (len(events) - 1) / (target_count - 1)) for index in range(target_count)})
        return [events[index] for index in selected_indexes]
