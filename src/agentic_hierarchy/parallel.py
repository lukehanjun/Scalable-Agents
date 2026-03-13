from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .evaluator import GraphEvaluator
from .models import AgentGraph, BudgetConstraints, CandidateResult, ExecutionPlan, HardwareProfile, SearchConfig, TaskProfile


class ParallelEvaluationRuntime:
    def make_plan(self, hardware: HardwareProfile, config: SearchConfig, population_size: int) -> ExecutionPlan:
        if hardware.gpu_count <= 1:
            return ExecutionPlan(
                strategy="local_thread_pool",
                island_count=1,
                workers_per_island=min(config.max_workers_per_island, max(1, hardware.cpu_cores)),
                communication_pattern="shared in-memory cache",
                migration_interval=max(2, config.generations),
                rationale="Single accelerator or CPU-only setup, so population evaluation stays local and parallelizes over candidates/trials.",
            )
        island_count = min(hardware.gpu_count, max(1, population_size // 4))
        return ExecutionPlan(
            strategy="gpu_island_model",
            island_count=island_count,
            workers_per_island=min(config.max_workers_per_island, max(1, hardware.per_gpu_concurrency)),
            communication_pattern="round-robin island sharding with low-frequency elite migration",
            migration_interval=2,
            rationale="Multiple GPUs favor island evolution. Each GPU gets a shard, evaluates locally, and exchanges only a few elites to reduce synchronization overhead.",
        )

    def evaluate_population(
        self,
        population: list[AgentGraph],
        evaluator: GraphEvaluator,
        profile: TaskProfile,
        budget: BudgetConstraints,
        config: SearchConfig,
        hardware: HardwareProfile,
        plan: ExecutionPlan,
    ) -> list[CandidateResult]:
        assignments = self._assign_devices(population, hardware, plan)
        max_workers = max(1, plan.island_count * plan.workers_per_island)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    evaluator.evaluate,
                    graph,
                    profile,
                    budget,
                    config.trials,
                    device_id,
                )
                for graph, device_id in assignments
            ]
        return [future.result() for future in futures]

    @staticmethod
    def _assign_devices(
        population: list[AgentGraph],
        hardware: HardwareProfile,
        plan: ExecutionPlan,
    ) -> list[tuple[AgentGraph, str]]:
        assignments: list[tuple[AgentGraph, str]] = []
        for index, graph in enumerate(population):
            if hardware.gpu_count > 0:
                island_index = index % max(1, plan.island_count)
                device_id = f"gpu:{island_index % hardware.gpu_count}"
            else:
                device_id = f"cpu:{index % max(1, hardware.cpu_cores)}"
            assignments.append((graph, device_id))
        return assignments
