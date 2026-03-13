from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class GraphNode:
    node_id: str
    operator: str
    role: str
    layer: int
    model_tier: str = "balanced"
    temperature: float = 0.2
    tools: list[str] = field(default_factory=list)
    memory_scope: str = "local"
    prompt_style: str = "precise"


@dataclass
class GraphEdge:
    source: str
    target: str
    channel: str = "full"


@dataclass
class AgentGraph:
    graph_id: str
    template: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    metadata: dict[str, Any] = field(default_factory=dict)
    stopping_rules: dict[str, int] = field(default_factory=lambda: {"max_rounds": 2})

    def clone(self, graph_id: str | None = None) -> "AgentGraph":
        return AgentGraph(
            graph_id=graph_id or f"graph-{uuid4().hex[:8]}",
            template=self.template,
            nodes=[GraphNode(**vars(node)) for node in self.nodes],
            edges=[GraphEdge(**vars(edge)) for edge in self.edges],
            metadata=dict(self.metadata),
            stopping_rules=dict(self.stopping_rules),
        )

    def operators(self) -> list[str]:
        return [node.operator for node in self.nodes]

    def has_operator(self, operator: str) -> bool:
        return any(node.operator == operator for node in self.nodes)

    def worker_count(self) -> int:
        return sum(1 for node in self.nodes if node.operator == "worker")

    def max_layer(self) -> int:
        return max((node.layer for node in self.nodes), default=0)

    def edge_density(self) -> float:
        if len(self.nodes) < 2:
            return 0.0
        max_edges = len(self.nodes) * (len(self.nodes) - 1) / 2
        return len(self.edges) / max_edges


@dataclass
class BudgetConstraints:
    max_tokens: int = 2_000
    max_latency_ms: int = 10_000
    max_cost: float = 4.0


@dataclass
class TaskProfile:
    task_id: str
    description: str
    domain: str
    complexity: int
    need_retrieval: bool
    need_verification: bool
    parallelism_potential: float
    robustness_sensitivity: float
    benchmark_family: str = "generic"
    keywords: list[str] = field(default_factory=list)


@dataclass
class EvaluationMetrics:
    quality: float
    tokens: int
    latency_ms: int
    cost: float
    variance: float
    robustness: float
    feasible: bool
    scalar_score: float
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class CandidateResult:
    graph: AgentGraph
    metrics: EvaluationMetrics
    device_id: str = "cpu:0"


@dataclass
class MutationEvent:
    generation: int
    event_index: int
    operation: str
    description: str
    parent_ids: list[str]
    child_graph: AgentGraph
    score_hint: float | None = None


@dataclass
class GenerationSummary:
    generation: int
    best_graph_id: str
    best_score: float
    feasible_count: int
    average_score: float
    frontier_size: int


@dataclass
class HardwareProfile:
    gpu_count: int = 1
    cpu_cores: int = 8
    per_gpu_concurrency: int = 4
    interconnect: str = "nvlink-or-pcie"


@dataclass
class ExecutionPlan:
    strategy: str
    island_count: int
    workers_per_island: int
    communication_pattern: str
    migration_interval: int
    rationale: str


@dataclass
class SearchConfig:
    population_size: int = 24
    generations: int = 10
    mutation_rate: float = 0.72
    crossover_rate: float = 0.28
    elitism: int = 4
    tournament_size: int = 3
    trials: int = 3
    max_workers_per_island: int = 8
    random_seed: int = 7


@dataclass
class SearchRun:
    task_profile: TaskProfile
    budget: BudgetConstraints
    hardware: HardwareProfile
    execution_plan: ExecutionPlan
    best_candidate: CandidateResult
    baseline_candidate: CandidateResult
    pareto_front: list[CandidateResult]
    generation_summaries: list[GenerationSummary]
    mutation_events: list[MutationEvent]
    condensed_events: list[MutationEvent]


@dataclass
class RepoSnippet:
    path: str
    content: str
    score: float


@dataclass
class ExecutionTraceStep:
    node_id: str
    operator: str
    action_type: str
    prompt_preview: str
    output_preview: str
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveExecutionResult:
    instance_id: str
    graph_id: str
    model_name: str
    repo_path: str | None
    final_patch: str
    final_response: str
    trace: list[ExecutionTraceStep]
    total_input_tokens: int
    total_output_tokens: int
    latency_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveBenchmarkCaseResult:
    instance_id: str
    repo: str
    evolved_graph_id: str
    evolved_patch_nonempty: bool
    baseline_patch_nonempty: bool
    evolved_input_tokens: int
    baseline_input_tokens: int
    evolved_output_tokens: int
    baseline_output_tokens: int
    evolved_patch_path: str
    baseline_patch_path: str


@dataclass
class LiveBenchmarkSummary:
    task_count: int
    model_name: str
    evolved_predictions_path: str
    baseline_predictions_path: str
    evolved_nonempty_patches: int
    baseline_nonempty_patches: int
    evolved_avg_input_tokens: float
    baseline_avg_input_tokens: float
    evolved_avg_output_tokens: float
    baseline_avg_output_tokens: float
    harness_command_evolved: str
    harness_command_baseline: str
    case_results: list[LiveBenchmarkCaseResult]
    case_details: list[dict[str, Any]] = field(default_factory=list)
