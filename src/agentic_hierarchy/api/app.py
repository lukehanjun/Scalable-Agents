from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..benchmarks import (
    load_jsonl_tasks,
    run_live_benchmark_auto,
    run_surrogate_benchmark,
)
from ..executor import LiveGraphExecutor
from ..evaluator import GraphEvaluator
from ..grammar import GraphFactory
from ..llm import OpenAIResponsesClient
from ..models import BudgetConstraints, HardwareProfile, SearchConfig
from ..parallel import ParallelEvaluationRuntime
from ..search import EvolutionarySearch


class BudgetPayload(BaseModel):
    max_tokens: int = 2_000
    max_latency_ms: int = 10_000
    max_cost: float = 4.0


class SearchPayload(BaseModel):
    population_size: int = 20
    generations: int = 8
    trials: int = 3
    mutation_rate: float = 0.72
    crossover_rate: float = 0.28
    elitism: int = 4
    tournament_size: int = 3
    max_workers_per_island: int = 8
    random_seed: int = 7


class HardwarePayload(BaseModel):
    gpu_count: int = 1
    cpu_cores: int = 8
    per_gpu_concurrency: int = 4
    interconnect: str = "nvlink-or-pcie"


class SearchRequest(BaseModel):
    task: str = Field(..., min_length=12)
    budget: BudgetPayload = Field(default_factory=BudgetPayload)
    search: SearchPayload = Field(default_factory=SearchPayload)
    hardware: HardwarePayload = Field(default_factory=HardwarePayload)


class BenchmarkRequest(BaseModel):
    tasks_path: str
    limit: int = 8
    budget: BudgetPayload = Field(default_factory=BudgetPayload)
    search: SearchPayload = Field(default_factory=SearchPayload)
    hardware: HardwarePayload = Field(default_factory=HardwarePayload)


class LiveTaskRequest(BaseModel):
    task: str = Field(..., min_length=12)
    repo_path: str | None = None
    model_name: str = "gpt-5-mini"
    budget: BudgetPayload = Field(default_factory=BudgetPayload)
    search: SearchPayload = Field(default_factory=SearchPayload)
    hardware: HardwarePayload = Field(default_factory=HardwarePayload)


class LiveBenchmarkRequest(BaseModel):
    question_count: int = Field(default=50, ge=1, le=500)
    model_name: str = "gpt-5-mini"
    output_dir: str = "artifacts/live-benchmark"
    dataset_name: str = "princeton-nlp/SWE-bench_Lite"
    dataset_split: str = "test"
    harness_max_workers: int = Field(default=4, ge=1, le=64)
    budget: BudgetPayload = Field(default_factory=BudgetPayload)
    search: SearchPayload = Field(default_factory=SearchPayload)
    hardware: HardwarePayload = Field(default_factory=HardwarePayload)


app = FastAPI(title="Agentic Hierarchy Lab", version="0.1.0")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/search/run")
def run_search(payload: SearchRequest) -> dict:
    search = EvolutionarySearch(
        factory=GraphFactory(seed=payload.search.random_seed),
        evaluator=GraphEvaluator(),
        runtime=ParallelEvaluationRuntime(),
    )
    result = search.run(
        task_description=payload.task,
        budget=BudgetConstraints(**payload.budget.model_dump()),
        config=SearchConfig(**payload.search.model_dump()),
        hardware=HardwareProfile(**payload.hardware.model_dump()),
    )
    return asdict(result)


@app.post("/api/benchmark/compare")
def compare_benchmark(payload: BenchmarkRequest) -> dict:
    tasks = load_jsonl_tasks(payload.tasks_path, limit=payload.limit)
    summary = run_surrogate_benchmark(
        tasks=tasks,
        budget=BudgetConstraints(**payload.budget.model_dump()),
        config=SearchConfig(**payload.search.model_dump()),
        hardware=HardwareProfile(**payload.hardware.model_dump()),
    )
    return asdict(summary)


@app.post("/api/live/execute")
def execute_live(payload: LiveTaskRequest) -> dict:
    search = EvolutionarySearch(
        factory=GraphFactory(seed=payload.search.random_seed),
        evaluator=GraphEvaluator(),
        runtime=ParallelEvaluationRuntime(),
    )
    budget = BudgetConstraints(**payload.budget.model_dump())
    config = SearchConfig(**payload.search.model_dump())
    hardware = HardwareProfile(**payload.hardware.model_dump())
    search_result = search.run(
        task_description=payload.task,
        budget=budget,
        config=config,
        hardware=hardware,
    )
    executor = LiveGraphExecutor(llm=OpenAIResponsesClient(model_name=payload.model_name))
    evolved = executor.execute_graph(
        graph=search_result.best_candidate.graph,
        instance_id="adhoc-task",
        problem_statement=payload.task,
        repo_path=payload.repo_path,
        budget=budget,
    )
    baseline = executor.execute_single_agent(
        instance_id="adhoc-task",
        problem_statement=payload.task,
        repo_path=payload.repo_path,
        budget=budget,
    )
    return {
        "search_run": asdict(search_result),
        "evolved_execution": asdict(evolved),
        "baseline_execution": asdict(baseline),
    }


@app.post("/api/benchmark/run-live")
def run_live_benchmark_api(payload: LiveBenchmarkRequest) -> dict:
    workspace_root = Path(__file__).resolve().parents[3]
    try:
        summary = run_live_benchmark_auto(
            workspace_root=workspace_root,
            question_count=payload.question_count,
            budget=BudgetConstraints(**payload.budget.model_dump()),
            config=SearchConfig(**payload.search.model_dump()),
            hardware=HardwareProfile(**payload.hardware.model_dump()),
            model_name=payload.model_name,
            output_dir=payload.output_dir,
            dataset_name=payload.dataset_name,
            dataset_split=payload.dataset_split,
            harness_max_workers=payload.harness_max_workers,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return asdict(summary)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
