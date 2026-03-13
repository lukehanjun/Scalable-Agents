from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .benchmarks import load_jsonl_tasks, run_live_benchmark, run_live_benchmark_auto
from .evaluator import GraphEvaluator
from .grammar import GraphFactory
from .models import BudgetConstraints, HardwareProfile, SearchConfig
from .parallel import ParallelEvaluationRuntime
from .search import EvolutionarySearch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budget-aware agentic graph search")
    parser.add_argument("--task", required=True, help="Task description to optimize for")
    parser.add_argument("--max-tokens", type=int, default=2_000)
    parser.add_argument("--max-latency-ms", type=int, default=9_000)
    parser.add_argument("--population-size", type=int, default=18)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpu-cores", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--live-auto", action="store_true", help="Run self-contained SWE-bench benchmark mode.")
    parser.add_argument("--question-count", type=int, default=50)
    parser.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--harness-workers", type=int, default=4)
    parser.add_argument("--live-tasks-path")
    parser.add_argument("--repo-root")
    parser.add_argument("--output-dir", default="artifacts/live-benchmark")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budget = BudgetConstraints(
        max_tokens=args.max_tokens,
        max_latency_ms=args.max_latency_ms,
    )
    config = SearchConfig(
        population_size=args.population_size,
        generations=args.generations,
        trials=args.trials,
        random_seed=args.seed,
    )
    hardware = HardwareProfile(
        gpu_count=args.gpus,
        cpu_cores=args.cpu_cores,
    )

    if args.live_auto:
        result = run_live_benchmark_auto(
            workspace_root=Path.cwd(),
            question_count=args.question_count,
            budget=budget,
            config=config,
            hardware=hardware,
            model_name=args.model,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            harness_max_workers=args.harness_workers,
        )
    elif args.live_tasks_path:
        tasks = load_jsonl_tasks(args.live_tasks_path, limit=args.limit)
        result = run_live_benchmark(
            tasks=tasks,
            budget=budget,
            config=config,
            hardware=hardware,
            model_name=args.model,
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            run_harness=True,
            harness_max_workers=args.harness_workers,
        )
    else:
        search = EvolutionarySearch(
            factory=GraphFactory(seed=args.seed),
            evaluator=GraphEvaluator(),
            runtime=ParallelEvaluationRuntime(),
        )
        result = search.run(
            task_description=args.task,
            budget=budget,
            config=config,
            hardware=hardware,
        )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
