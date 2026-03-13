from __future__ import annotations

import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from .executor import LiveGraphExecutor
from .grammar import GraphFactory
from .llm import OpenAIResponsesClient
from .models import (
    BudgetConstraints,
    HardwareProfile,
    LiveBenchmarkCaseResult,
    LiveBenchmarkSummary,
    SearchConfig,
)
from .repo_tools import resolve_repo_path
from .search import EvolutionarySearch
from .swebench_auto import (
    ensure_swebench_harness,
    load_swebench_tasks,
    prepare_task_repositories,
    run_harness_evaluation,
)


@dataclass
class BenchmarkTask:
    instance_id: str
    problem_statement: str
    repo: str = ""
    repo_path: str | None = None
    base_commit: str | None = None
    reference_patch: str | None = None


@dataclass
class BenchmarkComparisonSummary:
    task_count: int
    evolved_wins: int
    baseline_wins: int
    ties: int
    average_evolved_score: float
    average_baseline_score: float


def load_jsonl_tasks(path: str | Path, limit: int | None = None) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            record = json.loads(line)
            tasks.append(
                BenchmarkTask(
                    instance_id=record.get("instance_id", f"instance-{index}"),
                    problem_statement=record.get("problem_statement") or record.get("text") or record.get("prompt", ""),
                    repo=record.get("repo", ""),
                    repo_path=record.get("repo_path"),
                    base_commit=record.get("base_commit"),
                    reference_patch=record.get("patch") or record.get("reference_patch"),
                )
            )
    return tasks


def run_surrogate_benchmark(
    tasks: list[BenchmarkTask],
    budget: BudgetConstraints,
    config: SearchConfig | None = None,
    hardware: HardwareProfile | None = None,
) -> BenchmarkComparisonSummary:
    config = config or SearchConfig(generations=4, population_size=10, trials=2)
    hardware = hardware or HardwareProfile()
    search = EvolutionarySearch(factory=GraphFactory(seed=config.random_seed))
    evolved_scores: list[float] = []
    baseline_scores: list[float] = []
    evolved_wins = baseline_wins = ties = 0

    for task in tasks:
        result = search.run(
            task_description=task.problem_statement,
            budget=budget,
            config=config,
            hardware=hardware,
        )
        evolved_scores.append(result.best_candidate.metrics.scalar_score)
        baseline_scores.append(result.baseline_candidate.metrics.scalar_score)
        if result.best_candidate.metrics.scalar_score > result.baseline_candidate.metrics.scalar_score:
            evolved_wins += 1
        elif result.best_candidate.metrics.scalar_score < result.baseline_candidate.metrics.scalar_score:
            baseline_wins += 1
        else:
            ties += 1

    task_count = len(tasks)
    return BenchmarkComparisonSummary(
        task_count=task_count,
        evolved_wins=evolved_wins,
        baseline_wins=baseline_wins,
        ties=ties,
        average_evolved_score=round(sum(evolved_scores) / max(1, task_count), 4),
        average_baseline_score=round(sum(baseline_scores) / max(1, task_count), 4),
    )


def export_predictions_jsonl(predictions: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record) + "\n")
    return output_path


def swebench_eval_command(
    predictions_path: Path,
    run_id: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    max_workers: int = 8,
) -> str:
    return (
        "python -m swebench.harness.run_evaluation "
        f"--dataset_name {dataset_name} "
        f"--predictions_path {predictions_path} "
        f"--max_workers {max_workers} "
        f"--run_id {run_id}"
    )


def export_swebench_predictions(
    *,
    output_path: str | Path,
    predictions: list[dict[str, str]],
) -> Path:
    return export_predictions_jsonl(predictions, output_path)


def run_live_benchmark(
    *,
    tasks: list[BenchmarkTask],
    budget: BudgetConstraints,
    config: SearchConfig,
    hardware: HardwareProfile,
    model_name: str,
    repo_root: str | None,
    output_dir: str | Path,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    swebench_repo_dir: str | Path | None = None,
    run_harness: bool = True,
    harness_max_workers: int = 4,
) -> LiveBenchmarkSummary:
    search = EvolutionarySearch(factory=GraphFactory(seed=config.random_seed))
    executor = LiveGraphExecutor(llm=OpenAIResponsesClient(model_name=model_name))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evolved_predictions: list[dict[str, str]] = []
    baseline_predictions: list[dict[str, str]] = []
    case_results: list[LiveBenchmarkCaseResult] = []
    evolved_nonempty = 0
    baseline_nonempty = 0
    evolved_input_tokens = 0
    baseline_input_tokens = 0
    evolved_output_tokens = 0
    baseline_output_tokens = 0
    case_details: list[dict] = []
    instance_ids = [task.instance_id for task in tasks]
    case_lookup: dict[str, LiveBenchmarkCaseResult] = {}

    for task in tasks:
        repo_path = resolve_repo_path(task.repo_path, repo_root, task.repo)
        search_result = search.run(
            task_description=task.problem_statement,
            budget=budget,
            config=config,
            hardware=hardware,
        )
        evolved = executor.execute_graph(
            graph=search_result.best_candidate.graph,
            instance_id=task.instance_id,
            problem_statement=task.problem_statement,
            repo_path=repo_path,
            budget=budget,
        )
        baseline = executor.execute_single_agent(
            instance_id=task.instance_id,
            problem_statement=task.problem_statement,
            repo_path=repo_path,
            budget=budget,
        )

        evolved_patch = evolved.final_patch or ""
        baseline_patch = baseline.final_patch or ""
        evolved_nonempty += int(bool(evolved_patch.strip()))
        baseline_nonempty += int(bool(baseline_patch.strip()))
        evolved_input_tokens += evolved.total_input_tokens
        baseline_input_tokens += baseline.total_input_tokens
        evolved_output_tokens += evolved.total_output_tokens
        baseline_output_tokens += baseline.total_output_tokens

        evolved_predictions.append(
            {
                "instance_id": task.instance_id,
                "model_name_or_path": f"{model_name}-evolved",
                "model_patch": evolved_patch,
            }
        )
        baseline_predictions.append(
            {
                "instance_id": task.instance_id,
                "model_name_or_path": f"{model_name}-single-agent",
                "model_patch": baseline_patch,
            }
        )
        case_results.append(
            LiveBenchmarkCaseResult(
                instance_id=task.instance_id,
                repo=task.repo,
                evolved_graph_id=search_result.best_candidate.graph.graph_id,
                evolved_patch_nonempty=bool(evolved_patch.strip()),
                baseline_patch_nonempty=bool(baseline_patch.strip()),
                evolved_input_tokens=evolved.total_input_tokens,
                baseline_input_tokens=baseline.total_input_tokens,
                evolved_output_tokens=evolved.total_output_tokens,
                baseline_output_tokens=baseline.total_output_tokens,
                evolved_patch_path=str(output_dir / f"{task.instance_id}.evolved.patch"),
                baseline_patch_path=str(output_dir / f"{task.instance_id}.baseline.patch"),
                evolved_patch_similarity=_patch_similarity(evolved_patch, task.reference_patch),
                baseline_patch_similarity=_patch_similarity(baseline_patch, task.reference_patch),
            )
        )
        case_lookup[task.instance_id] = case_results[-1]
        search_run_path = output_dir / f"{task.instance_id}.search_run.json"
        evolved_execution_path = output_dir / f"{task.instance_id}.evolved_execution.json"
        baseline_execution_path = output_dir / f"{task.instance_id}.baseline_execution.json"
        (output_dir / f"{task.instance_id}.evolved.patch").write_text(evolved_patch, encoding="utf-8")
        (output_dir / f"{task.instance_id}.baseline.patch").write_text(baseline_patch, encoding="utf-8")
        search_run_path.write_text(json.dumps(asdict(search_result), indent=2), encoding="utf-8")
        evolved_execution_path.write_text(json.dumps(asdict(evolved), indent=2), encoding="utf-8")
        baseline_execution_path.write_text(json.dumps(asdict(baseline), indent=2), encoding="utf-8")
        case_details.append(
            {
                "instance_id": task.instance_id,
                "repo": task.repo,
                "repo_path": repo_path,
                "reference_patch_preview": (task.reference_patch or "")[:1000],
                "artifact_paths": {
                    "search_run": str(search_run_path),
                    "evolved_execution": str(evolved_execution_path),
                    "baseline_execution": str(baseline_execution_path),
                    "evolved_patch": str(output_dir / f"{task.instance_id}.evolved.patch"),
                    "baseline_patch": str(output_dir / f"{task.instance_id}.baseline.patch"),
                },
                "search_run": asdict(search_result),
                "evolved_execution": asdict(evolved),
                "baseline_execution": asdict(baseline),
            }
        )

    evolved_predictions_path = export_swebench_predictions(
        output_path=output_dir / "evolved_predictions.jsonl",
        predictions=evolved_predictions,
    )
    baseline_predictions_path = export_swebench_predictions(
        output_path=output_dir / "baseline_predictions.jsonl",
        predictions=baseline_predictions,
    )
    task_count = len(tasks)
    evolved_accuracy: float | None = None
    baseline_accuracy: float | None = None
    evolved_resolved: int | None = None
    baseline_resolved: int | None = None
    evolved_harness_results_path: str | None = None
    baseline_harness_results_path: str | None = None
    harness_error: str | None = None
    harness_supported = True
    evaluation_mode = "swebench_harness"
    harness_unavailable_reason: str | None = None

    evolved_similarity_values = [
        case.evolved_patch_similarity for case in case_results if case.evolved_patch_similarity is not None
    ]
    baseline_similarity_values = [
        case.baseline_patch_similarity for case in case_results if case.baseline_patch_similarity is not None
    ]
    evolved_avg_patch_similarity = (
        round(sum(evolved_similarity_values) / len(evolved_similarity_values), 4)
        if evolved_similarity_values
        else None
    )
    baseline_avg_patch_similarity = (
        round(sum(baseline_similarity_values) / len(baseline_similarity_values), 4)
        if baseline_similarity_values
        else None
    )
    evolved_reference_exact_match_rate = (
        round(sum(1 for value in evolved_similarity_values if value >= 1.0) / len(evolved_similarity_values), 4)
        if evolved_similarity_values
        else None
    )
    baseline_reference_exact_match_rate = (
        round(sum(1 for value in baseline_similarity_values if value >= 1.0) / len(baseline_similarity_values), 4)
        if baseline_similarity_values
        else None
    )

    if run_harness and swebench_repo_dir:
        try:
            run_suffix = uuid4().hex[:8]
            evolved_eval = run_harness_evaluation(
                dataset_name=dataset_name,
                predictions_path=evolved_predictions_path,
                run_id=f"evolved-graph-run-{run_suffix}",
                max_workers=max(1, harness_max_workers),
                instance_ids=instance_ids,
                swebench_repo=swebench_repo_dir,
                output_root=output_dir,
            )
            baseline_eval = run_harness_evaluation(
                dataset_name=dataset_name,
                predictions_path=baseline_predictions_path,
                run_id=f"single-agent-baseline-run-{run_suffix}",
                max_workers=max(1, harness_max_workers),
                instance_ids=instance_ids,
                swebench_repo=swebench_repo_dir,
                output_root=output_dir,
            )
            evolved_accuracy = evolved_eval.accuracy
            baseline_accuracy = baseline_eval.accuracy
            evolved_resolved = evolved_eval.resolved_count
            baseline_resolved = baseline_eval.resolved_count
            evolved_harness_results_path = evolved_eval.results_path
            baseline_harness_results_path = baseline_eval.results_path

            evolved_resolved_ids = set(evolved_eval.resolved_ids)
            baseline_resolved_ids = set(baseline_eval.resolved_ids)
            for instance_id, case in case_lookup.items():
                case.evolved_resolved = instance_id in evolved_resolved_ids
                case.baseline_resolved = instance_id in baseline_resolved_ids
        except RuntimeError as exc:
            harness_error = str(exc)
            if "not supported on native windows" in harness_error.lower():
                harness_supported = False
                evaluation_mode = "proxy_reference_patch"
                harness_unavailable_reason = (
                    "Official SWE-bench harness is unavailable on native Windows Python. "
                    "Use WSL2/Linux or Docker for official resolved accuracy."
                )

    return LiveBenchmarkSummary(
        task_count=task_count,
        model_name=model_name,
        evolved_predictions_path=str(evolved_predictions_path),
        baseline_predictions_path=str(baseline_predictions_path),
        evolved_nonempty_patches=evolved_nonempty,
        baseline_nonempty_patches=baseline_nonempty,
        evolved_avg_input_tokens=round(evolved_input_tokens / max(1, task_count), 2),
        baseline_avg_input_tokens=round(baseline_input_tokens / max(1, task_count), 2),
        evolved_avg_output_tokens=round(evolved_output_tokens / max(1, task_count), 2),
        baseline_avg_output_tokens=round(baseline_output_tokens / max(1, task_count), 2),
        harness_command_evolved=swebench_eval_command(
            predictions_path=evolved_predictions_path,
            run_id="evolved-graph-run",
            dataset_name=dataset_name,
            max_workers=max(1, harness_max_workers),
        ),
        harness_command_baseline=swebench_eval_command(
            predictions_path=baseline_predictions_path,
            run_id="single-agent-baseline-run",
            dataset_name=dataset_name,
            max_workers=max(1, harness_max_workers),
        ),
        evolved_accuracy=evolved_accuracy,
        baseline_accuracy=baseline_accuracy,
        evolved_resolved=evolved_resolved,
        baseline_resolved=baseline_resolved,
        evolved_harness_results_path=evolved_harness_results_path,
        baseline_harness_results_path=baseline_harness_results_path,
        harness_error=harness_error,
        harness_supported=harness_supported,
        evaluation_mode=evaluation_mode,
        harness_unavailable_reason=harness_unavailable_reason,
        evolved_reference_exact_match_rate=evolved_reference_exact_match_rate,
        baseline_reference_exact_match_rate=baseline_reference_exact_match_rate,
        evolved_avg_patch_similarity=evolved_avg_patch_similarity,
        baseline_avg_patch_similarity=baseline_avg_patch_similarity,
        case_results=case_results,
        case_details=case_details,
    )


def run_live_benchmark_auto(
    *,
    workspace_root: str | Path,
    question_count: int,
    budget: BudgetConstraints,
    config: SearchConfig,
    hardware: HardwareProfile,
    model_name: str,
    output_dir: str | Path,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    dataset_split: str = "test",
    harness_max_workers: int = 4,
) -> LiveBenchmarkSummary:
    workspace_root = Path(workspace_root)
    swebench_repo_dir = ensure_swebench_harness(workspace_root=workspace_root, install_if_missing=True)
    cache_override = os.getenv("SWEBENCH_DATASET_CACHE")
    tasks_cache_dir = Path(cache_override) if cache_override else (workspace_root / ".hf")
    repos_root = workspace_root / "external" / "swebench_repos"
    loaded = load_swebench_tasks(
        dataset_name=dataset_name,
        split=dataset_split,
        limit=question_count,
        cache_dir=tasks_cache_dir,
        seed=config.random_seed,
    )
    tasks = [BenchmarkTask(**record) for record in loaded]
    prepare_task_repositories(tasks=tasks, repos_root=repos_root)
    return run_live_benchmark(
        tasks=tasks,
        budget=budget,
        config=config,
        hardware=hardware,
        model_name=model_name,
        repo_root=str(repos_root),
        output_dir=output_dir,
        dataset_name=dataset_name,
        swebench_repo_dir=swebench_repo_dir,
        run_harness=True,
        harness_max_workers=harness_max_workers,
    )


def _patch_similarity(candidate: str, reference: str | None) -> float | None:
    if not reference:
        return None
    if not candidate:
        return 0.0
    return round(SequenceMatcher(a=candidate, b=reference).ratio(), 4)
