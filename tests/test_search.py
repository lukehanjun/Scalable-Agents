from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agentic_hierarchy.benchmarks import BenchmarkTask, export_swebench_predictions, run_surrogate_benchmark
from agentic_hierarchy.executor import extract_patch
from agentic_hierarchy.evaluator import GraphEvaluator
from agentic_hierarchy.grammar import GraphFactory
from agentic_hierarchy.models import BudgetConstraints, HardwareProfile, SearchConfig
from agentic_hierarchy.parallel import ParallelEvaluationRuntime
from agentic_hierarchy.repo_tools import resolve_repo_path
from agentic_hierarchy.search import EvolutionarySearch


class SearchSmokeTests(unittest.TestCase):
    def test_search_returns_condensed_trace(self) -> None:
        search = EvolutionarySearch(
            factory=GraphFactory(seed=3),
            evaluator=GraphEvaluator(),
            runtime=ParallelEvaluationRuntime(),
        )
        result = search.run(
            task_description="Design an agent workflow for a SWE-bench style repo issue under 2k tokens.",
            budget=BudgetConstraints(max_tokens=2_000, max_latency_ms=10_000),
            config=SearchConfig(population_size=10, generations=4, trials=2, random_seed=3),
            hardware=HardwareProfile(gpu_count=2, cpu_cores=8),
        )
        self.assertTrue(result.best_candidate.metrics.scalar_score >= result.baseline_candidate.metrics.scalar_score)
        self.assertGreater(len(result.mutation_events), 0)
        self.assertLessEqual(len(result.condensed_events), 24)

    def test_surrogate_benchmark_summary(self) -> None:
        tasks = [
            BenchmarkTask(instance_id="demo-1", problem_statement="Fix a failing Python test in a repository."),
            BenchmarkTask(instance_id="demo-2", problem_statement="Resolve a flaky benchmark regression with missing docs context."),
        ]
        summary = run_surrogate_benchmark(
            tasks=tasks,
            budget=BudgetConstraints(max_tokens=2_200, max_latency_ms=10_000),
            config=SearchConfig(population_size=8, generations=3, trials=2, random_seed=5),
            hardware=HardwareProfile(gpu_count=1, cpu_cores=4),
        )
        self.assertEqual(summary.task_count, 2)
        self.assertGreaterEqual(summary.average_evolved_score, summary.average_baseline_score)

    def test_patch_extraction_and_prediction_export(self) -> None:
        patch = extract_patch("```diff\ndiff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-print('a')\n+print('b')\n```")
        self.assertIn("diff --git a/app.py b/app.py", patch)
        with TemporaryDirectory() as temp_dir:
            output = export_swebench_predictions(
                output_path=Path(temp_dir) / "predictions.jsonl",
                predictions=[{"instance_id": "demo", "model_name_or_path": "test-model", "model_patch": patch}],
            )
            self.assertTrue(output.exists())
            self.assertIn('"instance_id": "demo"', output.read_text(encoding="utf-8"))

    def test_resolve_repo_path_prefers_explicit_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            explicit = Path(temp_dir) / "repo"
            explicit.mkdir()
            resolved = resolve_repo_path(str(explicit), None, "org/repo")
            self.assertEqual(Path(resolved), explicit.resolve())


if __name__ == "__main__":
    unittest.main()
