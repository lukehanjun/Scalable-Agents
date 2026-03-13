from __future__ import annotations

import re
import time
from collections import defaultdict

from .llm import LLMResponse, OpenAIResponsesClient
from .models import AgentGraph, BudgetConstraints, ExecutionTraceStep, LiveExecutionResult
from .repo_tools import LocalRepoContextBuilder


class LiveGraphExecutor:
    def __init__(self, llm: OpenAIResponsesClient | None = None) -> None:
        self.llm = llm or OpenAIResponsesClient()

    def execute_graph(
        self,
        *,
        graph: AgentGraph,
        instance_id: str,
        problem_statement: str,
        repo_path: str | None,
        budget: BudgetConstraints,
    ) -> LiveExecutionResult:
        started = time.perf_counter()
        repo_builder = LocalRepoContextBuilder(repo_path)
        cached_snippets = repo_builder.build_context(problem_statement) if graph.has_operator("retriever") else []
        repo_context = LocalRepoContextBuilder.summarize(cached_snippets)

        inbound_map = defaultdict(list)
        for edge in graph.edges:
            inbound_map[edge.target].append(edge.source)

        outputs: dict[str, str] = {}
        trace: list[ExecutionTraceStep] = []
        total_input_tokens = 0
        total_output_tokens = 0

        ordered_nodes = sorted(graph.nodes, key=lambda node: (node.layer, node.node_id))
        for node in ordered_nodes:
            parent_outputs = [outputs[parent_id] for parent_id in inbound_map[node.node_id] if parent_id in outputs]
            response = self._run_node(
                operator=node.operator,
                problem_statement=problem_statement,
                repo_context=repo_context,
                parent_outputs=parent_outputs,
                budget=budget,
            )
            outputs[node.node_id] = response.text
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            trace.append(
                ExecutionTraceStep(
                    node_id=node.node_id,
                    operator=node.operator,
                    action_type="llm" if node.operator != "retriever" else "retrieval",
                    prompt_preview=self._prompt_preview(node.operator, problem_statement, parent_outputs),
                    output_preview=response.text[:500],
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    metadata={"repo_path": repo_path or "", "parent_count": len(parent_outputs)},
                )
            )

        final_response = outputs.get("synthesizer") or outputs.get(ordered_nodes[-1].node_id, "")
        candidate_patch = extract_patch(final_response)
        if not candidate_patch:
            for node in reversed(ordered_nodes):
                candidate_patch = extract_patch(outputs.get(node.node_id, ""))
                if candidate_patch:
                    break
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return LiveExecutionResult(
            instance_id=instance_id,
            graph_id=graph.graph_id,
            model_name=self.llm.model_name,
            repo_path=repo_path,
            final_patch=candidate_patch,
            final_response=final_response,
            trace=trace,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            latency_ms=elapsed_ms,
            metadata={
                "repo_context_files": len(cached_snippets),
                "template": graph.template,
                "worker_count": graph.worker_count(),
            },
        )

    def execute_single_agent(
        self,
        *,
        instance_id: str,
        problem_statement: str,
        repo_path: str | None,
        budget: BudgetConstraints,
    ) -> LiveExecutionResult:
        started = time.perf_counter()
        repo_builder = LocalRepoContextBuilder(repo_path)
        snippets = repo_builder.build_context(problem_statement)
        repo_context = LocalRepoContextBuilder.summarize(snippets)
        system_prompt = (
            "You are a software repair agent working on a SWE-bench style issue. "
            "Return only a valid unified diff patch that addresses the issue. "
            "Do not include markdown fences or explanation."
        )
        user_prompt = (
            f"Issue:\n{problem_statement}\n\n"
            f"Repository context:\n{repo_context}\n\n"
            "Produce the smallest plausible patch that fixes the issue."
        )
        response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=min(1_600, budget.max_tokens),
        )
        final_patch = extract_patch(response.text)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return LiveExecutionResult(
            instance_id=instance_id,
            graph_id="single-agent-baseline",
            model_name=self.llm.model_name,
            repo_path=repo_path,
            final_patch=final_patch,
            final_response=response.text,
            trace=[
                ExecutionTraceStep(
                    node_id="baseline",
                    operator="single-agent",
                    action_type="llm",
                    prompt_preview=user_prompt[:300],
                    output_preview=response.text[:500],
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    metadata={"repo_path": repo_path or ""},
                )
            ],
            total_input_tokens=response.input_tokens,
            total_output_tokens=response.output_tokens,
            latency_ms=elapsed_ms,
            metadata={"repo_context_files": len(snippets)},
        )

    def _run_node(
        self,
        *,
        operator: str,
        problem_statement: str,
        repo_context: str,
        parent_outputs: list[str],
        budget: BudgetConstraints,
    ) -> LLMResponse:
        if operator == "retriever":
            return LLMResponse(text=repo_context, input_tokens=0, output_tokens=0, raw=None)

        system_prompt = self._system_prompt(operator)
        user_prompt = self._user_prompt(operator, problem_statement, repo_context, parent_outputs)
        max_output_tokens = min(1_200, max(250, budget.max_tokens // 2))
        return self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
        )

    @staticmethod
    def _system_prompt(operator: str) -> str:
        prompts = {
            "planner": "You are the planning node in a software repair graph. Produce a concise repair strategy.",
            "decomposer": "You break software repair into concrete subtasks and likely failure modes.",
            "worker": "You are an implementation worker. Draft a candidate fix patch or concrete code change plan.",
            "critic": "You critique a proposed fix. Focus on correctness gaps, missing files, and hidden regressions.",
            "verifier": "You verify whether a proposed fix is likely to satisfy the issue and avoid regressions.",
            "judge": "You compare candidate fixes and choose the strongest one, with explicit reasoning.",
            "synthesizer": (
                "You are the final patch synthesizer. Return only a unified diff patch. "
                "Do not use markdown fences or explanatory prose."
            ),
            "tool-user": "You focus on repository-aware repair tactics and implementation details.",
        }
        return prompts.get(operator, "You are part of a software repair workflow.")

    @staticmethod
    def _user_prompt(
        operator: str,
        problem_statement: str,
        repo_context: str,
        parent_outputs: list[str],
    ) -> str:
        parent_text = "\n\n".join(parent_outputs[-4:]) if parent_outputs else "No parent context."
        if operator == "planner":
            return f"Issue:\n{problem_statement}\n\nCreate a focused plan for fixing this issue."
        if operator == "decomposer":
            return f"Issue:\n{problem_statement}\n\nPlanner output:\n{parent_text}\n\nBreak this into subtasks."
        if operator == "worker":
            return (
                f"Issue:\n{problem_statement}\n\nRelevant repository context:\n{repo_context}\n\n"
                f"Upstream context:\n{parent_text}\n\n"
                "Draft a candidate fix. Prefer including a unified diff patch."
            )
        if operator in {"critic", "verifier", "judge"}:
            return (
                f"Issue:\n{problem_statement}\n\nRelevant repository context:\n{repo_context}\n\n"
                f"Candidate materials:\n{parent_text}\n\n"
                "Evaluate the candidate rigorously."
            )
        if operator == "synthesizer":
            return (
                f"Issue:\n{problem_statement}\n\nRelevant repository context:\n{repo_context}\n\n"
                f"Prior node outputs:\n{parent_text}\n\n"
                "Return the final best unified diff patch only."
            )
        if operator == "tool-user":
            return (
                f"Issue:\n{problem_statement}\n\nRelevant repository context:\n{repo_context}\n\n"
                "List the exact files and edits that should be changed."
            )
        return f"Issue:\n{problem_statement}\n\nContext:\n{parent_text}"

    @staticmethod
    def _prompt_preview(operator: str, problem_statement: str, parent_outputs: list[str]) -> str:
        return f"{operator}: {problem_statement[:140]} | parents={len(parent_outputs)}"


def extract_patch(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    fenced = re.search(r"```(?:diff|patch)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()

    if "diff --git " in stripped:
        return stripped[stripped.index("diff --git "):].strip()
    if stripped.startswith("--- ") and "\n+++ " in stripped:
        return stripped
    if stripped.startswith("Index: ") or stripped.startswith("*** Begin Patch"):
        return stripped
    return ""
