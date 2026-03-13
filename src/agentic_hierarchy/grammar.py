from __future__ import annotations

import random
from typing import Any
from uuid import uuid4

from .models import AgentGraph, GraphEdge, GraphNode, TaskProfile
from .operators import OPERATOR_LIBRARY


class GraphFactory:
    TEMPLATE_CHOICES = ("tree", "star", "chain", "layered_dag")
    MODEL_MIXES = ("cheap", "balanced", "quality")

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def seed_population(self, profile: TaskProfile, count: int) -> list[AgentGraph]:
        candidates: list[AgentGraph] = [self.single_agent_baseline()]
        options = []
        for template in self.TEMPLATE_CHOICES:
            for worker_count in (1, 2, 3, 4):
                for model_mix in self.MODEL_MIXES:
                    options.append(
                        {
                            "template": template,
                            "worker_count": worker_count,
                            "include_retriever": profile.need_retrieval or worker_count >= 3,
                            "include_critic": worker_count >= 2,
                            "include_verifier": profile.need_verification,
                            "include_judge": profile.complexity >= 4,
                            "include_decomposer": profile.complexity >= 3,
                            "include_tool_user": profile.domain == "software_engineering" and worker_count >= 2,
                            "summarized_ratio": 0.25 if worker_count <= 2 else 0.45,
                            "model_mix": model_mix,
                            "judge_feedback": profile.complexity >= 4,
                        }
                    )
        while len(candidates) < count:
            spec = options[(len(candidates) - 1) % len(options)]
            candidates.append(self.build_graph(**spec))
        return candidates[:count]

    def single_agent_baseline(self) -> AgentGraph:
        return self.build_graph(
            template="chain",
            worker_count=1,
            include_retriever=False,
            include_critic=False,
            include_verifier=False,
            include_judge=False,
            include_decomposer=False,
            include_tool_user=False,
            summarized_ratio=0.0,
            model_mix="balanced",
            judge_feedback=False,
        )

    def build_graph(
        self,
        template: str,
        worker_count: int,
        include_retriever: bool,
        include_critic: bool,
        include_verifier: bool,
        include_judge: bool,
        include_decomposer: bool,
        include_tool_user: bool,
        summarized_ratio: float,
        model_mix: str,
        judge_feedback: bool,
    ) -> AgentGraph:
        worker_count = max(1, min(worker_count, 6))
        summarized_ratio = max(0.0, min(summarized_ratio, 0.95))

        nodes: list[GraphNode] = [self._node("planner", 0, model_mix)]
        if include_decomposer:
            nodes.append(self._node("decomposer", 1, model_mix))
        if include_retriever:
            nodes.append(self._node("retriever", 1, model_mix))
        if include_tool_user:
            nodes.append(self._node("tool-user", 1, model_mix))

        worker_layer = 2 if any(node.layer == 1 for node in nodes) else 1
        for index in range(worker_count):
            nodes.append(self._node("worker", worker_layer, model_mix, suffix=index))

        review_layer = worker_layer + 1
        if include_critic:
            nodes.append(self._node("critic", review_layer, model_mix))
        if include_verifier:
            nodes.append(self._node("verifier", review_layer, model_mix))
        if include_judge:
            nodes.append(self._node("judge", review_layer + 1, model_mix))

        synth_layer = review_layer + (2 if include_judge else 1)
        nodes.append(self._node("synthesizer", synth_layer, model_mix))
        edges = self._build_edges(template, nodes, summarized_ratio, judge_feedback)

        metadata: dict[str, Any] = {
            "template": template,
            "worker_count": worker_count,
            "include_retriever": include_retriever,
            "include_critic": include_critic,
            "include_verifier": include_verifier,
            "include_judge": include_judge,
            "include_decomposer": include_decomposer,
            "include_tool_user": include_tool_user,
            "summarized_ratio": summarized_ratio,
            "model_mix": model_mix,
            "judge_feedback": judge_feedback,
        }
        return AgentGraph(
            graph_id=f"graph-{uuid4().hex[:8]}",
            template=template,
            nodes=nodes,
            edges=edges,
            metadata=metadata,
        )

    def mutate(self, graph: AgentGraph) -> tuple[AgentGraph, str, str]:
        spec = dict(graph.metadata)
        operation = self.rng.choice(
            [
                "add_worker",
                "remove_worker",
                "toggle_retriever",
                "toggle_critic",
                "toggle_verifier",
                "toggle_judge",
                "toggle_decomposer",
                "toggle_tool_user",
                "switch_template",
                "raise_summary",
                "lower_summary",
                "swap_model_mix",
                "toggle_judge_feedback",
            ]
        )
        description = operation.replace("_", " ")

        if operation == "add_worker":
            spec["worker_count"] = min(spec["worker_count"] + 1, 6)
        elif operation == "remove_worker":
            spec["worker_count"] = max(spec["worker_count"] - 1, 1)
        elif operation == "toggle_retriever":
            spec["include_retriever"] = not spec["include_retriever"]
        elif operation == "toggle_critic":
            spec["include_critic"] = not spec["include_critic"]
        elif operation == "toggle_verifier":
            spec["include_verifier"] = not spec["include_verifier"]
        elif operation == "toggle_judge":
            spec["include_judge"] = not spec["include_judge"]
        elif operation == "toggle_decomposer":
            spec["include_decomposer"] = not spec["include_decomposer"]
        elif operation == "toggle_tool_user":
            spec["include_tool_user"] = not spec["include_tool_user"]
        elif operation == "switch_template":
            alternatives = [choice for choice in self.TEMPLATE_CHOICES if choice != spec["template"]]
            spec["template"] = self.rng.choice(alternatives)
        elif operation == "raise_summary":
            spec["summarized_ratio"] = min(spec["summarized_ratio"] + 0.18, 0.95)
        elif operation == "lower_summary":
            spec["summarized_ratio"] = max(spec["summarized_ratio"] - 0.18, 0.0)
        elif operation == "swap_model_mix":
            alternatives = [choice for choice in self.MODEL_MIXES if choice != spec["model_mix"]]
            spec["model_mix"] = self.rng.choice(alternatives)
        elif operation == "toggle_judge_feedback":
            spec["judge_feedback"] = not spec["judge_feedback"]

        child = self.build_graph(**spec)
        return child, operation, description

    def crossover(self, left: AgentGraph, right: AgentGraph) -> tuple[AgentGraph, str]:
        spec = {}
        for key in left.metadata:
            if key == "worker_count":
                spec[key] = max(1, min(6, round((left.metadata[key] + right.metadata[key]) / 2)))
            elif key == "summarized_ratio":
                spec[key] = round((left.metadata[key] + right.metadata[key]) / 2, 2)
            else:
                spec[key] = self.rng.choice([left.metadata[key], right.metadata[key]])
        child = self.build_graph(**spec)
        return child, "hybridized topology and review settings"

    def _node(self, operator: str, layer: int, model_mix: str, suffix: int | None = None) -> GraphNode:
        suffix_text = f"-{suffix}" if suffix is not None else ""
        spec = OPERATOR_LIBRARY[operator]
        return GraphNode(
            node_id=f"{operator}{suffix_text}",
            operator=operator,
            role=spec.role,
            layer=layer,
            model_tier=model_mix,
            tools=list(spec.default_tools),
            memory_scope="shared" if operator in {"planner", "judge", "synthesizer"} else "local",
            prompt_style="analytical" if operator in {"critic", "verifier", "judge"} else "concise",
        )

    def _build_edges(
        self,
        template: str,
        nodes: list[GraphNode],
        summarized_ratio: float,
        judge_feedback: bool,
    ) -> list[GraphEdge]:
        layer_groups: dict[int, list[GraphNode]] = {}
        for node in nodes:
            layer_groups.setdefault(node.layer, []).append(node)

        def channel_for(index: int) -> str:
            return "summary" if (index % 10) / 10 < summarized_ratio else "full"

        edges: list[GraphEdge] = []
        edge_index = 0
        ordered_layers = sorted(layer_groups)
        planner = next(node for node in nodes if node.operator == "planner")
        synthesizer = next(node for node in nodes if node.operator == "synthesizer")
        workers = [node for node in nodes if node.operator == "worker"]

        if template == "chain":
            chain_nodes = []
            for layer in ordered_layers:
                chain_nodes.extend(sorted(layer_groups[layer], key=lambda node: node.node_id))
            for source, target in zip(chain_nodes, chain_nodes[1:]):
                edges.append(GraphEdge(source.node_id, target.node_id, channel_for(edge_index)))
                edge_index += 1
            return edges

        if template == "star":
            non_final = [node for node in nodes if node.operator != "synthesizer" and node.node_id != planner.node_id]
            for node in non_final:
                edges.append(GraphEdge(planner.node_id, node.node_id, channel_for(edge_index)))
                edge_index += 1
            for worker in workers:
                edges.append(GraphEdge(worker.node_id, synthesizer.node_id, channel_for(edge_index)))
                edge_index += 1
            for node in nodes:
                if node.operator in {"critic", "verifier", "judge"}:
                    edges.append(GraphEdge(node.node_id, synthesizer.node_id, channel_for(edge_index)))
                    edge_index += 1
            return self._dedupe_edges(edges)

        for source_layer, target_layer in zip(ordered_layers, ordered_layers[1:]):
            sources = sorted(layer_groups[source_layer], key=lambda node: node.node_id)
            targets = sorted(layer_groups[target_layer], key=lambda node: node.node_id)
            if template == "tree":
                if len(sources) == 1:
                    for target in targets:
                        edges.append(GraphEdge(sources[0].node_id, target.node_id, channel_for(edge_index)))
                        edge_index += 1
                else:
                    for index, target in enumerate(targets):
                        source = sources[index % len(sources)]
                        edges.append(GraphEdge(source.node_id, target.node_id, channel_for(edge_index)))
                        edge_index += 1
            else:
                for source in sources:
                    for target in targets:
                        edges.append(GraphEdge(source.node_id, target.node_id, channel_for(edge_index)))
                        edge_index += 1

        if template == "layered_dag" and workers:
            reviewers = [node for node in nodes if node.operator in {"critic", "verifier", "judge"}]
            for review in reviewers:
                for worker in workers:
                    edges.append(GraphEdge(worker.node_id, review.node_id, channel_for(edge_index)))
                    edge_index += 1

        if judge_feedback:
            judges = [node for node in nodes if node.operator == "judge"]
            for judge in judges:
                edges.append(GraphEdge(judge.node_id, synthesizer.node_id, channel_for(edge_index)))
                edge_index += 1
                edges.append(GraphEdge(planner.node_id, synthesizer.node_id, "summary"))

        return self._dedupe_edges(edges)

    @staticmethod
    def _dedupe_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[GraphEdge] = []
        for edge in edges:
            key = (edge.source, edge.target, edge.channel)
            if key not in seen and edge.source != edge.target:
                seen.add(key)
                deduped.append(edge)
        return deduped
