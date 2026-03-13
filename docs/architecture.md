# Architecture

## Core idea

Each candidate workflow is represented as a searchable program graph:

- Nodes represent agentic operators such as `planner`, `retriever`, `worker`, `critic`, and `judge`.
- Edges represent communication pathways and message compression choices.
- Graph metadata encodes template family, layer structure, worker count, model mix, and optional review stages.

Rather than exploring arbitrary graphs from the start, the system searches over a bounded grammar. That keeps the search tractable, makes mutation semantics interpretable, and ensures generated workflows are deployable under a strict budget.

## Search loop

The search loop is budget-aware and multi-objective:

- Objective dimensions: quality, latency, token cost, variance, robustness, and estimated dollar cost.
- Search method: evolutionary search with mutation, bounded crossover, elitism, tournament selection, and Pareto archival.
- Trace outputs: generation summaries, parent-child mutation records, graph snapshots, and a condensed filmstrip.

The condensed filmstrip samples 20-30 key mutation events from long runs so the UI can present a fast animation without forcing the user through every intermediate graph.

## Parallelization strategy

The runtime uses a hybrid plan:

- Across GPUs: island model. Each GPU owns an island with a population shard.
- Within each island: candidate evaluation is data parallel over tasks or trials.
- Communication: low-frequency elite migration, not full all-reduce. This lowers synchronization overhead and keeps islands diverse.
- CPU-only fallback: local thread pool with the same APIs.

This is a good fit for agent search because evaluation is usually dominated by model inference or tool execution rather than dense tensor collectives. In that regime, island search plus asynchronous migration avoids turning every generation into a synchronization barrier.

## Evaluation model

The included search evaluator is a surrogate that estimates:

- task-quality fit
- budget feasibility
- latency from layer depth and coordination overhead
- robustness from verification and judging motifs
- variance from lack of review or brittle topologies

For real experiments:

1. Replace the surrogate executor with an implementation that runs the graph on a task.
2. Keep the same budget accounting and trace schema.
3. Export predictions to JSONL and run the official SWE-bench harness.

## Live execution

The repository now also includes a live execution path:

- `.env`-driven OpenAI Responses API client
- deterministic local repo retrieval over a checked-out benchmark repository
- node-by-node graph execution that turns a searched topology into real LLM calls
- single-agent baseline generation for apples-to-apples prediction export

This keeps the fast inner loop practical while still enabling end-to-end patch generation for actual SWE-bench instances.

## Benchmarking

The benchmark module supports two stages:

- Fast iteration: load issue text from SWE-bench style JSONL and compare evolved graphs versus a single-agent baseline under the surrogate evaluator.
- Official evaluation: emit JSONL predictions and execute the official SWE-bench harness separately.

This split makes the framework usable in practice. You can do fast inner-loop search locally, then spend real benchmark compute only on the most promising graph families.
