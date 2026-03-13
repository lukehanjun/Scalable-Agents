const state = {
  mode: "full",
  benchmark: null,
  run: null,
  selectedCaseIndex: 0,
  currentEventIndex: 0,
  playing: false,
  timer: null,
};

const form = document.getElementById("benchmark-form");
const summaryCards = document.getElementById("summaryCards");
const benchmarkSummary = document.getElementById("benchmarkSummary");
const caseList = document.getElementById("caseList");
const evolvedMetrics = document.getElementById("evolvedMetrics");
const baselineMetrics = document.getElementById("baselineMetrics");
const evolvedPatch = document.getElementById("evolvedPatch");
const baselinePatch = document.getElementById("baselinePatch");
const evolvedTrace = document.getElementById("evolvedTrace");
const baselineTrace = document.getElementById("baselineTrace");
const caseTitle = document.getElementById("caseTitle");
const graphSectionTitle = document.getElementById("graphSectionTitle");
const eventList = document.getElementById("eventList");
const eventTitle = document.getElementById("eventTitle");
const eventCount = document.getElementById("eventCount");
const statusPill = document.getElementById("statusPill");
const graphCanvas = document.getElementById("graphCanvas");
const playButton = document.getElementById("playButton");
const toggles = document.querySelectorAll(".toggle");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopPlayback();
  setStatus("Running SWE-bench live", "running");
  clearPanels();

  try {
    const response = await fetch("/api/benchmark/run-live", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildBenchmarkPayload()),
    });
    if (!response.ok) {
      const errorBody = await safeJson(response);
      throw new Error(errorBody?.detail || `Benchmark failed with status ${response.status}`);
    }
    const result = await response.json();
    state.benchmark = result;
    state.selectedCaseIndex = 0;
    hydrateSelectedCase();
    renderBenchmark();
    setStatus("Benchmark ready", "ready");
  } catch (error) {
    console.error(error);
    setStatus("Error", "error");
    benchmarkSummary.innerHTML = `<div class="comparison-row"><span>Error</span><strong>${error.message}</strong></div>`;
  }
});

async function safeJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

playButton.addEventListener("click", () => {
  const events = activeEvents();
  if (!events.length) {
    return;
  }
  if (state.playing) {
    stopPlayback();
    return;
  }
  state.playing = true;
  playButton.textContent = "Pause";
  state.timer = window.setInterval(() => {
    state.currentEventIndex = (state.currentEventIndex + 1) % events.length;
    renderActiveEvent();
    highlightActiveEvent();
  }, 1100);
});

toggles.forEach((toggle) => {
  toggle.addEventListener("click", () => {
    toggles.forEach((button) => button.classList.remove("active"));
    toggle.classList.add("active");
    state.mode = toggle.dataset.mode;
    state.currentEventIndex = 0;
    stopPlayback();
    renderTimeline();
    renderActiveEvent();
  });
});

function buildBenchmarkPayload() {
  return {
    tasks_path: document.getElementById("tasksPath").value,
    repo_root: document.getElementById("repoRoot").value || null,
    model_name: document.getElementById("modelName").value,
    output_dir: "artifacts/live-benchmark",
    dataset_name: "princeton-nlp/SWE-bench_Lite",
    limit: Number(document.getElementById("benchmarkLimit").value),
    budget: {
      max_tokens: Number(document.getElementById("maxTokens").value),
      max_latency_ms: Number(document.getElementById("maxLatency").value),
      max_cost: 4.0,
    },
    search: {
      population_size: Number(document.getElementById("populationSize").value),
      generations: Number(document.getElementById("generations").value),
      trials: Number(document.getElementById("trials").value),
      mutation_rate: 0.72,
      crossover_rate: 0.28,
      elitism: 4,
      tournament_size: 3,
      max_workers_per_island: 8,
      random_seed: 7,
    },
    hardware: {
      gpu_count: Number(document.getElementById("gpuCount").value),
      cpu_cores: 8,
      per_gpu_concurrency: 4,
      interconnect: "nvlink-or-pcie",
    },
  };
}

function clearPanels() {
  summaryCards.innerHTML = "";
  benchmarkSummary.innerHTML = "";
  caseList.innerHTML = "";
  evolvedMetrics.innerHTML = "";
  baselineMetrics.innerHTML = "";
  evolvedPatch.textContent = "Run a benchmark to see the evolved patch.";
  baselinePatch.textContent = "Run a benchmark to see the baseline patch.";
  evolvedTrace.innerHTML = "";
  baselineTrace.innerHTML = "";
  eventList.innerHTML = "";
  eventCount.textContent = "0";
  eventTitle.textContent = "No evolution loaded";
  graphCanvas.innerHTML = "";
  caseTitle.textContent = "No case selected";
  graphSectionTitle.textContent = "Mutation history";
}

function setStatus(text, kind) {
  statusPill.textContent = text;
  statusPill.className = `status ${kind}`;
}

function hydrateSelectedCase() {
  const detail = selectedCaseDetail();
  state.run = detail ? detail.search_run : null;
  state.currentEventIndex = 0;
}

function renderBenchmark() {
  renderSummary();
  renderCaseSelector();
  renderSelectedCase();
  renderTimeline();
  renderActiveEvent();
}

function renderSummary() {
  const summary = state.benchmark;
  summaryCards.innerHTML = `
    <div class="summary-card">
      <span>Task count</span>
      <strong>${summary.task_count}</strong>
    </div>
    <div class="summary-card">
      <span>Evolved non-empty patches</span>
      <strong>${summary.evolved_nonempty_patches}</strong>
    </div>
    <div class="summary-card">
      <span>Baseline non-empty patches</span>
      <strong>${summary.baseline_nonempty_patches}</strong>
    </div>
  `;
  benchmarkSummary.innerHTML = `
    <div class="comparison-row">
      <span>Evolved predictions</span>
      <strong>${summary.evolved_predictions_path}</strong>
    </div>
    <div class="comparison-row">
      <span>Baseline predictions</span>
      <strong>${summary.baseline_predictions_path}</strong>
    </div>
    <div class="comparison-row">
      <span>Harness command</span>
      <strong>${summary.harness_command_evolved}</strong>
    </div>
  `;
}

function renderCaseSelector() {
  const details = state.benchmark.case_details || [];
  caseList.innerHTML = details
    .map(
      (detail, index) => `
        <button class="case-button ${index === state.selectedCaseIndex ? "active" : ""}" data-index="${index}">
          <small>${detail.repo || "repo unavailable"}</small>
          <div>${detail.instance_id}</div>
        </button>
      `
    )
    .join("");

  [...caseList.querySelectorAll(".case-button")].forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedCaseIndex = Number(button.dataset.index);
      stopPlayback();
      hydrateSelectedCase();
      renderCaseSelector();
      renderSelectedCase();
      renderTimeline();
      renderActiveEvent();
    });
  });
}

function renderSelectedCase() {
  const detail = selectedCaseDetail();
  if (!detail) {
    return;
  }
  caseTitle.textContent = detail.instance_id;
  graphSectionTitle.textContent = `Mutation history · ${detail.instance_id}`;

  const evolved = detail.evolved_execution;
  const baseline = detail.baseline_execution;
  evolvedMetrics.innerHTML = metricRows(evolved, detail.artifact_paths.evolved_patch);
  baselineMetrics.innerHTML = metricRows(baseline, detail.artifact_paths.baseline_patch);
  evolvedPatch.textContent = evolved.final_patch || "No unified diff patch was extracted.";
  baselinePatch.textContent = baseline.final_patch || "No unified diff patch was extracted.";
  evolvedTrace.innerHTML = traceMarkup(evolved.trace);
  baselineTrace.innerHTML = traceMarkup(baseline.trace);
}

function metricRows(execution, patchPath) {
  return `
    <div class="comparison-row">
      <span>Patch bytes</span>
      <strong>${(execution.final_patch || "").length}</strong>
    </div>
    <div class="comparison-row">
      <span>Token usage</span>
      <strong>${execution.total_input_tokens} in / ${execution.total_output_tokens} out</strong>
    </div>
    <div class="comparison-row">
      <span>Latency</span>
      <strong>${execution.latency_ms} ms</strong>
    </div>
    <div class="comparison-row">
      <span>Artifact</span>
      <strong>${patchPath}</strong>
    </div>
  `;
}

function traceMarkup(trace) {
  return (trace || [])
    .map(
      (step) => `
        <div class="trace-step">
          <small>${step.operator} · ${step.action_type}</small>
          <div>${step.output_preview || "No output preview available."}</div>
        </div>
      `
    )
    .join("");
}

function selectedCaseDetail() {
  return state.benchmark?.case_details?.[state.selectedCaseIndex] || null;
}

function renderTimeline() {
  const events = activeEvents();
  eventCount.textContent = `${events.length} events`;
  eventList.innerHTML = events
    .map(
      (event, index) => `
        <button class="event ${index === state.currentEventIndex ? "active" : ""}" data-index="${index}">
          <small>Gen ${event.generation} · ${event.operation.replaceAll("_", " ")}</small>
          <div>${event.description}</div>
        </button>
      `
    )
    .join("");

  [...eventList.querySelectorAll(".event")].forEach((button) => {
    button.addEventListener("click", () => {
      state.currentEventIndex = Number(button.dataset.index);
      stopPlayback();
      renderActiveEvent();
      highlightActiveEvent();
    });
  });
}

function activeEvents() {
  if (!state.run) {
    return [];
  }
  return state.mode === "condensed" ? state.run.condensed_events : state.run.mutation_events;
}

function renderActiveEvent() {
  const events = activeEvents();
  if (!events.length) {
    eventTitle.textContent = "No evolution loaded";
    graphCanvas.innerHTML = "";
    return;
  }
  const event = events[state.currentEventIndex];
  eventTitle.textContent = `${event.operation.replaceAll("_", " ")} · ${event.child_graph.template}`;
  drawGraph(event.child_graph);
  highlightActiveEvent();
}

function highlightActiveEvent() {
  [...eventList.querySelectorAll(".event")].forEach((button, index) => {
    button.classList.toggle("active", index === state.currentEventIndex);
  });
}

function stopPlayback() {
  if (state.timer) {
    window.clearInterval(state.timer);
  }
  state.timer = null;
  state.playing = false;
  playButton.textContent = "Play";
}

function drawGraph(graph) {
  graphCanvas.innerHTML = "";
  const width = 900;
  const height = 460;
  const paddingX = 90;
  const paddingY = 70;
  const layers = [...new Set(graph.nodes.map((node) => node.layer))].sort((a, b) => a - b);
  const layerMap = new Map();
  layers.forEach((layer) => {
    layerMap.set(
      layer,
      graph.nodes.filter((node) => node.layer === layer).sort((a, b) => a.node_id.localeCompare(b.node_id))
    );
  });

  const positions = new Map();
  layers.forEach((layer, layerIndex) => {
    const nodes = layerMap.get(layer);
    const x = paddingX + (layerIndex / Math.max(1, layers.length - 1)) * (width - paddingX * 2);
    nodes.forEach((node, index) => {
      const y = paddingY + ((index + 1) / (nodes.length + 1)) * (height - paddingY * 2);
      positions.set(node.node_id, { x, y });
    });
  });

  graph.edges.forEach((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) {
      return;
    }
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const midX = (source.x + target.x) / 2;
    path.setAttribute("d", `M ${source.x} ${source.y} C ${midX} ${source.y}, ${midX} ${target.y}, ${target.x} ${target.y}`);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", edge.channel === "summary" ? "#ffd087" : "#77a9ff");
    path.setAttribute("stroke-width", edge.channel === "summary" ? "2.3" : "1.7");
    path.setAttribute("opacity", "0.72");
    graphCanvas.appendChild(path);
  });

  graph.nodes.forEach((node) => {
    const pos = positions.get(node.node_id);
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(pos.x - 58));
    rect.setAttribute("y", String(pos.y - 26));
    rect.setAttribute("width", "116");
    rect.setAttribute("height", "52");
    rect.setAttribute("rx", "18");
    rect.setAttribute("fill", node.operator === "worker" ? "rgba(102, 213, 191, 0.15)" : "rgba(119, 169, 255, 0.15)");
    rect.setAttribute("stroke", node.operator === "judge" ? "#ffd087" : "rgba(236, 243, 255, 0.2)");
    rect.setAttribute("stroke-width", "1.2");
    group.appendChild(rect);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", String(pos.x));
    label.setAttribute("y", String(pos.y - 2));
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("class", "node-label");
    label.textContent = node.operator;
    group.appendChild(label);

    const role = document.createElementNS("http://www.w3.org/2000/svg", "text");
    role.setAttribute("x", String(pos.x));
    role.setAttribute("y", String(pos.y + 14));
    role.setAttribute("text-anchor", "middle");
    role.setAttribute("class", "node-role");
    role.textContent = node.model_tier;
    group.appendChild(role);

    graphCanvas.appendChild(group);
  });
}
