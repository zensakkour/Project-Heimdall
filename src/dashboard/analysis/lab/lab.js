import { byId } from "../shared.js";

const profileStorageKey = "heimdallProfile";
let activeProfile = "paris";

function syncProfileSelect() {
  const strategySelect = byId("geo-eval-strategy");
  if (!strategySelect) return;

  const storedProfile = localStorage.getItem(profileStorageKey);
  const initialProfile = storedProfile || strategySelect.value || "paris";
  activeProfile = initialProfile;
  strategySelect.value = initialProfile;

  strategySelect.addEventListener("change", () => {
    activeProfile = strategySelect.value || "paris";
    localStorage.setItem(profileStorageKey, activeProfile);
  });
}

function syncRetrievalToggle() {
  const retrievalToggle = byId("geo-eval-retrieval-only");
  if (!retrievalToggle) return;

  const stored = localStorage.getItem("heimdallRetrievalOnly");
  retrievalToggle.checked = stored !== "0";
  retrievalToggle.addEventListener("change", () => {
    localStorage.setItem("heimdallRetrievalOnly", retrievalToggle.checked ? "1" : "0");
  });
}

async function startGeoEval() {
  const imagesDir = byId("geo-eval-images")?.value?.trim() || "";
  const metadata = byId("geo-eval-metadata")?.value?.trim() || "";
  const limit = Number(byId("geo-eval-limit")?.value || "0");
  const retrievalToggle = byId("geo-eval-retrieval-only");
  const retrievalOnly = retrievalToggle ? Boolean(retrievalToggle.checked) : true;
  const strategySelect = byId("geo-eval-strategy");
  const selectedProfile = strategySelect?.value || activeProfile || "";
  const status = byId("geo-eval-status");
  const output = byId("geo-eval-output");

  if (!imagesDir || !metadata) {
    if (status) status.textContent = "Missing images dir or metadata path.";
    return;
  }

  const params = new URLSearchParams({
    images_dir: imagesDir,
    metadata,
    limit: String(Number.isFinite(limit) ? limit : 0),
    profile: selectedProfile,
    retrieval_only: retrievalOnly ? "1" : "0",
  });

  if (status) status.textContent = "Starting...";
  if (output) output.textContent = "Running...";
  await fetch(`/eval/geo/start?${params.toString()}`, { method: "POST" });
  pollGeoEval();
}

async function pollGeoEval() {
  const status = byId("geo-eval-status");
  const output = byId("geo-eval-output");
  const bar = byId("geo-eval-progress-bar");
  const text = byId("geo-eval-progress-text");
  const wrap = byId("geo-eval-progress");
  const res = await fetch("/eval/geo/status");
  if (!res.ok) return;

  const data = await res.json();
  if (status) status.textContent = data.status || "idle";
  if (wrap) wrap.classList.toggle("active", data.status === "running");

  if (data.progress && bar && text) {
    const total = Number(data.progress.total || 0);
    const done = Number(data.progress.processed || 0);
    const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
    bar.style.width = `${pct}%`;
    text.textContent = `${pct}%`;
  }

  if (data.last_result && output) {
    output.textContent = data.last_result;
  }

  if (data.status === "running") {
    setTimeout(pollGeoEval, 1200);
  }
}

function _fmtMetricValue(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function _shortImageName(pathValue) {
  const raw = String(pathValue || "").trim();
  if (!raw) return "-";
  const parts = raw.split(/[\\/]/);
  return parts.length ? parts[parts.length - 1] : raw;
}

function _formatGeoRandomOutput(raw) {
  let payload = null;
  try {
    payload = typeof raw === "string" ? JSON.parse(raw) : raw;
  } catch {
    return String(raw || "No random sample results yet.");
  }
  if (!payload || typeof payload !== "object") return "No random sample results yet.";

  const lines = [];
  lines.push(
    `Seed: ${payload.seed ?? "-"} | Requested: ${payload.requested_samples ?? "-"} | ` +
      `Evaluated: ${payload.evaluated ?? "-"}`
  );
  lines.push(
    `Distance: mean ${_fmtMetricValue(payload.mean_km, 3)} km | ` +
      `median ${_fmtMetricValue(payload.median_km, 3)} km | ` +
      `p90 ${_fmtMetricValue(payload.p90_km, 3)} km`
  );
  lines.push(
    `Accuracy: <=1km ${_fmtMetricValue(payload.within_1km_pct, 2)}% | ` +
      `<=2km ${_fmtMetricValue(payload.within_2km_pct, 2)}% | ` +
      `<=5km ${_fmtMetricValue(payload.within_5km_pct, 2)}% | ` +
      `<=10km ${_fmtMetricValue(payload.within_10km_pct, 2)}%`
  );

  const samples = Array.isArray(payload.samples) ? payload.samples : [];
  if (!samples.length) return lines.join("\n");

  const rows = samples
    .map((item) => {
      const dist = Number(item?.dist_km);
      return {
        image: _shortImageName(item?.image),
        distKm: Number.isFinite(dist) ? dist : Number.POSITIVE_INFINITY,
      };
    })
    .sort((a, b) => b.distKm - a.distKm);

  lines.push("");
  lines.push("Worst sample distances:");
  rows.slice(0, Math.min(12, rows.length)).forEach((row, idx) => {
    lines.push(`${String(idx + 1).padStart(2, " ")}. ${_fmtMetricValue(row.distKm, 3)} km | ${row.image}`);
  });
  return lines.join("\n");
}

async function startGeoRandomEval() {
  const imagesDir = byId("geo-eval-images")?.value?.trim() || "";
  const metadata = byId("geo-eval-metadata")?.value?.trim() || "";
  const sampleSize = Number(byId("geo-random-size")?.value || "16");
  const retrievalToggle = byId("geo-eval-retrieval-only");
  const retrievalOnly = retrievalToggle ? Boolean(retrievalToggle.checked) : true;
  const strategySelect = byId("geo-eval-strategy");
  const selectedProfile = strategySelect?.value || activeProfile || "";
  const status = byId("geo-random-status");
  const output = byId("geo-random-output");

  if (!imagesDir || !metadata) {
    if (status) status.textContent = "Missing images dir or metadata path.";
    return;
  }

  const params = new URLSearchParams({
    images_dir: imagesDir,
    metadata,
    sample_size: String(Number.isFinite(sampleSize) ? Math.max(1, Math.floor(sampleSize)) : 16),
    profile: selectedProfile,
    retrieval_only: retrievalOnly ? "1" : "0",
  });

  if (status) status.textContent = "Starting random sample run...";
  if (output) output.textContent = "Running random samples...";
  await fetch(`/eval/geo/random/start?${params.toString()}`, { method: "POST" });
  pollGeoRandomEval();
}

async function pollGeoRandomEval() {
  const status = byId("geo-random-status");
  const output = byId("geo-random-output");
  const bar = byId("geo-random-progress-bar");
  const text = byId("geo-random-progress-text");
  const wrap = byId("geo-random-progress");
  const res = await fetch("/eval/geo/random/status");
  if (!res.ok) return;

  const data = await res.json();
  if (status) {
    const seed = data.seed ? ` (seed ${data.seed})` : "";
    status.textContent = `${data.status || "idle"}${seed}`;
  }
  if (wrap) wrap.classList.toggle("active", data.status === "running");

  if (data.progress && bar && text) {
    const total = Number(data.progress.total || 0);
    const done = Number(data.progress.processed || 0);
    const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;
    bar.style.width = `${pct}%`;
    text.textContent = `${pct}%`;
  }

  if (data.last_result && output) {
    output.textContent = _formatGeoRandomOutput(data.last_result);
  }

  if (data.status === "running") {
    setTimeout(pollGeoRandomEval, 1200);
  }
}

const geoEvalBtn = byId("geo-eval-run");
if (geoEvalBtn) {
  geoEvalBtn.addEventListener("click", startGeoEval);
}

const geoRandomBtn = byId("geo-random-run");
if (geoRandomBtn) {
  geoRandomBtn.addEventListener("click", startGeoRandomEval);
}

async function pickPath(endpoint, targetId) {
  const input = byId(targetId);
  if (!input) return;
  try {
    const res = await fetch(endpoint, { method: "POST" });
    const data = await res.json();
    if (data.path) {
      input.value = data.path;
    }
  } catch (err) {
    console.error(err);
  }
}

const browseImages = byId("geo-eval-browse-images");
if (browseImages) {
  browseImages.addEventListener("click", () => pickPath("/fs/pick_dir", "geo-eval-images"));
}

const browseMetadata = byId("geo-eval-browse-metadata");
if (browseMetadata) {
  browseMetadata.addEventListener("click", () => pickPath("/fs/pick_file", "geo-eval-metadata"));
}

function fmtMetric(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function useSavedBenchmarkView() {
  const toggle = byId("bench-view-history");
  return Boolean(toggle && toggle.checked);
}

let benchmarkRunsCache = [];
let currentBenchmarkRun = null;

function fmtRunLabel(run) {
  if (!run) return "unknown";
  const runId = run.run_id || "-";
  const generatedAt = run.generated_at || "-";
  return `${generatedAt} | run_id=${runId}`;
}

function setBenchmarkOutputMeta(source, run) {
  const el = byId("bench-output-meta");
  if (!el) return;
  if (source === "current" && run) {
    el.textContent = `Showing: current run | ${fmtRunLabel(run)}`;
    return;
  }
  if (source === "saved" && run) {
    el.textContent = `Showing: selected saved run | ${fmtRunLabel(run)}`;
    return;
  }
  el.textContent = "Showing: no run selected.";
}

function populateBenchmarkCompareSelectors(runs) {
  const baselineSelect = byId("bench-compare-baseline");
  const candidateSelect = byId("bench-compare-candidate");
  if (!baselineSelect || !candidateSelect) return;

  const previousBaseline = baselineSelect.value || "";
  const previousCandidate = candidateSelect.value || "";
  baselineSelect.innerHTML = "";
  candidateSelect.innerHTML = "";

  if (!runs.length) {
    const emptyBaseline = document.createElement("option");
    emptyBaseline.value = "";
    emptyBaseline.textContent = "No saved runs yet";
    baselineSelect.appendChild(emptyBaseline);

    const emptyCandidate = document.createElement("option");
    emptyCandidate.value = "";
    emptyCandidate.textContent = "No saved runs yet";
    candidateSelect.appendChild(emptyCandidate);
    return;
  }

  runs.forEach((run) => {
    const runId = run?.run_id || "";
    if (!runId) return;
    const label = `${run?.generated_at || runId} | Best: ${run?.best_model || "-"}`;

    const bOpt = document.createElement("option");
    bOpt.value = runId;
    bOpt.textContent = label;
    baselineSelect.appendChild(bOpt);

    const cOpt = document.createElement("option");
    cOpt.value = runId;
    cOpt.textContent = label;
    candidateSelect.appendChild(cOpt);
  });

  const runIds = runs.map((run) => run?.run_id).filter(Boolean);
  const defaultCandidate = runIds[0] || "";
  const defaultBaseline = runIds[1] || defaultCandidate;

  baselineSelect.value = runIds.includes(previousBaseline) ? previousBaseline : defaultBaseline;
  candidateSelect.value = runIds.includes(previousCandidate) ? previousCandidate : defaultCandidate;
}

function updateSelectedBenchmarkRunMeta() {
  const selectEl = byId("bench-run-history");
  const metaEl = byId("bench-run-meta");
  const mode = useSavedBenchmarkView() ? "selected saved run" : "current run only";
  if (!metaEl) return;

  if (!useSavedBenchmarkView()) {
    if (currentBenchmarkRun) {
      metaEl.textContent = `Mode: ${mode} | Current run: ${fmtRunLabel(currentBenchmarkRun)}`;
    } else {
      metaEl.textContent = `Mode: ${mode} | No current run yet.`;
    }
    return;
  }

  if (!selectEl || !selectEl.value) {
    metaEl.textContent = `Mode: ${mode} | No historical run selected.`;
    return;
  }

  const selected = selectEl.selectedOptions && selectEl.selectedOptions.length
    ? selectEl.selectedOptions[0]
    : null;
  if (!selected) {
    metaEl.textContent = `Mode: ${mode} | No historical run selected.`;
    return;
  }

  const generatedAt = selected.dataset.generatedAt || selected.value;
  const bestModel = selected.dataset.bestModel || "-";
  const modelCount = selected.dataset.modelCount || "0";
  metaEl.textContent =
    `Mode: ${mode} | Selected run: ${generatedAt} | Best: ${bestModel} | Models: ${modelCount}`;
}

async function refreshBenchmarkRuns(preferredRunId = null) {
  const selectEl = byId("bench-run-history");
  if (!selectEl) return [];
  const currentRunId = preferredRunId || selectEl.value || "";
  let runs = [];

  try {
    const res = await fetch("/eval/benchmarks/runs?limit=200", { cache: "no-store" });
    if (!res.ok) return [];
    const payload = await res.json();
    runs = Array.isArray(payload?.runs) ? payload.runs : [];
  } catch {
    benchmarkRunsCache = [];
    populateBenchmarkCompareSelectors([]);
    return [];
  }

  benchmarkRunsCache = runs;
  selectEl.innerHTML = "";
  if (!runs.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No saved runs yet";
    selectEl.appendChild(option);
    populateBenchmarkCompareSelectors([]);
    updateSelectedBenchmarkRunMeta();
    return [];
  }

  runs.forEach((run) => {
    const runId = run?.run_id || "";
    if (!runId) return;
    const option = document.createElement("option");
    option.value = runId;
    option.dataset.generatedAt = run?.generated_at || runId;
    option.dataset.bestModel = run?.best_model || "-";
    option.dataset.modelCount = String(Number(run?.model_count || 0));
    option.textContent = `${option.dataset.generatedAt} | Best: ${option.dataset.bestModel}`;
    selectEl.appendChild(option);
  });

  const target = currentRunId && runs.some((run) => run?.run_id === currentRunId)
    ? currentRunId
    : runs[0]?.run_id;
  if (target) {
    selectEl.value = target;
  }

  populateBenchmarkCompareSelectors(runs);
  updateSelectedBenchmarkRunMeta();
  return runs;
}

async function loadSelectedBenchmarkRun(options = {}) {
  const selectEl = byId("bench-run-history");
  const statusEl = byId("bench-status");
  const outputEl = byId("bench-output");
  const silent = Boolean(options?.silent);
  const force = Boolean(options?.force);
  if (!force && !useSavedBenchmarkView()) return;
  if (!selectEl || !selectEl.value) return;

  const runId = selectEl.value;
  if (!silent && statusEl) {
    statusEl.textContent = `Loading saved run: ${runId}`;
  }

  try {
    const res = await fetch(`/eval/benchmarks/runs/${encodeURIComponent(runId)}`, { cache: "no-store" });
    if (!res.ok) {
      if (statusEl) statusEl.textContent = `Failed to load run: ${runId}`;
      return;
    }

    const payload = await res.json();
    renderBenchmarkSummary(payload);
    setBenchmarkOutputMeta("saved", payload);
    if (outputEl) outputEl.textContent = JSON.stringify(payload, null, 2);
    if (!silent && statusEl) {
      statusEl.textContent = `Loaded run: ${payload?.generated_at || runId}`;
    }
  } catch {
    if (statusEl) statusEl.textContent = `Failed to load run: ${runId}`;
  }
}

function renderBenchmarkCompareSummary(compare) {
  const statusEl = byId("bench-compare-status");
  if (!statusEl) return;
  const baseline = compare?.baseline_generated_at || compare?.baseline_run_id || "-";
  const candidate = compare?.candidate_generated_at || compare?.candidate_run_id || "-";
  const scenarioCount = Array.isArray(compare?.scenario_deltas) ? compare.scenario_deltas.length : 0;
  const modelCount = Array.isArray(compare?.model_deltas) ? compare.model_deltas.length : 0;
  statusEl.textContent =
    `Compared baseline ${baseline} vs candidate ${candidate} | ` +
    `Scenarios: ${scenarioCount} | Models: ${modelCount}`;
}

async function runBenchmarkComparison(options = {}) {
  const baselineSelect = byId("bench-compare-baseline");
  const candidateSelect = byId("bench-compare-candidate");
  const outputEl = byId("bench-compare-output");
  const statusEl = byId("bench-compare-status");
  const appendProgress = Boolean(options?.appendProgress);
  if (!baselineSelect || !candidateSelect) return;

  const baselineRunId = baselineSelect.value || "";
  const candidateRunId = candidateSelect.value || "";
  if (!baselineRunId || !candidateRunId) {
    if (statusEl) statusEl.textContent = "Pick baseline and candidate runs first.";
    return;
  }
  if (baselineRunId === candidateRunId) {
    if (statusEl) statusEl.textContent = "Baseline and candidate must be different runs.";
    return;
  }

  if (statusEl) statusEl.textContent = "Comparing benchmark runs...";
  const params = new URLSearchParams({
    baseline_run_id: baselineRunId,
    candidate_run_id: candidateRunId,
    append_progress: appendProgress ? "1" : "0",
  });

  try {
    const res = await fetch(`/eval/benchmarks/compare?${params.toString()}`, { method: "POST" });
    if (!res.ok) {
      let msg = `Compare failed (${res.status})`;
      try {
        const err = await res.json();
        if (err?.error) msg = `Compare failed: ${err.error}`;
      } catch {
        // Keep default message.
      }
      if (statusEl) statusEl.textContent = msg;
      return;
    }

    const payload = await res.json();
    renderBenchmarkCompareSummary(payload);
    if (statusEl && appendProgress) {
      statusEl.textContent += payload.progress_appended
        ? " | Appended to PROGRESS.md"
        : " | Did not append to PROGRESS.md";
    }

    if (outputEl) {
      const snippet = payload?.progress_md_snippet
        ? `${payload.progress_md_snippet}\n\n`
        : "";
      outputEl.textContent = snippet + JSON.stringify(payload, null, 2);
    }
  } catch {
    if (statusEl) statusEl.textContent = "Compare failed due to network/server error.";
  }
}

function renderBenchmarkSummary(summary) {
  const geoBody = byId("bench-geo-body");
  const modelBody = byId("bench-model-body");
  const bestModel = byId("bench-best-model");
  if (!geoBody || !modelBody || !bestModel) return;

  const geoRows = Array.isArray(summary?.geo_scenarios) ? summary.geo_scenarios : [];
  geoBody.innerHTML = "";
  if (!geoRows.length) {
    geoBody.innerHTML = '<tr><td colspan="5">No scenario metrics.</td></tr>';
  } else {
    geoRows.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.scenario || row.name || "-"}</td>
        <td>${fmtMetric(row.mean_km, 3)}</td>
        <td>${fmtMetric(row.median_km, 3)}</td>
        <td>${fmtMetric(row.within_5km_pct, 2)}</td>
        <td>${fmtMetric(row.within_10km_pct, 2)}</td>
      `;
      geoBody.appendChild(tr);
    });
  }

  const modelRows = Array.isArray(summary?.backbone_benchmark?.models)
    ? summary.backbone_benchmark.models
    : [];
  modelBody.innerHTML = "";
  if (!modelRows.length) {
    modelBody.innerHTML = '<tr><td colspan="5">No backbone metrics.</td></tr>';
  } else {
    modelRows.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.model_id || "-"}</td>
        <td>${fmtMetric(row.mean_km, 3)}</td>
        <td>${fmtMetric(row.median_km, 3)}</td>
        <td>${fmtMetric(row.within_5km_pct, 2)}</td>
        <td>${fmtMetric(row.within_10km_pct, 2)}</td>
      `;
      modelBody.appendChild(tr);
    });
  }

  const considered = modelRows
    .map((row) => row?.model_id)
    .filter((id) => typeof id === "string" && id.trim().length > 0);
  const uniqueConsidered = [...new Set(considered)];
  const consideredText = uniqueConsidered.length ? uniqueConsidered.join(", ") : "-";
  bestModel.textContent =
    `Best model: ${summary?.backbone_benchmark?.best_model || "-"} | ` +
    `Models considered: ${consideredText}`;
}

async function pollBenchmarks() {
  const statusEl = byId("bench-status");
  const outputEl = byId("bench-output");
  const progressWrap = byId("bench-progress");
  const progressBar = byId("bench-progress-bar");
  const progressText = byId("bench-progress-text");
  let data;

  try {
    const res = await fetch("/eval/benchmarks/status", { cache: "no-store" });
    if (!res.ok) {
      if (statusEl) statusEl.textContent = `Status: polling failed (${res.status})`;
      setTimeout(pollBenchmarks, 2000);
      return;
    }
    data = await res.json();
  } catch (err) {
    if (statusEl) statusEl.textContent = `Status: polling failed (${String(err)})`;
    setTimeout(pollBenchmarks, 2000);
    return;
  }

  const status = data.status || "idle";
  const stage = data.stage || "-";
  if (statusEl) statusEl.textContent = `Status: ${status} | Stage: ${stage}`;

  if (data.progress && progressBar && progressText) {
    const total = Number(data.progress.total || 0);
    const current = Number(data.progress.current || 0);
    const pctRaw = Number(data.progress.percent);
    const pct =
      Number.isFinite(pctRaw) && pctRaw >= 0
        ? Math.min(100, Math.max(0, Math.round(pctRaw)))
        : total > 0
          ? Math.min(100, Math.max(0, Math.round((current / total) * 100)))
          : 0;
    const msg = data.progress.message || "";
    progressBar.style.width = `${pct}%`;
    progressText.textContent = msg ? `${pct}% - ${msg}` : `${pct}%`;
  }

  if (progressWrap) {
    const showProgress =
      status === "running" ||
      status === "done" ||
      status === "error" ||
      Boolean(data.progress);
    progressWrap.classList.toggle("active", showProgress);
  }

  if (outputEl && data.last_result && !useSavedBenchmarkView()) {
    outputEl.textContent = data.last_result;
  }

  if (status === "done") {
    const doneRunId = data.run_id || null;
    await refreshBenchmarkRuns(doneRunId);
    if (!data.last_result) {
      if (!useSavedBenchmarkView()) setBenchmarkOutputMeta("none", null);
      return;
    }

    try {
      const parsed = JSON.parse(data.last_result);
      currentBenchmarkRun = parsed;
      if (!useSavedBenchmarkView()) {
        renderBenchmarkSummary(parsed);
        setBenchmarkOutputMeta("current", parsed);
        if (outputEl) outputEl.textContent = JSON.stringify(parsed, null, 2);
      }
    } catch {
      if (doneRunId) {
        try {
          const res = await fetch(`/eval/benchmarks/runs/${encodeURIComponent(doneRunId)}`, {
            cache: "no-store",
          });
          if (res.ok) {
            const persisted = await res.json();
            currentBenchmarkRun = persisted;
            if (!useSavedBenchmarkView()) {
              renderBenchmarkSummary(persisted);
              setBenchmarkOutputMeta("current", persisted);
              if (outputEl) outputEl.textContent = JSON.stringify(persisted, null, 2);
            }
          }
        } catch {
          // Keep raw output visible.
        }
      }
    }
  }

  if (status === "error") {
    await refreshBenchmarkRuns(data.run_id || null);
  }

  if (status === "running") {
    setTimeout(pollBenchmarks, 1500);
  }
}

async function startBenchmarks() {
  const statusEl = byId("bench-status");
  const outputEl = byId("bench-output");
  const progressWrap = byId("bench-progress");
  const progressBar = byId("bench-progress-bar");
  const progressText = byId("bench-progress-text");
  const params = new URLSearchParams({
    images_dir: byId("bench-images-dir")?.value?.trim() || "data/spacenet_paris_test/chips",
    metadata: byId("bench-metadata")?.value?.trim() || "data/spacenet_paris_test/metadata.csv",
    limit: String(Number(byId("bench-geo-limit")?.value || "120")),
    train_images_dir: byId("bench-train-images-dir")?.value?.trim() || "data/spacenet_paris/chips",
    train_metadata: byId("bench-train-metadata")?.value?.trim() || "data/spacenet_paris/metadata.csv",
    eval_images_dir: byId("bench-images-dir")?.value?.trim() || "data/spacenet_paris_test/chips",
    eval_metadata: byId("bench-metadata")?.value?.trim() || "data/spacenet_paris_test/metadata.csv",
    train_limit: String(Number(byId("bench-train-limit")?.value || "120")),
    eval_limit: String(Number(byId("bench-eval-limit")?.value || "60")),
    model_ids:
      byId("bench-model-ids")?.value?.trim() ||
      "openai/clip-vit-large-patch14,google/siglip-base-patch16-224",
    reuse_indices: byId("bench-reuse-indices")?.checked ? "1" : "0",
  });

  if (statusEl) statusEl.textContent = "Status: starting...";
  if (outputEl) outputEl.textContent = "Running benchmark comparison...";
  setBenchmarkOutputMeta("none", null);
  if (progressWrap) progressWrap.classList.add("active");
  if (progressBar) progressBar.style.width = "0%";
  if (progressText) progressText.textContent = "0% - Preparing benchmark jobs";

  try {
    const res = await fetch(`/eval/benchmarks/start?${params.toString()}`, { method: "POST" });
    if (!res.ok) {
      const text = await res.text();
      if (statusEl) statusEl.textContent = `Status: failed to start (${res.status})`;
      if (outputEl) outputEl.textContent = text || "Failed to start benchmark run.";
      return;
    }
  } catch (err) {
    if (statusEl) statusEl.textContent = "Status: failed to start";
    if (outputEl) outputEl.textContent = `Failed to start benchmark run: ${String(err)}`;
    return;
  }

  pollBenchmarks();
}

const benchRunBtn = byId("bench-run");
if (benchRunBtn) {
  benchRunBtn.addEventListener("click", startBenchmarks);
}

const benchRunRefreshBtn = byId("bench-run-refresh");
if (benchRunRefreshBtn) {
  benchRunRefreshBtn.addEventListener("click", async () => {
    await refreshBenchmarkRuns();
    if (useSavedBenchmarkView()) {
      await loadSelectedBenchmarkRun({ silent: true, force: true });
    }
  });
}

const benchRunLoadBtn = byId("bench-run-load");
if (benchRunLoadBtn) {
  benchRunLoadBtn.addEventListener("click", async () => {
    const toggle = byId("bench-view-history");
    if (toggle) toggle.checked = true;
    updateSelectedBenchmarkRunMeta();
    await loadSelectedBenchmarkRun({ force: true });
  });
}

const benchRunHistorySelect = byId("bench-run-history");
if (benchRunHistorySelect) {
  benchRunHistorySelect.addEventListener("change", async () => {
    updateSelectedBenchmarkRunMeta();
    if (useSavedBenchmarkView()) {
      await loadSelectedBenchmarkRun({ silent: true, force: true });
    }
  });
}

const benchCompareRunBtn = byId("bench-compare-run");
if (benchCompareRunBtn) {
  benchCompareRunBtn.addEventListener("click", async () => {
    await runBenchmarkComparison({ appendProgress: false });
  });
}

const benchCompareAppendBtn = byId("bench-compare-append-progress");
if (benchCompareAppendBtn) {
  benchCompareAppendBtn.addEventListener("click", async () => {
    await runBenchmarkComparison({ appendProgress: true });
  });
}

const benchViewHistoryToggle = byId("bench-view-history");
if (benchViewHistoryToggle) {
  benchViewHistoryToggle.addEventListener("change", async () => {
    updateSelectedBenchmarkRunMeta();
    if (useSavedBenchmarkView()) {
      await loadSelectedBenchmarkRun({ force: true });
      return;
    }
    if (currentBenchmarkRun) {
      renderBenchmarkSummary(currentBenchmarkRun);
      const outputEl = byId("bench-output");
      if (outputEl) outputEl.textContent = JSON.stringify(currentBenchmarkRun, null, 2);
      setBenchmarkOutputMeta("current", currentBenchmarkRun);
    } else {
      const outputEl = byId("bench-output");
      if (outputEl) outputEl.textContent = "No benchmark output yet. Run a benchmark first.";
      setBenchmarkOutputMeta("none", null);
    }
  });
}

async function initBenchmarkHistory() {
  const runs = await refreshBenchmarkRuns();
  if (runs.length && useSavedBenchmarkView()) {
    await loadSelectedBenchmarkRun({ silent: true, force: true });
    return;
  }
  updateSelectedBenchmarkRunMeta();
  setBenchmarkOutputMeta("none", null);
}

function init() {
  syncProfileSelect();
  syncRetrievalToggle();
  initBenchmarkHistory();
}

window.addEventListener("load", init);
