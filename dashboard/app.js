const summaryPath = "data/summary.json";

function $(id) {
  return document.getElementById(id);
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toFixed(digits);
}

function render(summary) {
  const testStatus = $("test-status");
  const testDetails = $("test-details");
  const updated = $("last-updated");

  if (summary.tests) {
    const passed = summary.tests.passed || 0;
    const failed = summary.tests.failed || 0;
    const status = failed === 0 && summary.tests.return_code === 0 ? "PASS" : "FAIL";
    testStatus.textContent = status;
    testStatus.className = `status ${status === "PASS" ? "pass" : "fail"}`;
    testDetails.textContent = `passed: ${passed}, failed: ${failed}, skipped: ${summary.tests.skipped || 0}`;
  } else {
    testStatus.textContent = "No test report";
    testDetails.textContent = "Run tools/run_tests_report.py to generate test data.";
  }

  const scores = summary.scores || [];
  $("count").textContent = scores.length.toString();
  $("avg").textContent = formatNumber(summary.avg_score);
  $("high").textContent = summary.high_tier_count?.toString() || "0";

  const tbody = $("scores-body");
  tbody.innerHTML = "";
  scores.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.image}</td>
      <td>${formatNumber(row.score)}</td>
      <td>${row.geo_tier || "-"}</td>
      <td>${formatNumber(row.geo_conf, 2)}</td>
      <td>${row.uncertainty_m ?? "-"}</td>
    `;
    tbody.appendChild(tr);
  });

  if (summary.generated_at) {
    updated.textContent = `Updated ${summary.generated_at}`;
  }
}

async function loadSummary() {
  try {
    const res = await fetch(summaryPath, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const summary = await res.json();
    render(summary);
  } catch (err) {
    render({ scores: [] });
  }
}

$("refresh").addEventListener("click", loadSummary);
loadSummary();
