const summaryPath = "data/summary.json";

function $(id) {
  return document.getElementById(id);
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toFixed(digits);
}

function formatMaybe(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toFixed(digits);
}

let mapInstance = null;
let mapLayer = null;
let trackInstance = null;
let trackLayer = null;
const MAX_MAP_CANDIDATES = 20;

function ensureMap() {
  if (mapInstance) return;
  mapInstance = L.map("map-canvas", { zoomControl: true }).setView([0, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(mapInstance);
  mapLayer = L.layerGroup().addTo(mapInstance);
}

function ensureTrackMap() {
  if (trackInstance) return;
  trackInstance = L.map("track-canvas", { zoomControl: true }).setView([0, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(trackInstance);
  trackLayer = L.layerGroup().addTo(trackInstance);
}

function ellipsePolygon(lat, lon, majorM, minorM, orientationDeg) {
  const steps = 64;
  const points = [];
  const theta = (orientationDeg || 0) * (Math.PI / 180);
  const metersPerDegLat = 111320.0;
  const metersPerDegLon = 111320.0 * Math.cos((lat * Math.PI) / 180);
  for (let i = 0; i <= steps; i += 1) {
    const angle = (i / steps) * Math.PI * 2;
    const x = majorM * Math.cos(angle);
    const y = minorM * Math.sin(angle);
    const xr = x * Math.cos(theta) - y * Math.sin(theta);
    const yr = x * Math.sin(theta) + y * Math.cos(theta);
    const dLat = yr / metersPerDegLat;
    const dLon = xr / metersPerDegLon;
    points.push([lat + dLat, lon + dLon]);
  }
  return points;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function weightFrom(item) {
  const cand = item.candidate || {};
  return item.posterior_weight ?? cand.retrieval_score ?? 0;
}

function weightColor(weight) {
  const w = clamp(weight, 0, 1);
  const lightness = 38 + w * 30;
  return `hsl(172, 75%, ${lightness}%)`;
}

function renderMap(row) {
  ensureMap();
  mapLayer.clearLayers();

  const fusion = row?.fusion;
  const candidates = Array.isArray(fusion?.candidates) ? fusion.candidates : [];
  if (!fusion || candidates.length === 0) {
    mapInstance.setView([0, 0], 2);
    return;
  }

  const bounds = L.latLngBounds([]);
  const sorted = [...candidates].sort((a, b) => weightFrom(b) - weightFrom(a));
  const maxWeight = sorted.length ? Math.max(...sorted.map(weightFrom)) : 1;
  const visible = sorted.slice(0, MAX_MAP_CANDIDATES);

  visible.forEach((item, idx) => {
    const cand = item.candidate || {};
    if (cand.latitude === undefined || cand.longitude === undefined) return;
    const rawWeight = weightFrom(item);
    const weight = maxWeight > 0 ? rawWeight / maxWeight : 0;
    const color = weightColor(weight);
    const radius = idx === 0 ? 9 : 3 + weight * 6;
    const marker = L.circleMarker([cand.latitude, cand.longitude], {
      radius,
      color,
      fillColor: color,
      fillOpacity: idx === 0 ? 0.85 : 0.4 + weight * 0.35,
      weight: idx === 0 ? 2 : 1,
    }).addTo(mapLayer);
    marker.bindPopup(`Candidate<br/>w=${formatMaybe(rawWeight, 3)}`);
    bounds.extend([cand.latitude, cand.longitude]);
  });

  const meanLat = fusion.mean_latitude;
  const meanLon = fusion.mean_longitude;
  if (meanLat !== undefined && meanLon !== undefined) {
    const meanMarker = L.circleMarker([meanLat, meanLon], {
      radius: 8,
      color: "rgba(255, 255, 255, 0.9)",
      fillColor: "rgba(255, 255, 255, 0.9)",
      fillOpacity: 0.9,
      weight: 2,
    }).addTo(mapLayer);
    meanMarker.bindPopup("Fused mean");
    bounds.extend([meanLat, meanLon]);
  }

  const ellipse = fusion.ellipse || {};
  if (ellipse.major_axis_m && ellipse.minor_axis_m && meanLat !== undefined && meanLon !== undefined) {
    const polygon = L.polygon(
      ellipsePolygon(meanLat, meanLon, ellipse.major_axis_m, ellipse.minor_axis_m, ellipse.orientation_deg),
      {
        color: "rgba(52, 245, 197, 0.7)",
        weight: 1.4,
        dashArray: "6 6",
        fillColor: "rgba(52, 245, 197, 0.08)",
        fillOpacity: 0.12,
      }
    ).addTo(mapLayer);
    polygon.bindPopup("Uncertainty ellipse");
  } else if (fusion.uncertainty_radius_m && meanLat !== undefined && meanLon !== undefined) {
    L.circle([meanLat, meanLon], {
      radius: fusion.uncertainty_radius_m,
      color: "rgba(52, 245, 197, 0.6)",
      weight: 1.2,
      dashArray: "6 6",
      fillOpacity: 0.05,
    }).addTo(mapLayer);
  }

  mapInstance.fitBounds(bounds.pad(0.25));
}

function renderTrack(scores) {
  ensureTrackMap();
  trackLayer.clearLayers();
  const points = (scores || [])
    .map((row) => {
      const fusion = row.fusion;
      if (!fusion) return null;
      if (fusion.mean_latitude === undefined || fusion.mean_longitude === undefined) return null;
      return [fusion.mean_latitude, fusion.mean_longitude];
    })
    .filter(Boolean);

  if (points.length < 2) {
    trackInstance.setView([0, 0], 2);
    return;
  }

  const line = L.polyline(points, {
    color: "rgba(79, 209, 197, 0.9)",
    weight: 3,
  }).addTo(trackLayer);
  points.forEach((point) => {
    L.circleMarker(point, {
      radius: 4,
      color: "#ffd166",
      fillColor: "#ffd166",
      fillOpacity: 0.9,
      weight: 1,
    }).addTo(trackLayer);
  });
  trackInstance.fitBounds(line.getBounds().pad(0.2));
}

function renderFusionDetail(row) {
  const summary = $("fusion-summary");
  const grid = $("fusion-candidates");
  const tbody = $("candidate-table-body");
  grid.innerHTML = "";
  tbody.innerHTML = "";

  const fusion = row?.fusion;
  if (!fusion) {
    summary.textContent = "No fusion data available for this image.";
    renderMap(null);
    return;
  }

  const ellipse = fusion.ellipse || {};
  summary.textContent = `Mean: ${formatMaybe(fusion.mean_latitude, 5)}, ${formatMaybe(
    fusion.mean_longitude,
    5
  )} | Radius: ${formatMaybe(fusion.uncertainty_radius_m, 1)} m | Ellipse: ${formatMaybe(
    ellipse.major_axis_m,
    1
  )} x ${formatMaybe(ellipse.minor_axis_m, 1)} m @ ${formatMaybe(ellipse.orientation_deg, 1)} deg`;

  const candidates = Array.isArray(fusion.candidates) ? fusion.candidates : [];
  candidates.forEach((item, idx) => {
    const cand = item.candidate || {};
    const evidence = item.evidence || {};
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td>${formatMaybe(cand.latitude, 5)}</td>
      <td>${formatMaybe(cand.longitude, 5)}</td>
      <td>${formatMaybe(item.posterior_weight, 3)}</td>
      <td>${formatMaybe(cand.retrieval_score, 3)}</td>
      <td>${formatMaybe(evidence.shadow_residual_deg, 1)}</td>
      <td>${formatMaybe(evidence.terrain_residual, 1)}</td>
      <td>${cand.match_id || "-"}</td>
    `;
    tbody.appendChild(tr);
  });

  candidates.forEach((item, idx) => {
    const cand = item.candidate || {};
    const evidence = item.evidence || {};
    const card = document.createElement("div");
    card.className = "fusion-card";
    card.innerHTML = `
      <div class="fusion-header">#${idx + 1}  -  w=${formatMaybe(item.posterior_weight, 3)}</div>
      <div class="fusion-row">Lat/Lon: ${formatMaybe(cand.latitude, 5)}, ${formatMaybe(
        cand.longitude,
        5
      )}</div>
      <div class="fusion-row">Retrieval: ${formatMaybe(cand.retrieval_score, 3)} ${
      cand.match_id ? ` -  ${cand.match_id}` : ""
    }</div>
      <div class="fusion-row">Shadow residual: ${formatMaybe(
        evidence.shadow_residual_deg,
        1
      )} deg</div>
      <div class="fusion-row">Terrain residual: ${formatMaybe(evidence.terrain_residual, 1)}</div>
      <div class="fusion-row">Likelihoods: ${Object.entries(evidence.likelihoods || {})
        .map(([k, v]) => `${k}:${formatMaybe(v, 4)}`)
        .join(", ")}</div>
      <div class="fusion-row muted">${evidence.explanation || "-"}</div>
    `;
    grid.appendChild(card);
  });

  renderMap(row);
}

function renderObjects(row) {
  const body = $("object-table-body");
  const summary = $("verification-summary");
  body.innerHTML = "";
  if (!row) {
    summary.textContent = "Verification status will appear here.";
    return;
  }

  const verification = row.verification || {};
  if (verification) {
    const shadow = verification.shadow_ok === undefined ? "-" : verification.shadow_ok ? "ok" : "fail";
    const topo = verification.topo_ok === undefined ? "-" : verification.topo_ok ? "ok" : "fail";
    summary.textContent = `Shadow: ${shadow} | Terrain: ${topo} | ${verification.notes || ""}`;
  } else {
    summary.textContent = "No verification data.";
  }

  const detections = Array.isArray(row.detections) ? row.detections : [];
  if (!detections.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = "<td colspan=\"2\">No detections</td>";
    body.appendChild(tr);
    return;
  }
  detections.forEach((det) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${det.label || "-"}</td>
      <td>${formatMaybe(det.confidence, 2)}</td>
    `;
    body.appendChild(tr);
  });
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
    testDetails.textContent = "Run python -m src.tools.run_tests_report to generate test data.";
  }

  const scores = summary.scores || [];
  $("count").textContent = scores.length.toString();
  $("avg").textContent = formatNumber(summary.avg_score);
  $("high").textContent = summary.high_tier_count?.toString() || "0";

  const tbody = $("scores-body");
  tbody.innerHTML = "";
  scores.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.dataset.index = index.toString();
    tr.innerHTML = `
      <td>${row.image}</td>
      <td>${formatNumber(row.score)}</td>
      <td>${row.geo_tier || "-"}</td>
      <td>${formatNumber(row.geo_conf, 2)}</td>
      <td>${row.uncertainty_m ?? "-"}</td>
      <td>${row.fusion?.uncertainty_radius_m ?? "-"}</td>
    `;
    tr.addEventListener("click", () => {
      renderFusionDetail(row);
      renderObjects(row);
    });
    tbody.appendChild(tr);
  });

  if (summary.generated_at) {
    updated.textContent = `Updated ${summary.generated_at}`;
  }

  if (scores.length > 0) {
    renderFusionDetail(scores[0]);
    renderObjects(scores[0]);
  } else {
    renderFusionDetail(null);
    renderObjects(null);
  }

  renderTrack(scores);
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

