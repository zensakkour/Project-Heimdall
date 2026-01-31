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

let mapView = null;
let trackView = null;
let currentRow = null;
let topLimit = 20;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function weightFrom(item) {
  const cand = item.candidate || {};
  return item.posterior_weight ?? cand.retrieval_score ?? 0;
}

function weightColor(weight) {
  const w = clamp(weight, 0, 1);
  const lightness = 35 + w * 35;
  return `hsl(172, 75%, ${lightness}%)`;
}

class GlobeView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.centerLon = 0;
    this.centerLat = 20;
    this.points = [];
    this.track = [];
    this.mean = null;
    this.ringRadiusM = null;
    this.dragging = false;
    this.lastX = 0;
    this.lastY = 0;
    this.canvas.style.cursor = "grab";
    this._bind();
    this.resize();
  }

  _bind() {
    const resize = () => this.resize();
    window.addEventListener("resize", resize);
    this.canvas.addEventListener("mousedown", (e) => {
      this.dragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      this.canvas.style.cursor = "grabbing";
    });
    window.addEventListener("mouseup", () => {
      this.dragging = false;
      this.canvas.style.cursor = "grab";
    });
    window.addEventListener("mousemove", (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastX;
      const dy = e.clientY - this.lastY;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      this.centerLon = (this.centerLon - dx * 0.3) % 360;
      this.centerLat = clamp(this.centerLat + dy * 0.2, -80, 80);
      this.render();
    });
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.floor(rect.width * ratio));
    this.canvas.height = Math.max(1, Math.floor(rect.height * ratio));
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.render();
  }

  setData({ points = [], mean = null, ringRadiusM = null, track = [] }) {
    this.points = points;
    this.mean = mean;
    this.ringRadiusM = ringRadiusM;
    this.track = track;
    this.render();
  }

  _project(lat, lon, r) {
    const rad = Math.PI / 180;
    const lat0 = this.centerLat * rad;
    const lon0 = this.centerLon * rad;
    const latR = lat * rad;
    const lonR = lon * rad;
    const cosc =
      Math.sin(lat0) * Math.sin(latR) + Math.cos(lat0) * Math.cos(latR) * Math.cos(lonR - lon0);
    if (cosc < 0) return null;
    const x = r * Math.cos(latR) * Math.sin(lonR - lon0);
    const y = r * (Math.cos(lat0) * Math.sin(latR) - Math.sin(lat0) * Math.cos(latR) * Math.cos(lonR - lon0));
    return { x, y };
  }

  _drawGraticule(cx, cy, r) {
    const ctx = this.ctx;
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    const latLines = [-60, -30, 0, 30, 60];
    latLines.forEach((lat) => {
      ctx.beginPath();
      let started = false;
      for (let lon = -180; lon <= 180; lon += 4) {
        const p = this._project(lat, lon, r);
        if (!p) {
          started = false;
          continue;
        }
        const x = cx + p.x;
        const y = cy + p.y;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    });
    for (let lon = -150; lon <= 150; lon += 30) {
      ctx.beginPath();
      let started = false;
      for (let lat = -80; lat <= 80; lat += 4) {
        const p = this._project(lat, lon, r);
        if (!p) {
          started = false;
          continue;
        }
        const x = cx + p.x;
        const y = cy + p.y;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }
  }

  _drawRing(cx, cy, r, center, radiusM) {
    if (!center || !radiusM) return;
    const earth = 6371000;
    const ang = radiusM / earth;
    const lat1 = (center.lat * Math.PI) / 180;
    const lon1 = (center.lon * Math.PI) / 180;
    const ctx = this.ctx;
    ctx.beginPath();
    let started = false;
    for (let b = 0; b <= 360; b += 6) {
      const brng = (b * Math.PI) / 180;
      const lat2 = Math.asin(
        Math.sin(lat1) * Math.cos(ang) + Math.cos(lat1) * Math.sin(ang) * Math.cos(brng)
      );
      const lon2 =
        lon1 +
        Math.atan2(
          Math.sin(brng) * Math.sin(ang) * Math.cos(lat1),
          Math.cos(ang) - Math.sin(lat1) * Math.sin(lat2)
        );
      const p = this._project((lat2 * 180) / Math.PI, (lon2 * 180) / Math.PI, r);
      if (!p) {
        started = false;
        continue;
      }
      const x = cx + p.x;
      const y = cy + p.y;
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.strokeStyle = "rgba(52,245,197,0.55)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([6, 6]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  render() {
    const ctx = this.ctx;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    const cx = w / 2;
    const cy = h / 2;
    const r = Math.min(w, h) * 0.38;
    ctx.clearRect(0, 0, w, h);
    const gradient = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.3, r * 0.2, cx, cy, r);
    gradient.addColorStop(0, "rgba(52, 245, 197, 0.12)");
    gradient.addColorStop(1, "rgba(10, 14, 20, 0.95)");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    this._drawGraticule(cx, cy, r);

    if (this.track.length > 1) {
      ctx.beginPath();
      let started = false;
      this.track.forEach((p) => {
        const proj = this._project(p.lat, p.lon, r);
        if (!proj) {
          started = false;
          return;
        }
        const x = cx + proj.x;
        const y = cy + proj.y;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.strokeStyle = "rgba(52, 245, 197, 0.6)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    this._drawRing(cx, cy, r, this.mean, this.ringRadiusM);

    this.points.forEach((p) => {
      const proj = this._project(p.lat, p.lon, r);
      if (!proj) return;
      ctx.beginPath();
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.alpha;
      ctx.arc(cx + proj.x, cy + proj.y, p.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    });

    if (this.mean) {
      const proj = this._project(this.mean.lat, this.mean.lon, r);
      if (proj) {
        ctx.beginPath();
        ctx.fillStyle = "rgba(255,255,255,0.9)";
        ctx.arc(cx + proj.x, cy + proj.y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.6)";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }
}

function renderMap(row) {
  if (!mapView) {
    mapView = new GlobeView($("map-canvas"));
  }

  const fusion = row?.fusion;
  const candidates = Array.isArray(fusion?.candidates) ? fusion.candidates : [];
  if (!fusion || candidates.length === 0) {
    mapView.setData({ points: [], mean: null, ringRadiusM: null, track: [] });
    return;
  }

  const sorted = [...candidates].sort((a, b) => weightFrom(b) - weightFrom(a));
  const maxWeight = sorted.length ? Math.max(...sorted.map(weightFrom)) : 1;
  const visible = sorted.slice(0, topLimit);
  const points = [];

  visible.forEach((item, idx) => {
    const cand = item.candidate || {};
    if (cand.latitude === undefined || cand.longitude === undefined) return;
    const rawWeight = weightFrom(item);
    const weight = maxWeight > 0 ? rawWeight / maxWeight : 0;
    const color = weightColor(weight);
    points.push({
      lat: cand.latitude,
      lon: cand.longitude,
      size: idx === 0 ? 6 : 2 + weight * 4,
      alpha: idx === 0 ? 0.9 : 0.4 + weight * 0.4,
      color,
    });
  });

  const meanLat = fusion.mean_latitude;
  const meanLon = fusion.mean_longitude;
  const ringRadius =
    fusion?.uncertainty_radius_m ??
    (fusion?.ellipse?.major_axis_m ? fusion.ellipse.major_axis_m * 0.6 : null);

  mapView.setData({
    points,
    mean: meanLat !== undefined && meanLon !== undefined ? { lat: meanLat, lon: meanLon } : null,
    ringRadiusM: ringRadius,
    track: [],
  });
}

function renderTrack(scores) {
  if (!trackView) {
    trackView = new GlobeView($("track-canvas"));
  }
  const points = (scores || [])
    .map((row) => {
      const fusion = row.fusion;
      if (!fusion) return null;
      if (fusion.mean_latitude === undefined || fusion.mean_longitude === undefined) return null;
      return { lat: fusion.mean_latitude, lon: fusion.mean_longitude };
    })
    .filter(Boolean);

  if (points.length < 2) {
    trackView.setData({ points: [], mean: null, ringRadiusM: null, track: [] });
    return;
  }

  trackView.setData({ points: [], mean: null, ringRadiusM: null, track: points });
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
      currentRow = row;
      renderFusionDetail(row);
      renderObjects(row);
    });
    tbody.appendChild(tr);
  });

  if (summary.generated_at) {
    updated.textContent = `Updated ${summary.generated_at}`;
  }

  if (scores.length > 0) {
    currentRow = scores[0];
    renderFusionDetail(scores[0]);
    renderObjects(scores[0]);
  } else {
    currentRow = null;
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

const topSelect = $("map-topn");
if (topSelect) {
  const stored = Number(localStorage.getItem("heimdallTopN") || "20");
  topSelect.value = String(stored);
  topLimit = stored;
  topSelect.addEventListener("change", () => {
    topLimit = Number(topSelect.value || "20");
    localStorage.setItem("heimdallTopN", String(topLimit));
    if (currentRow) renderMap(currentRow);
  });
}

