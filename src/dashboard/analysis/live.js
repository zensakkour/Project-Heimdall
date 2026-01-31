async function postForm(url, formData) {
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

function byId(id) {
  return document.getElementById(id);
}

function setLoading(active, text) {
  const progress = byId("progress");
  const progressText = byId("progress-text");
  const analyzeImage = byId("analyze-image");
  const analyzeVideo = byId("analyze-video");
  if (progress) {
    progress.classList.toggle("active", active);
    if (text && progressText) progressText.textContent = text;
  }
  if (analyzeImage) analyzeImage.disabled = active;
  if (analyzeVideo) analyzeVideo.disabled = active;
}

function renderSummary(result) {
  const summary = byId("summary");
  const geo = result.result.geo;
  const fusion = result.result.fusion;
  const geoDebug = result.geo_debug;
  const tierFromFusion = () => {
    if (!fusion || !Array.isArray(fusion.candidates) || fusion.candidates.length === 0) return "";
    const weight = fusion.candidates[0].posterior_weight ?? 0;
    if (weight >= 0.75) return "high";
    if (weight >= 0.45) return "medium";
    return "low";
  };
  const tierFromCandidates = () => {
    const candidates = result.result.candidates || [];
    if (!Array.isArray(candidates) || candidates.length === 0) return "";
    const score = candidates[0].retrieval_score ?? 0;
    if (score >= 0.75) return "high";
    if (score >= 0.45) return "medium";
    return "low";
  };
  const fusionText = fusion
    ? ` | Fusion radius: ${fusion.uncertainty_radius_m?.toFixed(1) ?? "-"}m`
    : "";
  const debugText = geoDebug
    ? ` | Geo candidates: ${geoDebug.candidate_count ?? 0}${geoDebug.error ? ` (${geoDebug.error})` : ""}`
    : "";
  summary.textContent = `Score: ${result.result.score.toFixed(3)} | Geo tier: ${
    geo?.confidence_tier || tierFromFusion() || tierFromCandidates() || "-"
  } | Detections: ${result.result.detections.length}${fusionText}${debugText}`;
}

let currentImage = null;
let currentDetections = [];
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let dragging = false;
let lastX = 0;
let lastY = 0;

let liveMap = null;
let liveLayer = null;
let trackMap = null;
let trackLayer = null;
const MAX_MAP_CANDIDATES = 20;

function ensureLiveMaps() {
  if (!liveMap) {
    liveMap = L.map("live-map", { zoomControl: true }).setView([0, 0], 2);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(liveMap);
    liveLayer = L.layerGroup().addTo(liveMap);
  }
  if (!trackMap) {
    trackMap = L.map("live-track", { zoomControl: true }).setView([0, 0], 2);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(trackMap);
    trackLayer = L.layerGroup().addTo(trackMap);
  }
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

function renderLiveMap(result) {
  ensureLiveMaps();
  liveLayer.clearLayers();
  if (!result || !result.result) {
    liveMap.setView([0, 0], 2);
    return;
  }
  const fusion = result.result.fusion;
  const candidates = Array.isArray(fusion?.candidates) ? fusion.candidates : [];
  if (!fusion || candidates.length === 0) {
    liveMap.setView([0, 0], 2);
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
    L.circleMarker([cand.latitude, cand.longitude], {
      radius: idx === 0 ? 9 : 3 + weight * 6,
      color,
      fillColor: color,
      fillOpacity: idx === 0 ? 0.85 : 0.4 + weight * 0.35,
      weight: idx === 0 ? 2 : 1,
    }).addTo(liveLayer);
    bounds.extend([cand.latitude, cand.longitude]);
  });
  const meanLat = fusion.mean_latitude;
  const meanLon = fusion.mean_longitude;
  if (meanLat !== undefined && meanLon !== undefined) {
    L.circleMarker([meanLat, meanLon], {
      radius: 8,
      color: "rgba(255, 255, 255, 0.9)",
      fillColor: "rgba(255, 255, 255, 0.9)",
      fillOpacity: 0.9,
      weight: 2,
    }).addTo(liveLayer);
    bounds.extend([meanLat, meanLon]);
  }
  const ellipse = fusion.ellipse || {};
  if (ellipse.major_axis_m && ellipse.minor_axis_m && meanLat !== undefined && meanLon !== undefined) {
    L.polygon(ellipsePolygon(meanLat, meanLon, ellipse.major_axis_m, ellipse.minor_axis_m, ellipse.orientation_deg), {
      color: "rgba(52, 245, 197, 0.7)",
      weight: 1.4,
      dashArray: "6 6",
      fillColor: "rgba(52, 245, 197, 0.08)",
      fillOpacity: 0.12,
    }).addTo(liveLayer);
  }
  liveMap.fitBounds(bounds.pad(0.3));
}

function renderLiveTrack(frames) {
  ensureLiveMaps();
  trackLayer.clearLayers();
  const points = (frames || [])
    .map((frame) => {
      const fusion = frame.result?.fusion;
      if (!fusion) return null;
      if (fusion.mean_latitude === undefined || fusion.mean_longitude === undefined) return null;
      return [fusion.mean_latitude, fusion.mean_longitude];
    })
    .filter(Boolean);
  if (points.length < 2) {
    trackMap.setView([0, 0], 2);
    return;
  }
  const line = L.polyline(points, {
    color: "rgba(52, 245, 197, 0.9)",
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
  trackMap.fitBounds(line.getBounds().pad(0.25));
}

function renderImage(dataUrl, detections) {
  currentDetections = detections || [];
  const canvas = byId("canvas");
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    canvas.width = img.width;
    canvas.height = img.height;
    scale = 1;
    offsetX = 0;
    offsetY = 0;
    drawScene(null);
  };
  img.src = dataUrl;
}

function renderFrames(frames) {
  const container = byId("frames");
  container.innerHTML = "";
  frames.forEach((frame) => {
    const card = document.createElement("div");
    card.className = "frame-card";
    card.innerHTML = `
      <img src="${frame.image_data}" alt="frame" />
      <div class="frame-meta">
        t=${frame.timestamp_s.toFixed(1)}s | score=${frame.result.score.toFixed(3)}
      </div>
    `;
    container.appendChild(card);
  });
}

function drawDetections(ctx, detections, activeIndex) {
  if (!detections) return;
  detections.forEach((det, idx) => {
    const pts = det.obb || [];
    if (pts.length !== 4) return;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    ctx.lineTo(pts[1][0], pts[1][1]);
    ctx.lineTo(pts[2][0], pts[2][1]);
    ctx.lineTo(pts[3][0], pts[3][1]);
    ctx.closePath();
    if (idx === activeIndex) {
      ctx.fillStyle = "rgba(52,245,197,0.18)";
      ctx.fill();
      ctx.lineWidth = 5;
      ctx.strokeStyle = "#34f5c5";
    } else {
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(90, 98, 108, 0.45)";
    }
    ctx.stroke();
  });
}

function drawScene(activeIndex) {
  const canvas = byId("canvas");
  const ctx = canvas.getContext("2d");
  if (!currentImage) return;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(scale, 0, 0, scale, offsetX, offsetY);
  ctx.drawImage(currentImage, 0, 0);
  drawDetections(ctx, currentDetections, activeIndex);
}

function renderList(detections, imageDataUrl) {
  const list = byId("detection-list");
  const details = byId("details");
  list.innerHTML = "";
  if (!detections || detections.length === 0) {
    details.textContent = "No detections available.";
    return;
  }
  detections.forEach((det, idx) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <span>${det.label}</span>
      <span>${det.confidence.toFixed(2)}</span>
    `;
    item.addEventListener("click", () => {
      document.querySelectorAll(".list-item").forEach((el) => el.classList.remove("active"));
      item.classList.add("active");
      details.textContent = JSON.stringify(det, null, 2);
      drawScene(idx);
    });
    list.appendChild(item);
  });
  details.textContent = JSON.stringify(detections[0], null, 2);
  list.children[0].classList.add("active");
  drawScene(0);
}

byId("analyze-image").addEventListener("click", async () => {
  const imageFile = byId("image-file").files[0];
  if (!imageFile) return;
  const form = new FormData();
  form.append("image", imageFile);

  try {
    setLoading(true, "Analyzing image...");
    const result = await postForm("/analyze/image", form);
    renderSummary(result);
    renderImage(result.image_data, result.result.detections);
    renderList(result.result.detections, result.image_data);
    renderLiveMap(result);
    renderLiveTrack([]);
    renderFrames([]);
  } catch (err) {
    byId("summary").textContent = `Error: ${err.message || err}`;
  } finally {
    setLoading(false);
  }
});

byId("analyze-video").addEventListener("click", async () => {
  const videoFile = byId("video-file").files[0];
  if (!videoFile) return;
  const interval = parseFloat(byId("interval").value || "2");
  const maxFrames = parseInt(byId("max-frames").value || "12", 10);
  const form = new FormData();
  form.append("video", videoFile);
  form.append("interval_s", interval);
  form.append("max_frames", maxFrames);
  try {
    setLoading(true, "Analyzing video...");
    const result = await postForm("/analyze/video", form);
    if (result.frames.length > 0) {
      renderSummary(result.frames[0]);
      renderImage(result.frames[0].image_data, result.frames[0].result.detections);
      renderList(result.frames[0].result.detections, result.frames[0].image_data);
      renderLiveMap(result.frames[0]);
      renderLiveTrack(result.frames);
      renderFrames(result.frames);
    }
  } catch (err) {
    byId("summary").textContent = `Error: ${err.message || err}`;
  } finally {
    setLoading(false);
  }
});

const canvas = byId("canvas");
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const delta = e.deltaY < 0 ? 1.1 : 0.9;
  scale = Math.max(0.2, Math.min(6, scale * delta));
  drawScene(null);
});

canvas.addEventListener("mousedown", (e) => {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
  canvas.style.cursor = "grabbing";
});

window.addEventListener("mouseup", () => {
  dragging = false;
  canvas.style.cursor = "grab";
});

window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;
  offsetX += dx;
  offsetY += dy;
  drawScene(null);
});

byId("zoom-in").addEventListener("click", () => {
  scale = Math.min(6, scale * 1.2);
  drawScene(null);
});

byId("zoom-out").addEventListener("click", () => {
  scale = Math.max(0.2, scale / 1.2);
  drawScene(null);
});

byId("zoom-reset").addEventListener("click", () => {
  scale = 1;
  offsetX = 0;
  offsetY = 0;
  drawScene(null);
});

async function fetchEvalStatus() {
  try {
    const res = await fetch("/eval/dota/status");
    if (!res.ok) return;
    const data = await res.json();
    byId("eval-status").textContent = `Status: ${data.status}`;
    if (data.last_result) {
      try {
        const parsed = JSON.parse(data.last_result);
        byId("eval-results").textContent = JSON.stringify(parsed, null, 2);
        renderEval(parsed);
      } catch {
        byId("eval-results").textContent = data.last_result;
      }
    }
  } catch {
    // ignore
  }
}

byId("run-eval").addEventListener("click", async () => {
  await fetch("/eval/dota/start", { method: "POST" });
  fetchEvalStatus();
});

setInterval(fetchEvalStatus, 4000);
fetchEvalStatus();

function renderEval(report) {
  const metrics = byId("eval-metrics");
  const classes = byId("eval-classes");
  metrics.innerHTML = "";
  classes.innerHTML = "";
  if (!report) return;

  if (report.overall) {
    const map = report.overall.map ?? "-";
    const map50 = report.overall.map50 ?? "-";
    metrics.innerHTML = `
      <div class="metric-card"><strong>${map}</strong>mAP</div>
      <div class="metric-card"><strong>${map50}</strong>mAP50</div>
    `;
  }

  const perClass = report.per_class || [];
  if (perClass.length > 0) {
    const rows = perClass
      .map(
        (row) => `
      <tr>
        <td>${row.name}</td>
        <td>${row.map ?? "-"}</td>
        <td>${row.p ?? "-"}</td>
        <td>${row.r ?? "-"}</td>
      </tr>`
      )
      .join("");
    classes.innerHTML = `
      <table>
        <thead>
          <tr><th>Class</th><th>mAP</th><th>P</th><th>R</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }
}
