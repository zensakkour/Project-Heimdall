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
  if (progress) {
    progress.classList.toggle("active", active);
    if (text && progressText) progressText.textContent = text;
  }
  if (analyzeImage) analyzeImage.disabled = active;
}

const fileInput = byId("image-file");
const fileName = byId("image-file-name");
const fileButton = document.querySelector(".file-button");
const filePicker = fileName ? fileName.closest(".file-picker") : null;
if (fileInput && fileName) {
  const getName = () => {
    const file = fileInput.files && fileInput.files[0];
    if (file && file.name) return file.name;
    const val = fileInput.value || "";
    const fallback = val.split(/[/\\\\]/).pop();
    return fallback || "No file selected";
  };
  const update = () => {
    const name = getName();
    fileName.textContent = name;
    const hasFile = name !== "No file selected";
    if (filePicker) filePicker.classList.toggle("has-file", hasFile);
    if (fileButton) fileButton.textContent = hasFile ? name : "Choose file";
  };
  const scheduleUpdate = () => requestAnimationFrame(update);
  fileInput.addEventListener("change", scheduleUpdate);
  fileInput.addEventListener("input", scheduleUpdate);
  if (filePicker) {
    filePicker.addEventListener("click", () => setTimeout(update, 0));
  }
  update();
}

function renderSummary(result) {
  const summary = byId("summary");
  const geo = result.result.geo;
  const fusion = result.result.fusion;
  const geoDebug = result.geo_debug;
  const geoConfidence = () => {
    if (fusion && Array.isArray(fusion.candidates) && fusion.candidates.length > 0) {
      return fusion.candidates[0].posterior_weight ?? null;
    }
    const candidates = result.result.candidates || [];
    if (Array.isArray(candidates) && candidates.length > 0) {
      return candidates[0].retrieval_score ?? null;
    }
    return null;
  };
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
  const geoConf = geoConfidence();
  const geoConfText = geoConf !== null ? ` | Geo conf: ${(geoConf * 100).toFixed(1)}%` : "";
  const debugText = geoDebug
    ? ` | Geo candidates: ${geoDebug.candidate_count ?? 0}${geoDebug.error ? ` (${geoDebug.error})` : ""}`
    : "";
  summary.textContent = `Score: ${result.result.score.toFixed(3)} | Geo tier: ${
    geo?.confidence_tier || tierFromFusion() || tierFromCandidates() || "-"
  }${geoConfText} | Detections: ${result.result.detections.length}${fusionText}${debugText}`;
}

let currentImage = null;
let currentDetections = [];
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let dragging = false;
let lastX = 0;
let lastY = 0;

let liveMapView = null;
let liveTopLimit = 20;
let lastResult = null;

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


let liveLeaflet = null;
let liveLayerGroup = null;
let liveCircle = null;
let liveMeanMarker = null;

function ensureLiveMap() {
  if (liveLeaflet) return;
  const el = byId("live-map");
  liveLeaflet = L.map(el, {
    zoomControl: false,
    attributionControl: false,
    worldCopyJump: true,
  }).setView([20, 0], 2);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 7,
  }).addTo(liveLeaflet);

  liveLayerGroup = L.layerGroup().addTo(liveLeaflet);
  setTimeout(() => liveLeaflet.invalidateSize(), 0);
}

function renderLiveMap(result) {
  ensureLiveMap();
  if (!result || !result.result) {
    liveLayerGroup.clearLayers();
    if (liveCircle) liveLeaflet.removeLayer(liveCircle);
    if (liveMeanMarker) liveLeaflet.removeLayer(liveMeanMarker);
    return;
  }
  const fusion = result.result.fusion;
  const candidates = Array.isArray(fusion?.candidates) ? fusion.candidates : [];
  liveLayerGroup.clearLayers();
  if (liveCircle) liveLeaflet.removeLayer(liveCircle);
  if (liveMeanMarker) liveLeaflet.removeLayer(liveMeanMarker);
  if (!fusion || candidates.length === 0) return;

  const sorted = [...candidates].sort((a, b) => weightFrom(b) - weightFrom(a));
  const maxWeight = sorted.length ? Math.max(...sorted.map(weightFrom)) : 1;
  const visible = sorted.slice(0, liveTopLimit);

  visible.forEach((item, idx) => {
    const cand = item.candidate || {};
    if (cand.latitude === undefined || cand.longitude === undefined) return;
    const rawWeight = weightFrom(item);
    const weight = maxWeight > 0 ? rawWeight / maxWeight : 0;
    const color = weightColor(weight);
    const radius = idx === 0 ? 9 : 4 + weight * 7;
    const marker = L.circleMarker([cand.latitude, cand.longitude], {
      radius,
      color,
      fillColor: color,
      weight: 2,
      fillOpacity: idx === 0 ? 0.9 : 0.6,
    }).addTo(liveLayerGroup);
    const rank = idx + 1;
    const conf = rawWeight ?? 0;
    const retr = item.candidate?.retrieval_score ?? item.retrieval_score ?? null;
    const posterior = item.posterior_weight ?? null;
    const lines = [
      `<strong>Candidate #${rank}</strong>`,
      `Lat/Lon: ${cand.latitude.toFixed(5)}, ${cand.longitude.toFixed(5)}`,
      `Confidence: ${(conf * 100).toFixed(1)}%`,
    ];
    if (posterior !== null) lines.push(`Fusion weight: ${(posterior * 100).toFixed(1)}%`);
    if (retr !== null) lines.push(`Retrieval score: ${(retr * 100).toFixed(1)}%`);
    marker.bindTooltip(lines.join("<br/>"), { direction: "top", opacity: 0.95 });
    marker.bindPopup(lines.join("<br/>"));
  });

  const meanLat = fusion.mean_latitude;
  const meanLon = fusion.mean_longitude;
  const ringRadius =
    fusion?.uncertainty_radius_m ??
    (fusion?.ellipse?.major_axis_m ? fusion.ellipse.major_axis_m * 0.6 : null);

  if (meanLat !== undefined && meanLon !== undefined) {
    liveMeanMarker = L.circleMarker([meanLat, meanLon], {
      radius: 7,
      color: "#d7f2de",
      weight: 2,
      fillOpacity: 0.9,
    }).addTo(liveLeaflet);
    liveMeanMarker.bindTooltip(
      `Fused mean<br/>Lat/Lon: ${meanLat.toFixed(5)}, ${meanLon.toFixed(5)}`,
      { direction: "top", opacity: 0.95 }
    );

    if (ringRadius) {
      liveCircle = L.circle([meanLat, meanLon], {
        radius: ringRadius,
        color: "#7fb88a",
        weight: 1.5,
        fillOpacity: 0.05,
        dashArray: "6 6",
        interactive: false,
      }).addTo(liveLeaflet);
      liveCircle.bindTooltip(
        `Uncertainty radius: ${ringRadius.toFixed(0)} m`,
        { direction: "top", opacity: 0.95 }
      );
    }

    liveLeaflet.setView([meanLat, meanLon], 3, { animate: false });
  }
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
    const conf = Math.max(0, Math.min(1, det.confidence ?? 0));
    item.innerHTML = `
      <span>${det.label}</span>
      <div class="list-meta">
        <div class="confidence-bar"><div class="confidence-fill" style="width:${(conf * 100).toFixed(1)}%"></div></div>
        <span class="confidence">${(conf * 100).toFixed(1)}%</span>
      </div>
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
    lastResult = result;
    renderSummary(result);
    renderImage(result.image_data, result.result.detections);
    renderList(result.result.detections, result.image_data);
    renderLiveMap(result);
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
  scale = Math.min(6, scale * 1.15);
  drawScene(null);
});

byId("zoom-out").addEventListener("click", () => {
  scale = Math.max(0.2, scale * 0.87);
  drawScene(null);
});

byId("zoom-reset").addEventListener("click", () => {
  scale = 1;
  offsetX = 0;
  offsetY = 0;
  drawScene(null);
});

byId("map-zoom-in").addEventListener("click", () => {
  ensureLiveMap();
  liveLeaflet.zoomIn();
});

byId("map-zoom-out").addEventListener("click", () => {
  ensureLiveMap();
  liveLeaflet.zoomOut();
});

byId("map-zoom-reset").addEventListener("click", () => {
  ensureLiveMap();
  liveLeaflet.setView([20, 0], 2, { animate: false });
});

const topSelect = byId("live-topn");
const topLabel = byId("live-topn-label");
if (topSelect) {
  const stored = Number(localStorage.getItem("heimdallTopN") || "20");
  topSelect.value = String(stored);
  liveTopLimit = stored;
  if (topLabel) {
    topLabel.textContent = `Showing top ${liveTopLimit} candidates on map`;
  }
  topSelect.addEventListener("change", () => {
    liveTopLimit = Number(topSelect.value || "20");
    localStorage.setItem("heimdallTopN", String(liveTopLimit));
    if (topLabel) {
      topLabel.textContent = `Showing top ${liveTopLimit} candidates on map`;
    }
    if (lastResult) renderLiveMap(lastResult);
  });
}

