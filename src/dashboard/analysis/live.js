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

let liveMapView = null;
let trackMapView = null;
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

function renderLiveMap(result) {
  if (!liveMapView) {
    liveMapView = new GlobeView(byId("live-map"));
  }
  if (!result || !result.result) {
    liveMapView.setData({ points: [], mean: null, ringRadiusM: null, track: [] });
    return;
  }
  const fusion = result.result.fusion;
  const candidates = Array.isArray(fusion?.candidates) ? fusion.candidates : [];
  if (!fusion || candidates.length === 0) {
    liveMapView.setData({ points: [], mean: null, ringRadiusM: null, track: [] });
    return;
  }
  const sorted = [...candidates].sort((a, b) => weightFrom(b) - weightFrom(a));
  const maxWeight = sorted.length ? Math.max(...sorted.map(weightFrom)) : 1;
  const visible = sorted.slice(0, liveTopLimit);
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
  liveMapView.setData({
    points,
    mean: meanLat !== undefined && meanLon !== undefined ? { lat: meanLat, lon: meanLon } : null,
    ringRadiusM: ringRadius,
    track: [],
  });
}

function renderLiveTrack(frames) {
  if (!trackMapView) {
    trackMapView = new GlobeView(byId("live-track"));
  }
  const points = (frames || [])
    .map((frame) => {
      const fusion = frame.result?.fusion;
      if (!fusion) return null;
      if (fusion.mean_latitude === undefined || fusion.mean_longitude === undefined) return null;
      return { lat: fusion.mean_latitude, lon: fusion.mean_longitude };
    })
    .filter(Boolean);
  if (points.length < 2) {
    trackMapView.setData({ points: [], mean: null, ringRadiusM: null, track: [] });
    return;
  }
  trackMapView.setData({ points: [], mean: null, ringRadiusM: null, track: points });
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
    lastResult = result;
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
      lastResult = result.frames[0];
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

const topSelect = byId("live-topn");
if (topSelect) {
  const stored = Number(localStorage.getItem("heimdallTopN") || "20");
  topSelect.value = String(stored);
  liveTopLimit = stored;
  topSelect.addEventListener("change", () => {
    liveTopLimit = Number(topSelect.value || "20");
    localStorage.setItem("heimdallTopN", String(liveTopLimit));
    if (lastResult) renderLiveMap(lastResult);
  });
}

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
