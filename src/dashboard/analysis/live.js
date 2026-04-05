function parseServerError(text) {
  if (!text) return "";
  const trimmed = String(text).trim();
  if (!trimmed) return "";
  try {
    const parsed = JSON.parse(trimmed);
    if (typeof parsed === "string") return parsed;
    if (parsed && typeof parsed === "object") {
      if (typeof parsed.detail === "string") return parsed.detail;
      if (Array.isArray(parsed.detail)) {
        const msgs = parsed.detail
          .map((item) => (item && item.msg ? item.msg : ""))
          .filter(Boolean);
        if (msgs.length) return msgs.join("; ");
      }
      if (typeof parsed.error === "string") return parsed.error;
      if (typeof parsed.message === "string") return parsed.message;
    }
  } catch {
    // Keep fallback raw text.
  }
  return trimmed.replace(/\s+/g, " ");
}

function normalizeError(err) {
  if (!err) return "Unexpected error.";
  if (typeof err === "string") return parseServerError(err) || "Unexpected error.";
  if (err instanceof Error) return parseServerError(err.message) || "Unexpected error.";
  return parseServerError(String(err)) || "Unexpected error.";
}

async function postForm(url, formData) {
  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const text = await res.text();
    const message = parseServerError(text) || `Request failed (HTTP ${res.status}).`;
    throw new Error(message);
  }
  return res.json();
}

function byId(id) {
  return document.getElementById(id);
}

function setSummaryState(state, message) {
  const summary = byId("summary");
  if (!summary) return;
  summary.classList.remove("state-idle", "state-loading", "state-success", "state-error");
  summary.classList.add(`state-${state}`);
  summary.textContent = message;
}

const modeButtons = Array.from(document.querySelectorAll(".mode-tab"));
function setActiveTab(tab) {
  modeButtons.forEach((btn) => {
    const active = btn.dataset.tab === tab;
    btn.setAttribute("aria-pressed", active ? "true" : "false");
  });
  const analysis = byId("tab-analysis");
  const scoring = byId("tab-scoring");
  if (analysis) analysis.classList.toggle("tab-hidden", tab !== "analysis");
  if (scoring) scoring.classList.toggle("tab-hidden", tab !== "scoring");
}
if (modeButtons.length) {
  modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
  });
}

const profileButtons = Array.from(document.querySelectorAll(".profile-tab"));
const profileSelect = byId("profile-select");
const profileStorageKey = "heimdallProfile";
let activeProfile = "paris";
const strategySelect = byId("geo-eval-strategy");

function setActiveProfile(profile) {
  if (!profile) return;
  activeProfile = profile;
  localStorage.setItem(profileStorageKey, profile);
  if (profileSelect) {
    profileSelect.value = profile;
  }
  if (strategySelect) {
    strategySelect.value = profile;
  }
  profileButtons.forEach((btn) => {
    const isActive = btn.dataset.profile === profile;
    btn.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
}

if (profileButtons.length) {
  const storedProfile = localStorage.getItem(profileStorageKey);
  const available = profileButtons.map((btn) => btn.dataset.profile);
  const initialProfile = available.includes(storedProfile) ? storedProfile : available[0];
  setActiveProfile(initialProfile);
  profileButtons.forEach((btn) => {
    btn.addEventListener("click", () => setActiveProfile(btn.dataset.profile));
  });
}

if (profileSelect) {
  const storedProfile = localStorage.getItem(profileStorageKey);
  const initialProfile = storedProfile || profileSelect.value || "paris";
  setActiveProfile(initialProfile);
  profileSelect.addEventListener("change", () => setActiveProfile(profileSelect.value));
}

if (strategySelect) {
  strategySelect.value = activeProfile;
  strategySelect.addEventListener("change", () => {
    setActiveProfile(strategySelect.value);
  });
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
  if (active) {
    setSummaryState("loading", text || "Analyzing image...");
  }
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
    if (fileButton) fileButton.textContent = "Choose file";
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
  const modeText = result.safe_demo ? " | Mode: safe demo" : "";
  setSummaryState(
    "success",
    `Score: ${result.result.score.toFixed(3)} | Geo tier: ${
    geo?.confidence_tier || tierFromFusion() || tierFromCandidates() || "-"
  }${geoConfText} | Detections: ${result.result.detections.length}${fusionText}${debugText}${modeText}`
  );
}

let currentImage = null;
let currentDetections = [];
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let dragging = false;
let lastX = 0;
let lastY = 0;

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
  const lightness = 38 + w * 32;
  return `hsl(171, 74%, ${lightness}%)`;
}

const emptyFeatureCollection = { type: "FeatureCollection", features: [] };
const initialCenter = [0, 20];
const initialZoom = 1.6;

let liveMap = null;
let liveMapReady = null;
let livePopup = null;
let liveCandidatePoints = [];
let selectedRank = null;
let hoverPopup = null;

function ensureLiveMap() {
  if (liveMap) return liveMapReady;
  const el = byId("live-map");
  liveMap = new maplibregl.Map({
    container: el,
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    center: initialCenter,
    zoom: initialZoom,
    attributionControl: false,
  });

  liveMap.addControl(new maplibregl.NavigationControl({ showCompass: true }), "bottom-right");
  liveMap.addControl(new maplibregl.ScaleControl({ maxWidth: 90, unit: "metric" }), "bottom-left");

  liveMapReady = new Promise((resolve) => {
    liveMap.on("load", () => {
      if (liveMap.setProjection) {
        try {
          liveMap.setProjection({ type: "globe" });
        } catch (err) {
          // Ignore unsupported projection in older maplibre builds.
        }
      }
      if (liveMap.setFog) {
        liveMap.setFog({
          color: "rgb(8, 15, 18)",
          "high-color": "rgb(5, 10, 12)",
          "space-color": "rgb(3, 6, 8)",
          "horizon-blend": 0.12,
          "star-intensity": 0.15,
        });
      }

      liveMap.addSource("candidates", {
        type: "geojson",
        data: emptyFeatureCollection,
      });
      liveMap.addSource("mean", {
        type: "geojson",
        data: emptyFeatureCollection,
      });
      liveMap.addSource("ring", {
        type: "geojson",
        data: emptyFeatureCollection,
      });

      liveMap.addLayer({
        id: "ring-layer",
        type: "line",
        source: "ring",
        paint: {
          "line-color": "rgba(120, 246, 215, 0.6)",
          "line-width": 1.4,
          "line-dasharray": [2, 2],
        },
      });

      liveMap.addLayer({
        id: "candidate-layer",
        type: "circle",
        source: "candidates",
        paint: {
          "circle-color": ["get", "color"],
          "circle-radius": [
            "interpolate",
            ["linear"],
            ["get", "weightNorm"],
            0,
            3,
            1,
            10,
          ],
          "circle-opacity": [
            "case",
            ["==", ["get", "rank"], 1],
            0.95,
            0.7,
          ],
          "circle-stroke-color": "rgba(196, 255, 236, 0.85)",
          "circle-stroke-width": [
            "case",
            ["==", ["get", "rank"], 1],
            1.6,
            0.8,
          ],
        },
      });

      liveMap.addLayer({
        id: "mean-layer",
        type: "circle",
        source: "mean",
        paint: {
          "circle-color": "rgba(230, 255, 245, 0.95)",
          "circle-radius": 6,
          "circle-stroke-color": "rgba(180, 255, 236, 0.9)",
          "circle-stroke-width": 1.6,
        },
      });

      liveMap.on("mouseenter", "candidate-layer", () => {
        liveMap.getCanvas().style.cursor = "pointer";
      });
      liveMap.on("mouseleave", "candidate-layer", () => {
        liveMap.getCanvas().style.cursor = "";
        if (hoverPopup) {
          hoverPopup.remove();
          hoverPopup = null;
        }
      });
      liveMap.on("mouseenter", "mean-layer", () => {
        liveMap.getCanvas().style.cursor = "pointer";
      });
      liveMap.on("mouseleave", "mean-layer", () => {
        liveMap.getCanvas().style.cursor = "";
      });

      liveMap.on("click", "candidate-layer", (e) => {
        if (!e.features || !e.features.length) return;
        const feature = e.features[0];
        const props = feature.properties || {};
        const coords = feature.geometry.coordinates;
        const rank = Number(props.rank || 0);
        if (rank) setSelectedRank(rank);
        const conf = Number(props.rawWeight || 0);
        const posterior = props.posterior !== undefined ? Number(props.posterior) : null;
        const retr = props.retrieval !== undefined ? Number(props.retrieval) : null;
        const html = [
          `<strong>Candidate #${rank}</strong>`,
          `Lat/Lon: ${coords[1].toFixed(5)}, ${coords[0].toFixed(5)}`,
          `Confidence: ${(conf * 100).toFixed(1)}%`,
        ];
        if (posterior !== null && !Number.isNaN(posterior)) {
          html.push(`Fusion weight: ${(posterior * 100).toFixed(1)}%`);
        }
        if (retr !== null && !Number.isNaN(retr)) {
          html.push(`Retrieval score: ${(retr * 100).toFixed(1)}%`);
        }
        if (livePopup) livePopup.remove();
        livePopup = new maplibregl.Popup({ offset: 12 })
          .setLngLat(coords)
          .setHTML(html.join("<br/>"))
          .addTo(liveMap);
      });

      liveMap.on("click", "mean-layer", (e) => {
        if (!e.features || !e.features.length) return;
        const coords = e.features[0].geometry.coordinates;
        if (livePopup) livePopup.remove();
        livePopup = new maplibregl.Popup({ offset: 12 })
          .setLngLat(coords)
          .setHTML(`Fused mean<br/>Lat/Lon: ${coords[1].toFixed(5)}, ${coords[0].toFixed(5)}`)
          .addTo(liveMap);
      });

      resolve();
    });
  });

  return liveMapReady;
}

function circlePolygon(lat, lon, radiusMeters, points = 64) {
  const coords = [];
  const rad = radiusMeters / 6371000;
  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;
  for (let i = 0; i <= points; i += 1) {
    const bearing = (2 * Math.PI * i) / points;
    const lat2 = Math.asin(
      Math.sin(latRad) * Math.cos(rad) +
        Math.cos(latRad) * Math.sin(rad) * Math.cos(bearing)
    );
    const lon2 =
      lonRad +
      Math.atan2(
        Math.sin(bearing) * Math.sin(rad) * Math.cos(latRad),
        Math.cos(rad) - Math.sin(latRad) * Math.sin(lat2)
      );
    coords.push([(lon2 * 180) / Math.PI, (lat2 * 180) / Math.PI]);
  }
  return coords;
}

function renderLiveMap(result) {
  ensureLiveMap();
  if (!result || !result.result) {
    if (liveMap && liveMap.getSource("candidates")) {
      liveMap.getSource("candidates").setData(emptyFeatureCollection);
      liveMap.getSource("mean").setData(emptyFeatureCollection);
      liveMap.getSource("ring").setData(emptyFeatureCollection);
    }
    selectedRank = null;
    renderGeoRanking([]);
    return;
  }

  const fusion = result.result.fusion;
  const candidates = Array.isArray(fusion?.candidates) ? fusion.candidates : [];
  if (!fusion || candidates.length === 0) {
    if (liveMap && liveMap.getSource("candidates")) {
      liveMap.getSource("candidates").setData(emptyFeatureCollection);
      liveMap.getSource("mean").setData(emptyFeatureCollection);
      liveMap.getSource("ring").setData(emptyFeatureCollection);
    }
    selectedRank = null;
    renderGeoRanking([]);
    return;
  }

  const sorted = [...candidates].sort((a, b) => weightFrom(b) - weightFrom(a));
  const maxWeight = sorted.length ? Math.max(...sorted.map(weightFrom)) : 1;
  const visible = sorted.slice(0, liveTopLimit);

  const features = visible
    .map((item, idx) => {
      const cand = item.candidate || {};
      if (cand.latitude === undefined || cand.longitude === undefined) return null;
      const rawWeight = weightFrom(item);
      const weightNorm = maxWeight > 0 ? rawWeight / maxWeight : 0;
      return {
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: [cand.longitude, cand.latitude],
        },
        properties: {
          rank: idx + 1,
          rawWeight,
          weightNorm,
          color: weightColor(weightNorm),
          posterior: item.posterior_weight ?? null,
          retrieval: item.candidate?.retrieval_score ?? item.retrieval_score ?? null,
        },
      };
    })
    .filter(Boolean);

  liveCandidatePoints = visible
    .map((item, idx) => {
      const cand = item.candidate || {};
      if (cand.latitude === undefined || cand.longitude === undefined) return null;
      return {
        rank: idx + 1,
        latitude: cand.latitude,
        longitude: cand.longitude,
        weight: weightFrom(item),
      };
    })
    .filter(Boolean);

  renderGeoRanking(liveCandidatePoints);

  const meanLat = fusion.mean_latitude;
  const meanLon = fusion.mean_longitude;
  const ringRadius =
    fusion?.uncertainty_radius_m ??
    (fusion?.ellipse?.major_axis_m ? fusion.ellipse.major_axis_m * 0.6 : null);

  const meanFeature =
    meanLat !== undefined && meanLon !== undefined
      ? {
          type: "Feature",
          geometry: { type: "Point", coordinates: [meanLon, meanLat] },
          properties: {},
        }
      : null;

  const ringFeature =
    meanLat !== undefined && meanLon !== undefined && ringRadius
      ? {
          type: "Feature",
          geometry: {
            type: "Polygon",
            coordinates: [circlePolygon(meanLat, meanLon, ringRadius)],
          },
          properties: {},
        }
      : null;

  ensureLiveMap();
  liveMapReady.then(() => {
    liveMap.getSource("candidates").setData({
      type: "FeatureCollection",
      features,
    });
    liveMap.getSource("mean").setData({
      type: "FeatureCollection",
      features: meanFeature ? [meanFeature] : [],
    });
    liveMap.getSource("ring").setData({
      type: "FeatureCollection",
      features: ringFeature ? [ringFeature] : [],
    });

    if (meanLat !== undefined && meanLon !== undefined) {
      liveMap.easeTo({
        center: [meanLon, meanLat],
        zoom: 2.6,
        duration: 700,
        bearing: 0,
        pitch: 0,
      });
    }

    applyCandidateHighlight();
  });
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
    list.innerHTML = `<div class="list-empty muted">No detections found for this image.</div>`;
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
  if (!imageFile) {
    setSummaryState("error", "Choose an image file before running analysis.");
    return;
  }
  const form = new FormData();
  form.append("image", imageFile);

  try {
    setLoading(true, "Analyzing image...");
    const profileQuery = activeProfile ? `?profile=${encodeURIComponent(activeProfile)}` : "";
    const result = await postForm(`/analyze/image${profileQuery}`, form);
    lastResult = result;
    renderSummary(result);
    renderImage(result.image_data, result.result.detections);
    renderList(result.result.detections, result.image_data);
    renderLiveMap(result);
  } catch (err) {
    setSummaryState("error", `Analysis failed: ${normalizeError(err)}`);
  } finally {
    setLoading(false);
  }
});

const canvas = byId("canvas");
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  const zoomFactor = e.deltaY < 0 ? 1.12 : 0.88;
  const newScale = Math.max(0.2, Math.min(6, scale * zoomFactor));
  const scaleDelta = newScale / scale;

  // Zoom towards cursor by adjusting offsets in screen space.
  offsetX = mouseX - (mouseX - offsetX) * scaleDelta;
  offsetY = mouseY - (mouseY - offsetY) * scaleDelta;
  scale = newScale;
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
  if (liveMap) liveMap.zoomIn();
});

byId("map-zoom-out").addEventListener("click", () => {
  ensureLiveMap();
  if (liveMap) liveMap.zoomOut();
});

byId("map-zoom-reset").addEventListener("click", () => {
  ensureLiveMap();
  if (liveMap) {
    liveMap.easeTo({
      center: initialCenter,
      zoom: initialZoom,
      bearing: 0,
      pitch: 0,
      duration: 500,
    });
  }
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

function renderGeoRanking(items) {
  const container = byId("geo-ranking");
  if (!container) return;
  container.innerHTML = "";
  const header = document.createElement("div");
  header.className = "geo-rank-item geo-rank-header";
  header.innerHTML = `
    <span class="rank-badge">Rank</span>
    <span>Coordinates</span>
    <span class="rank-conf">Confidence</span>
  `;
  container.appendChild(header);
  if (!items || items.length === 0) {
    container.classList.add("muted");
    const empty = document.createElement("div");
    empty.className = "geo-empty";
    empty.textContent = "No geo ranking yet. Run analysis to populate candidates.";
    container.appendChild(empty);
    return;
  }
  container.classList.remove("muted");
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "geo-rank-item";
    row.dataset.rank = String(item.rank);
    row.innerHTML = `
      <span class="rank-badge">#${item.rank}</span>
      <span>${item.latitude.toFixed(4)}, ${item.longitude.toFixed(4)}</span>
      <span class="rank-conf">${(item.weight * 100).toFixed(1)}%</span>
    `;
    row.addEventListener("click", () => {
      setSelectedRank(item.rank);
      ensureLiveMap();
      if (liveMap) {
        liveMap.easeTo({
          center: [item.longitude, item.latitude],
          zoom: 3.2,
          duration: 650,
        });
      }
    });
    container.appendChild(row);
  });
}

function applyCandidateHighlight() {
  if (!liveMap) return;
  const highlight = selectedRank ?? -1;
  liveMap.setPaintProperty("candidate-layer", "circle-color", [
    "case",
    ["==", ["get", "rank"], highlight],
    "rgba(255, 90, 90, 0.95)",
    ["get", "color"],
  ]);
  liveMap.setPaintProperty("candidate-layer", "circle-stroke-color", [
    "case",
    ["==", ["get", "rank"], highlight],
    "rgba(255, 140, 140, 1)",
    "rgba(196, 255, 236, 0.85)",
  ]);
  liveMap.setPaintProperty("candidate-layer", "circle-stroke-width", [
    "case",
    ["==", ["get", "rank"], highlight],
    2.4,
    ["case", ["==", ["get", "rank"], 1], 1.6, 0.8],
  ]);
}

function setSelectedRank(rank) {
  selectedRank = rank;
  document.querySelectorAll(".geo-rank-item").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.rank) === rank);
  });
  applyCandidateHighlight();
}

// Initialize the globe immediately so it is visible before analysis runs.
window.addEventListener("load", () => {
  ensureLiveMap();
  if (liveMap) {
    setTimeout(() => liveMap.resize(), 0);
  }
  setSummaryState("idle", "Upload an image and click Analyze Image to start.");
  renderGeoRanking([]);
  setActiveTab("analysis");
  const retrievalToggle = byId("geo-eval-retrieval-only");
  if (retrievalToggle) {
    const stored = localStorage.getItem("heimdallRetrievalOnly");
    if (stored === "0") {
      retrievalToggle.checked = false;
    } else {
      retrievalToggle.checked = true;
    }
    retrievalToggle.addEventListener("change", () => {
      localStorage.setItem(
        "heimdallRetrievalOnly",
        retrievalToggle.checked ? "1" : "0"
      );
    });
  }
});

async function startGeoEval() {
  const imagesDir = byId("geo-eval-images")?.value?.trim() || "";
  const metadata = byId("geo-eval-metadata")?.value?.trim() || "";
  const limit = Number(byId("geo-eval-limit")?.value || "0");
  const retrievalToggle = byId("geo-eval-retrieval-only");
  const retrievalOnly = retrievalToggle ? Boolean(retrievalToggle.checked) : true;
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

const geoEvalBtn = byId("geo-eval-run");
if (geoEvalBtn) {
  geoEvalBtn.addEventListener("click", startGeoEval);
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
  const mode = useSavedBenchmarkView() ? "selected saved run" : "latest run";
  if (!metaEl) return;
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
        // keep default message
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
  const res = await fetch("/eval/benchmarks/status");
  if (!res.ok) return;
  const data = await res.json();
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
  if (status === "done" && data.last_result) {
    try {
      const parsed = JSON.parse(data.last_result);
      await refreshBenchmarkRuns(data.run_id || parsed.run_id || null);
      if (!useSavedBenchmarkView()) {
        renderBenchmarkSummary(parsed);
      }
    } catch {
      // Keep raw output visible.
    }
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
  if (progressWrap) progressWrap.classList.add("active");
  if (progressBar) progressBar.style.width = "0%";
  if (progressText) progressText.textContent = "0% - Preparing benchmark jobs";
  await fetch(`/eval/benchmarks/start?${params.toString()}`, { method: "POST" });
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
    await loadBenchmarkSummaryFromFile();
  });
}

async function loadBenchmarkSummaryFromFile(options = {}) {
  const statusEl = byId("bench-status");
  const outputEl = byId("bench-output");
  const silent = Boolean(options?.silent);
  try {
    const res = await fetch("/data/benchmark_compare.json", { cache: "no-store" });
    if (!res.ok) return;
    const payload = await res.json();
    renderBenchmarkSummary(payload);
    if (outputEl) outputEl.textContent = JSON.stringify(payload, null, 2);
    if (!silent && statusEl) {
      statusEl.textContent = `Loaded latest run: ${payload?.generated_at || "-"}`;
    }
  } catch {
    // No cached benchmark summary yet.
  }
}

async function initBenchmarkHistory() {
  const runs = await refreshBenchmarkRuns();
  if (runs.length && useSavedBenchmarkView()) {
    await loadSelectedBenchmarkRun({ silent: true, force: true });
    return;
  }
  await loadBenchmarkSummaryFromFile({ silent: true });
}

initBenchmarkHistory();
