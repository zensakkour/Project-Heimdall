import { byId, formatUtcNowLabel, normalizeError, postForm } from "./shared.js";

const profileStorageKey = "heimdallProfile";
let activeProfile = "paris";

function setSummaryState(state, message) {
  const summary = byId("summary");
  if (!summary) return;
  summary.classList.remove("state-idle", "state-loading", "state-success", "state-error");
  summary.classList.add(`state-${state}`);
  summary.textContent = message;
}

function setMetric(id, value) {
  const el = byId(id);
  if (!el) return;
  el.textContent = value;
}

function setMetricsBaseline() {
  setMetric("metric-location", "-");
  setMetric("metric-tier", "-");
  setMetric("metric-confidence", "-");
  setMetric("metric-radius", "-");
  setMetric("metric-mode", "-");
}

function syncProfileSelect() {
  const profileSelect = byId("profile-select");
  if (!profileSelect) return;
  const stored = localStorage.getItem(profileStorageKey);
  const initial = stored || profileSelect.value || "paris";
  activeProfile = initial;
  profileSelect.value = initial;
  profileSelect.addEventListener("change", () => {
    activeProfile = profileSelect.value || "paris";
    localStorage.setItem(profileStorageKey, activeProfile);
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

function setupFilePicker() {
  const fileInput = byId("image-file");
  const fileName = byId("image-file-name");
  const fileButton = document.querySelector(".file-button");
  const filePicker = fileName ? fileName.closest(".file-picker") : null;
  if (!fileInput || !fileName) return;

  const getName = () => {
    const file = fileInput.files && fileInput.files[0];
    if (file && file.name) return file.name;
    const val = fileInput.value || "";
    const fallback = val.split(/[/\\]/).pop();
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
  if (filePicker) filePicker.addEventListener("click", () => setTimeout(update, 0));
  update();
}

function updateLastRun() {
  const lastUpdated = byId("last-updated");
  if (!lastUpdated) return;
  lastUpdated.textContent = formatUtcNowLabel("Last run: ");
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

  const topCandidate = Array.isArray(fusion?.candidates) ? fusion.candidates[0]?.candidate || null : null;
  const fallbackCandidate = Array.isArray(result.result.candidates)
    ? result.result.candidates[0] || null
    : null;
  const topLat = topCandidate?.latitude ?? fallbackCandidate?.latitude ?? fusion?.mean_latitude;
  const topLon = topCandidate?.longitude ?? fallbackCandidate?.longitude ?? fusion?.mean_longitude;

  const locationText =
    topLat !== undefined && topLon !== undefined
      ? `${Number(topLat).toFixed(5)}, ${Number(topLon).toFixed(5)}`
      : "-";

  const tier = geo?.confidence_tier || tierFromFusion() || tierFromCandidates() || "-";
  const confidenceValue = geoConfidence();
  const confidenceText = confidenceValue !== null ? `${(confidenceValue * 100).toFixed(1)}%` : "-";

  const radiusMeters =
    fusion?.uncertainty_radius_m ?? (fusion?.ellipse?.major_axis_m ? fusion.ellipse.major_axis_m * 0.6 : null);
  const radiusText = radiusMeters !== null && radiusMeters !== undefined
    ? `${Number(radiusMeters).toFixed(1)} m`
    : "-";

  const modeText = result.safe_demo ? "Safe demo" : "Live inference";

  setMetric("metric-location", locationText);
  setMetric("metric-tier", tier);
  setMetric("metric-confidence", confidenceText);
  setMetric("metric-radius", radiusText);
  setMetric("metric-mode", modeText);

  const debugParts = [];
  if (geoDebug && geoDebug.candidate_count !== undefined) {
    debugParts.push(`${geoDebug.candidate_count} geo candidates`);
  }
  if (result.result?.detections) {
    debugParts.push(`${result.result.detections.length} detections`);
  }
  if (geoDebug?.error) {
    debugParts.push(`note: ${geoDebug.error}`);
  }

  const suffix = debugParts.length ? ` (${debugParts.join(" | ")})` : "";
  setSummaryState(
    "success",
    `Most likely location is ${locationText} with ${tier} confidence at ${confidenceText} certainty.${suffix}`
  );

  updateLastRun();
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
  const lightness = 36 + w * 30;
  return `hsl(186, 68%, ${lightness}%)`;
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
  if (!el) return Promise.resolve();

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
        } catch {
          // Ignore unsupported projection.
        }
      }

      if (liveMap.setFog) {
        liveMap.setFog({
          color: "rgb(7, 13, 16)",
          "high-color": "rgb(5, 9, 12)",
          "space-color": "rgb(2, 5, 8)",
          "horizon-blend": 0.11,
          "star-intensity": 0.12,
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
          "line-color": "rgba(130, 210, 230, 0.58)",
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
          "circle-radius": ["interpolate", ["linear"], ["get", "weightNorm"], 0, 3, 1, 10],
          "circle-opacity": ["case", ["==", ["get", "rank"], 1], 0.95, 0.72],
          "circle-stroke-color": "rgba(218, 244, 255, 0.85)",
          "circle-stroke-width": ["case", ["==", ["get", "rank"], 1], 1.6, 0.8],
        },
      });

      liveMap.addLayer({
        id: "mean-layer",
        type: "circle",
        source: "mean",
        paint: {
          "circle-color": "rgba(240, 250, 255, 0.95)",
          "circle-radius": 6,
          "circle-stroke-color": "rgba(180, 232, 255, 0.9)",
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
        const lines = [
          `Candidate #${rank}`,
          `Lat/Lon: ${coords[1].toFixed(5)}, ${coords[0].toFixed(5)}`,
          `Confidence: ${(conf * 100).toFixed(1)}%`,
        ];
        if (posterior !== null && !Number.isNaN(posterior)) {
          lines.push(`Fusion weight: ${(posterior * 100).toFixed(1)}%`);
        }
        if (retr !== null && !Number.isNaN(retr)) {
          lines.push(`Retrieval score: ${(retr * 100).toFixed(1)}%`);
        }

        if (livePopup) livePopup.remove();
        livePopup = new maplibregl.Popup({ offset: 12 })
          .setLngLat(coords)
          .setText(lines.join("\n"))
          .addTo(liveMap);
      });

      liveMap.on("click", "mean-layer", (e) => {
        if (!e.features || !e.features.length) return;
        const coords = e.features[0].geometry.coordinates;
        if (livePopup) livePopup.remove();
        livePopup = new maplibregl.Popup({ offset: 12 })
          .setLngLat(coords)
          .setText(`Fused mean\nLat/Lon: ${coords[1].toFixed(5)}, ${coords[0].toFixed(5)}`)
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
  if (!canvas) return;

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
      ctx.fillStyle = "rgba(107, 197, 255, 0.18)";
      ctx.fill();
      ctx.lineWidth = 4;
      ctx.strokeStyle = "#6bc5ff";
    } else {
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(110, 130, 146, 0.45)";
    }
    ctx.stroke();
  });
}

function drawScene(activeIndex) {
  const canvas = byId("canvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!currentImage) return;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(scale, 0, 0, scale, offsetX, offsetY);
  ctx.drawImage(currentImage, 0, 0);
  drawDetections(ctx, currentDetections, activeIndex);
}

function renderList(detections) {
  const list = byId("detection-list");
  const details = byId("details");
  if (!list || !details) return;

  list.replaceChildren();
  if (!detections || detections.length === 0) {
    const empty = document.createElement("div");
    empty.className = "list-empty muted";
    empty.textContent = "No detections found for this image.";
    list.appendChild(empty);
    details.textContent = "No detections available.";
    return;
  }

  detections.forEach((det, idx) => {
    const item = document.createElement("div");
    item.className = "list-item";

    const conf = Math.max(0, Math.min(1, det.confidence ?? 0));
    const label = document.createElement("span");
    label.textContent = String(det.label ?? "unknown");

    const meta = document.createElement("div");
    meta.className = "list-meta";

    const bar = document.createElement("div");
    bar.className = "confidence-bar";

    const fill = document.createElement("div");
    fill.className = "confidence-fill";
    fill.style.width = `${(conf * 100).toFixed(1)}%`;
    bar.appendChild(fill);

    const confText = document.createElement("span");
    confText.className = "confidence";
    confText.textContent = `${(conf * 100).toFixed(1)}%`;

    meta.appendChild(bar);
    meta.appendChild(confText);
    item.appendChild(label);
    item.appendChild(meta);

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

function setupAnalyzeAction() {
  const analyzeButton = byId("analyze-image");
  const imageInput = byId("image-file");
  if (!analyzeButton || !imageInput) return;

  analyzeButton.addEventListener("click", async () => {
    const imageFile = imageInput.files?.[0];
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
      renderList(result.result.detections);
      renderLiveMap(result);
    } catch (err) {
      setSummaryState("error", `Analysis failed: ${normalizeError(err)}`);
    } finally {
      setLoading(false);
    }
  });
}

function setupCanvasControls() {
  const canvas = byId("canvas");
  if (!canvas) return;

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.88;
    const newScale = Math.max(0.2, Math.min(6, scale * zoomFactor));
    const scaleDelta = newScale / scale;

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

  byId("zoom-in")?.addEventListener("click", () => {
    scale = Math.min(6, scale * 1.15);
    drawScene(null);
  });

  byId("zoom-out")?.addEventListener("click", () => {
    scale = Math.max(0.2, scale * 0.87);
    drawScene(null);
  });

  byId("zoom-reset")?.addEventListener("click", () => {
    scale = 1;
    offsetX = 0;
    offsetY = 0;
    drawScene(null);
  });
}

function setupMapControls() {
  byId("map-zoom-in")?.addEventListener("click", () => {
    ensureLiveMap();
    if (liveMap) liveMap.zoomIn();
  });

  byId("map-zoom-out")?.addEventListener("click", () => {
    ensureLiveMap();
    if (liveMap) liveMap.zoomOut();
  });

  byId("map-zoom-reset")?.addEventListener("click", () => {
    ensureLiveMap();
    if (!liveMap) return;
    liveMap.easeTo({
      center: initialCenter,
      zoom: initialZoom,
      bearing: 0,
      pitch: 0,
      duration: 500,
    });
  });

  const topSelect = byId("live-topn");
  const topLabel = byId("live-topn-label");
  if (!topSelect) return;

  const stored = Number(localStorage.getItem("heimdallTopN") || "20");
  topSelect.value = String(stored);
  liveTopLimit = stored;
  if (topLabel) topLabel.textContent = `Showing top ${liveTopLimit} candidates on map`;

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

  container.replaceChildren();

  const header = document.createElement("div");
  header.className = "geo-rank-item geo-rank-header";

  const headerRank = document.createElement("span");
  headerRank.className = "rank-badge";
  headerRank.textContent = "Rank";

  const headerCoords = document.createElement("span");
  headerCoords.textContent = "Coordinates";

  const headerConf = document.createElement("span");
  headerConf.className = "rank-conf";
  headerConf.textContent = "Confidence";

  header.appendChild(headerRank);
  header.appendChild(headerCoords);
  header.appendChild(headerConf);
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

    const rank = document.createElement("span");
    rank.className = "rank-badge";
    rank.textContent = `#${item.rank}`;

    const coords = document.createElement("span");
    coords.textContent = `${item.latitude.toFixed(4)}, ${item.longitude.toFixed(4)}`;

    const conf = document.createElement("span");
    conf.className = "rank-conf";
    conf.textContent = `${(item.weight * 100).toFixed(1)}%`;

    row.appendChild(rank);
    row.appendChild(coords);
    row.appendChild(conf);

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
    "rgba(255, 110, 110, 0.95)",
    ["get", "color"],
  ]);
  liveMap.setPaintProperty("candidate-layer", "circle-stroke-color", [
    "case",
    ["==", ["get", "rank"], highlight],
    "rgba(255, 164, 164, 1)",
    "rgba(218, 244, 255, 0.85)",
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

function init() {
  syncProfileSelect();
  setupFilePicker();
  setupAnalyzeAction();
  setupCanvasControls();
  setupMapControls();
  setMetricsBaseline();
  setSummaryState("idle", "Upload an image and click Analyze Image to start.");
  renderGeoRanking([]);

  ensureLiveMap();
  if (liveMap) {
    setTimeout(() => liveMap.resize(), 0);
  }
}

window.addEventListener("load", init);
