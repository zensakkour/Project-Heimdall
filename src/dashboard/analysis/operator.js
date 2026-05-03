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
  const validValues = new Set(Array.from(profileSelect.options).map((option) => option.value));
  const initial = stored && stored !== "legacy" && validValues.has(stored) ? stored : "paris";
  activeProfile = initial;
  profileSelect.value = initial;
  localStorage.setItem(profileStorageKey, initial);
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

function renderRavenResults(result) {
  const container = byId("results-list");
  if (!container) return;
  container.replaceChildren();

  const fusion = result?.result?.fusion;
  const candidates = Array.isArray(fusion?.candidates) ? fusion.candidates : [];

  if (candidates.length === 0) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.style.textAlign = "center";
    empty.style.marginTop = "40px";
    empty.textContent = "No candidates found for this image.";
    container.appendChild(empty);
    return;
  }

  const sorted = [...candidates].sort((a, b) => weightFrom(b) - weightFrom(a));

  sorted.slice(0, 20).forEach((item, idx) => {
    const cand = item.candidate || {};
    const rank = idx + 1;
    const card = document.createElement("div");
    card.className = `candidate-card ${rank === 1 ? "active" : ""}`;
    card.dataset.rank = String(rank);

    const rankEl = document.createElement("div");
    rankEl.className = "card-rank";
    rankEl.textContent = String(rank);

    const bodyEl = document.createElement("div");
    bodyEl.className = "card-body";

    const titleEl = document.createElement("div");
    titleEl.className = "card-address";
    titleEl.textContent = `Target Point ${rank}`;

    const locEl = document.createElement("div");
    locEl.className = "card-location";
    locEl.textContent = `Paris, France`;

    const confLabel = document.createElement("span");
    confLabel.className = "card-confidence";
    confLabel.textContent = rank === 1 ? "Top match" : `Rank #${rank}`;

    const metaEl = document.createElement("div");
    metaEl.className = "card-meta";
    metaEl.textContent = `${Number(cand.latitude).toFixed(5)}, ${Number(cand.longitude).toFixed(5)}`;

    const inspectBtn = document.createElement("button");
    inspectBtn.className = "btn-inspect";
    inspectBtn.textContent = "Inspect on map";

    bodyEl.append(confLabel, titleEl, locEl, metaEl, inspectBtn);

    const statusEl = document.createElement("div");
    statusEl.className = "status-chip";
    statusEl.textContent = rank === 1 ? "Confirmed" : "Unreviewed";

    card.append(rankEl, bodyEl, statusEl);

    card.addEventListener("click", () => {
      document.querySelectorAll(".candidate-card").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      setSelectedRank(rank);
      if (liveMap) {
        liveMap.flyTo({
          center: [cand.longitude, cand.latitude],
          zoom: Math.max(liveMap.getZoom(), 12),
          duration: 1000
        });
      }
    });

    container.appendChild(card);
  });
}

function renderSummary(result) {
  const geo = result.result.geo;
  const fusion = result.result.fusion;
  const geoDebug = result.geo_debug;

  // Raven Title/Thumb Updates
  const thumb = byId("source-thumb");
  if (thumb) {
    thumb.src = result.image_data || "";
    thumb.style.display = result.image_data ? "block" : "none";
  }
  const filename = byId("panel-filename");
  if (filename) filename.textContent = result.filename || "Untitled Analysis";

  // Diagnostics Update
  const diagConf = byId("diag-conf");
  if (diagConf) diagConf.textContent = result.result?.geo?.confidence_tier || "-";
  const diagRadius = byId("diag-radius");
  if (diagRadius) diagRadius.textContent = fusion?.uncertainty_radius_m ? `${fusion.uncertainty_radius_m.toFixed(1)}m` : "-";
  const rawJson = byId("raw-json");
  if (rawJson) rawJson.textContent = JSON.stringify(result, null, 2);
  const diagBackend = byId("diag-backend");
  if (diagBackend) diagBackend.textContent = result.result?.backend || "-";
  const diagWorker = byId("diag-worker");
  if (diagWorker) diagWorker.textContent = result.runtime?.worker_mode || "-";

  renderRavenResults(result);

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

  // Update Diagnostics Drawer
  const rawJsonEl = byId("raw-json");
  if (rawJsonEl) {
    rawJsonEl.textContent = JSON.stringify(result, null, 2);
  }

  const diagBackend = byId("diag-backend");
  if (diagBackend) {
    diagBackend.textContent = result.result?.backend || (result.safe_demo ? "demo" : "-");
  }
  const diagWorker = byId("diag-worker");
  if (diagWorker) {
    diagWorker.textContent = result.runtime?.worker_mode || "-";
  }

  const diagTierBadge = byId("diag-tier-badge");
  if (diagTierBadge) {
    diagTierBadge.textContent = tier.toUpperCase();
    diagTierBadge.className = "diag-value badge " + (
      tier === "high" ? "badge-success" : 
      tier === "medium" ? "badge-warning" : 
      "badge-danger"
    );
  }

  const diagReason = byId("diag-reason");
  if (diagReason) {
    diagReason.textContent = result.fallback_reason || "Nominal";
    diagReason.style.color = result.fallback_reason ? "var(--danger)" : "var(--text-soft)";
  }

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

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatMeters(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const meters = Number(value);
  if (meters >= 1000) return `${(meters / 1000).toFixed(2)} km`;
  return `${meters.toFixed(0)} m`;
}

function sourceFromMatchId(matchId) {
  if (!matchId) return "retrieval";
  const text = String(matchId);
  if (text.includes(":")) return text.split(":")[0];
  return text.includes("/") ? text.split("/")[0] : "retrieval";
}

function setInspectorText(id, value) {
  const el = byId(id);
  if (el) el.textContent = value;
}

function renderCandidateInspector(candidate) {
  const title = byId("geo-inspector-title");
  const inspector = byId("geo-inspector");
  if (!title || !inspector) return;
  if (!candidate) {
    inspector.classList.add("muted");
    title.textContent = "Select a candidate point or row to inspect it.";
    setInspectorText("geo-inspector-rank", "-");
    setInspectorText("geo-inspector-coords", "-");
    setInspectorText("geo-inspector-posterior", "-");
    setInspectorText("geo-inspector-retrieval", "-");
    setInspectorText("geo-inspector-interval", "-");
    setInspectorText("geo-inspector-source", "-");
    return;
  }
  inspector.classList.remove("muted");
  title.textContent = `Candidate #${candidate.rank} locked for verification`;
  setInspectorText("geo-inspector-rank", `#${candidate.rank}`);
  setInspectorText(
    "geo-inspector-coords",
    `${Number(candidate.latitude).toFixed(5)}, ${Number(candidate.longitude).toFixed(5)}`
  );
  setInspectorText("geo-inspector-posterior", formatPercent(candidate.posterior));
  setInspectorText("geo-inspector-retrieval", formatPercent(candidate.retrieval));
  setInspectorText("geo-inspector-interval", formatMeters(liveFusionSummary?.ringRadius));
  setInspectorText("geo-inspector-source", candidate.source || "retrieval");
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
let liveFusionSummary = null;
let radarAnimationId = null;

function startRadarPulse() {
  if (radarAnimationId || window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches) return;
  const tick = () => {
    radarAnimationId = requestAnimationFrame(tick);
    if (!liveMap || !liveMap.getLayer("candidate-glow-layer")) return;
    const phase = (performance.now() % 1800) / 1800;
    const pulse = 0.5 + 0.5 * Math.sin(phase * Math.PI * 2);
    liveMap.setPaintProperty("candidate-glow-layer", "circle-radius", [
      "+",
      ["interpolate", ["linear"], ["get", "weightNorm"], 0, 9, 1, 20],
      5 + pulse * 10,
    ]);
    liveMap.setPaintProperty("candidate-glow-layer", "circle-opacity", [
      "case",
      ["==", ["get", "rank"], selectedRank ?? -1],
      0.28 + pulse * 0.28,
      ["interpolate", ["linear"], ["get", "weightNorm"], 0, 0.06 + pulse * 0.04, 1, 0.18 + pulse * 0.1],
    ]);
    if (liveMap.getLayer("selected-ring-layer")) {
      liveMap.setPaintProperty("selected-ring-layer", "line-opacity", 0.54 + pulse * 0.42);
      liveMap.setPaintProperty("selected-ring-layer", "line-width", 1.6 + pulse * 1.3);
    }
    if (liveMap.getLayer("mean-halo-layer")) {
      liveMap.setPaintProperty("mean-halo-layer", "circle-radius", 11 + pulse * 7);
      liveMap.setPaintProperty("mean-halo-layer", "circle-opacity", 0.22 + pulse * 0.2);
    }
  };
  radarAnimationId = requestAnimationFrame(tick);
}

function ensureLiveMap() {
  if (liveMap) return liveMapReady;
  const el = byId("live-map");
  if (!el) return Promise.resolve();

  liveMap = new maplibregl.Map({
    container: el,
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    center: initialCenter,
    zoom: initialZoom,
    pitch: 45, // Raven tilted view
    bearing: -15,
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
          color: "rgb(11, 20, 27)",
          "high-color": "rgb(8, 13, 18)",
          "space-color": "rgb(2, 5, 9)",
          "horizon-blend": 0.26,
          "star-intensity": 0.34,
        });
      }

      liveMap.addSource("candidates", {
        type: "geojson",
        data: emptyFeatureCollection,
      });
      liveMap.addSource("candidate-links", {
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
      liveMap.addSource("selected-ring", {
        type: "geojson",
        data: emptyFeatureCollection,
      });
      liveMap.addSource("selected-spoke", {
        type: "geojson",
        data: emptyFeatureCollection,
      });

      liveMap.addLayer({
        id: "ring-layer",
        type: "line",
        source: "ring",
        paint: {
          "line-color": "rgba(16, 185, 129, 0.4)",
          "line-width": 2,
          "line-dasharray": [2, 1],
          "line-opacity": 0.8,
        },
      });

      liveMap.addLayer({
        id: "ring-fill",
        type: "fill",
        source: "ring",
        paint: {
          "fill-color": "rgba(16, 185, 129, 0.05)",
          "fill-opacity": 0.3,
        },
      });

      liveMap.addLayer({
        id: "candidate-link-layer",
        type: "line",
        source: "candidate-links",
        paint: {
          "line-color": ["coalesce", ["get", "color"], "rgba(126, 212, 246, 0.46)"],
          "line-width": ["interpolate", ["linear"], ["get", "weightNorm"], 0, 0.8, 1, 2.2],
          "line-opacity": ["interpolate", ["linear"], ["get", "weightNorm"], 0, 0.18, 1, 0.52],
          "line-blur": 0.45,
        },
      });

      liveMap.addLayer({
        id: "selected-spoke-layer",
        type: "line",
        source: "selected-spoke",
        paint: {
          "line-color": "rgba(255, 150, 150, 0.78)",
          "line-width": 1.4,
          "line-dasharray": [1, 1.8],
          "line-opacity": 0.82,
        },
      });

      liveMap.addLayer({
        id: "selected-ring-fill",
        type: "fill",
        source: "selected-ring",
        paint: {
          "fill-color": "rgba(255, 130, 130, 0.08)",
          "fill-opacity": 0.42,
        },
      });

      liveMap.addLayer({
        id: "selected-ring-layer",
        type: "line",
        source: "selected-ring",
        paint: {
          "line-color": "rgba(255, 150, 150, 0.92)",
          "line-width": 2.2,
          "line-dasharray": [1.2, 1.5],
          "line-opacity": 0.9,
        },
      });

      liveMap.addLayer({
        id: "candidate-glow-layer",
        type: "circle",
        source: "candidates",
        paint: {
          "circle-color": ["get", "color"],
          "circle-radius": ["interpolate", ["linear"], ["get", "weightNorm"], 0, 8, 1, 18],
          "circle-opacity": ["interpolate", ["linear"], ["get", "weightNorm"], 0, 0.08, 1, 0.24],
          "circle-blur": 0.9,
        },
      });

  liveMap.addLayer({
    id: "candidate-layer",
    type: "circle",
    source: "candidates",
    paint: {
      "circle-color": ["case", ["==", ["get", "rank"], 1], "rgba(16, 185, 129, 0.95)", ["get", "color"]],
      "circle-radius": ["case", ["==", ["get", "rank"], 1], 12, ["interpolate", ["linear"], ["get", "weightNorm"], 0, 4, 1, 8]],
      "circle-stroke-color": "#fff",
      "circle-stroke-width": 1.5,
    },
  });

      liveMap.addLayer({
        id: "mean-halo-layer",
        type: "circle",
        source: "mean",
        paint: {
          "circle-color": "rgba(196, 238, 255, 0.24)",
          "circle-radius": 12,
          "circle-blur": 0.85,
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
        liveMap.flyTo({
          center: coords,
          zoom: Math.max(liveMap.getZoom(), 4.8),
          bearing: 25,
          pitch: 42,
          duration: 850,
        });
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

      startRadarPulse();
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
      liveMap.getSource("candidate-links").setData(emptyFeatureCollection);
      liveMap.getSource("mean").setData(emptyFeatureCollection);
      liveMap.getSource("ring").setData(emptyFeatureCollection);
      liveMap.getSource("selected-ring").setData(emptyFeatureCollection);
      liveMap.getSource("selected-spoke").setData(emptyFeatureCollection);
    }
    selectedRank = null;
    liveFusionSummary = null;
    renderCandidateInspector(null);
    renderGeoRanking([]);
    return;
  }

  const fusion = result.result.fusion;
  const candidates = Array.isArray(fusion?.candidates) ? fusion.candidates : [];
  if (!fusion || candidates.length === 0) {
    if (liveMap && liveMap.getSource("candidates")) {
      liveMap.getSource("candidates").setData(emptyFeatureCollection);
      liveMap.getSource("candidate-links").setData(emptyFeatureCollection);
      liveMap.getSource("mean").setData(emptyFeatureCollection);
      liveMap.getSource("ring").setData(emptyFeatureCollection);
      liveMap.getSource("selected-ring").setData(emptyFeatureCollection);
      liveMap.getSource("selected-spoke").setData(emptyFeatureCollection);
    }
    selectedRank = null;
    liveFusionSummary = null;
    renderCandidateInspector(null);
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
          matchId: item.candidate?.match_id ?? "",
          imagePath: item.candidate?.image_path ?? "",
          source: sourceFromMatchId(item.candidate?.match_id),
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
        posterior: item.posterior_weight ?? null,
        retrieval: item.candidate?.retrieval_score ?? item.retrieval_score ?? null,
        matchId: item.candidate?.match_id ?? "",
        imagePath: item.candidate?.image_path ?? "",
        source: sourceFromMatchId(item.candidate?.match_id),
      };
    })
    .filter(Boolean);

  renderGeoRanking(liveCandidatePoints);

  const meanLat = fusion.mean_latitude;
  const meanLon = fusion.mean_longitude;
  const ringRadius =
    fusion?.uncertainty_radius_m ??
    (fusion?.ellipse?.major_axis_m ? fusion.ellipse.major_axis_m * 0.6 : null);
  liveFusionSummary = { meanLat, meanLon, ringRadius };

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

  const linkFeatures =
    meanLat !== undefined && meanLon !== undefined
      ? visible
          .slice(0, Math.min(visible.length, 14))
          .map((item, idx) => {
            const cand = item.candidate || {};
            if (cand.latitude === undefined || cand.longitude === undefined) return null;
            const rawWeight = weightFrom(item);
            const weightNorm = maxWeight > 0 ? rawWeight / maxWeight : 0;
            return {
              type: "Feature",
              geometry: {
                type: "LineString",
                coordinates: [
                  [meanLon, meanLat],
                  [cand.longitude, cand.latitude],
                ],
              },
              properties: {
                rank: idx + 1,
                weightNorm,
                color: weightColor(weightNorm),
              },
            };
          })
          .filter(Boolean)
      : [];

  ensureLiveMap();
  liveMapReady.then(() => {
    liveMap.getSource("candidates").setData({
      type: "FeatureCollection",
      features,
    });
    liveMap.getSource("candidate-links").setData({
      type: "FeatureCollection",
      features: linkFeatures,
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
        bearing: 14,
        pitch: 30,
      });
    }

    if (!liveCandidatePoints.some((item) => item.rank === selectedRank) && liveCandidatePoints.length) {
      selectedRank = liveCandidatePoints[0].rank;
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
        liveMap.flyTo({
          center: [item.longitude, item.latitude],
          zoom: Math.max(liveMap.getZoom(), 4.8),
          bearing: 25,
          pitch: 42,
          duration: 850,
        });
      }
    });

    container.appendChild(row);
  });
}

function applyCandidateHighlight() {
  if (!liveMap) return;
  const highlight = selectedRank ?? -1;
  if (liveMap.getLayer("candidate-glow-layer")) {
    liveMap.setPaintProperty("candidate-glow-layer", "circle-color", [
      "case",
      ["==", ["get", "rank"], highlight],
      "rgba(255, 130, 130, 0.96)",
      ["get", "color"],
    ]);
    liveMap.setPaintProperty("candidate-glow-layer", "circle-opacity", [
      "case",
      ["==", ["get", "rank"], highlight],
      0.3,
      ["interpolate", ["linear"], ["get", "weightNorm"], 0, 0.08, 1, 0.24],
    ]);
  }
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
  if (liveMap.getLayer("candidate-link-layer")) {
    liveMap.setPaintProperty("candidate-link-layer", "line-color", [
      "case",
      ["==", ["get", "rank"], highlight],
      "rgba(255, 150, 150, 0.86)",
      ["coalesce", ["get", "color"], "rgba(126, 212, 246, 0.46)"],
    ]);
    liveMap.setPaintProperty("candidate-link-layer", "line-opacity", [
      "case",
      ["==", ["get", "rank"], highlight],
      0.85,
      ["interpolate", ["linear"], ["get", "weightNorm"], 0, 0.18, 1, 0.52],
    ]);
  }
  updateSelectedCandidateGeometry();
}

function setSelectedRank(rank) {
  selectedRank = rank;
  document.querySelectorAll(".geo-rank-item").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.rank) === rank);
  });
  applyCandidateHighlight();
  const selected = liveCandidatePoints.find((item) => item.rank === rank) || null;
  renderCandidateInspector(selected);
}

function updateSelectedCandidateGeometry() {
  if (!liveMap || !liveMap.getSource("selected-ring") || !liveMap.getSource("selected-spoke")) return;
  const selected = liveCandidatePoints.find((item) => item.rank === selectedRank);
  if (!selected || !liveFusionSummary) {
    liveMap.getSource("selected-ring").setData(emptyFeatureCollection);
    liveMap.getSource("selected-spoke").setData(emptyFeatureCollection);
    renderCandidateInspector(null);
    return;
  }
  const radius = liveFusionSummary.ringRadius;
  const ringFeature =
    radius && radius > 0
      ? {
          type: "Feature",
          geometry: {
            type: "Polygon",
            coordinates: [circlePolygon(selected.latitude, selected.longitude, radius)],
          },
          properties: { rank: selected.rank },
        }
      : null;
  const spokeFeature =
    liveFusionSummary.meanLat !== undefined && liveFusionSummary.meanLon !== undefined
      ? {
          type: "Feature",
          geometry: {
            type: "LineString",
            coordinates: [
              [liveFusionSummary.meanLon, liveFusionSummary.meanLat],
              [selected.longitude, selected.latitude],
            ],
          },
          properties: { rank: selected.rank },
        }
      : null;
  liveMap.getSource("selected-ring").setData({
    type: "FeatureCollection",
    features: ringFeature ? [ringFeature] : [],
  });
  liveMap.getSource("selected-spoke").setData({
    type: "FeatureCollection",
    features: spokeFeature ? [spokeFeature] : [],
  });
  renderCandidateInspector(selected);
}

function setupDiagnostics() {
  const trigger = byId("diag-trigger");
  const accordion = byId("diag-accordion");
  if (trigger && accordion) {
    trigger.addEventListener("click", () => {
      accordion.classList.toggle("active");
    });
  }
}

function init() {
  syncProfileSelect();
  setupFilePicker();
  setupAnalyzeAction();
  setupCanvasControls();
  setupMapControls();
  setupDiagnostics();
  setMetricsBaseline();
  setSummaryState("idle", "Upload an image and click Analyze Image to start.");
  renderGeoRanking([]);
  renderCandidateInspector(null);

  ensureLiveMap();
  if (liveMap) {
    setTimeout(() => liveMap.resize(), 0);
  }
}

window.addEventListener("load", init);
