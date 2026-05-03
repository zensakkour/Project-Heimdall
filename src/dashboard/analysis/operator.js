import { byId, formatUtcNowLabel, normalizeError, postForm } from "./shared.js";

const profileStorageKey = "heimdallProfile";
let activeProfile = "paris";
let liveMap = null;
let liveMapReady = null;
let selectedRank = null;
let liveCandidatePoints = [];
let liveFusionSummary = null;

const initialCenter = [2.3522, 48.8566]; // Paris
const initialZoom = 2;
const emptyFeatureCollection = { type: "FeatureCollection", features: [] };

/* --- Map Core --- */

function ensureLiveMap() {
  if (liveMap) return liveMapReady;
  const el = byId("live-map");
  if (!el) return Promise.resolve();

  liveMap = new maplibregl.Map({
    container: el,
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    center: initialCenter,
    zoom: initialZoom,
    pitch: 45,
    bearing: -15,
    attributionControl: false,
  });

  liveMapReady = new Promise((resolve) => {
    liveMap.on("load", () => {
      // Sources
      liveMap.addSource("candidates", { type: "geojson", data: emptyFeatureCollection });
      liveMap.addSource("ring", { type: "geojson", data: emptyFeatureCollection });
      liveMap.addSource("mean", { type: "geojson", data: emptyFeatureCollection });

      // Layers
      liveMap.addLayer({
        id: "ring-fill",
        type: "fill",
        source: "ring",
        paint: { "fill-color": "#10b981", "fill-opacity": 0.05 }
      });
      liveMap.addLayer({
        id: "ring-layer",
        type: "line",
        source: "ring",
        paint: { "line-color": "#10b981", "line-width": 2, "line-dasharray": [2, 1], "line-opacity": 0.6 }
      });
      liveMap.addLayer({
        id: "candidate-layer",
        type: "circle",
        source: "candidates",
        paint: {
          "circle-color": ["case", ["==", ["get", "rank"], 1], "#10b981", "#6fa7cb"],
          "circle-radius": ["case", ["==", ["get", "rank"], 1], 10, 6],
          "circle-stroke-color": "#fff",
          "circle-stroke-width": 1.5
        }
      });
      liveMap.addLayer({
        id: "mean-layer",
        type: "circle",
        source: "mean",
        paint: { "circle-color": "#fff", "circle-radius": 4, "circle-stroke-color": "#10b981", "circle-stroke-width": 2 }
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
    const lat2 = Math.asin(Math.sin(latRad) * Math.cos(rad) + Math.cos(latRad) * Math.sin(rad) * Math.cos(bearing));
    const lon2 = lonRad + Math.atan2(Math.sin(bearing) * Math.sin(rad) * Math.cos(latRad), Math.cos(rad) - Math.sin(latRad) * Math.sin(lat2));
    coords.push([(lon2 * 180) / Math.PI, (lat2 * 180) / Math.PI]);
  }
  return coords;
}

/* --- UI Rendering --- */

function setMetric(id, value) {
  const el = byId(id);
  if (el) el.textContent = value;
}

function renderCandidateList(result) {
  const container = byId("results-list");
  if (!container) return;
  container.replaceChildren();

  const fusion = result?.result?.fusion;
  const candidates = Array.isArray(fusion?.candidates) ? fusion.candidates : [];

  if (candidates.length === 0) {
    container.innerHTML = '<div class="empty-state">No candidates found.</div>';
    return;
  }

  const sorted = [...candidates].sort((a, b) => (b.posterior_weight ?? 0) - (a.posterior_weight ?? 0));
  liveCandidatePoints = sorted.map((item, idx) => ({
    rank: idx + 1,
    lat: item.candidate?.latitude,
    lon: item.candidate?.longitude,
    weight: item.posterior_weight ?? 0
  }));

  sorted.slice(0, 20).forEach((item, idx) => {
    const rank = idx + 1;
    const cand = item.candidate || {};
    const card = document.createElement("div");
    card.className = `candidate-card ${rank === 1 ? "active" : ""}`;
    
    card.innerHTML = `
      <div class="card-top">
        <span class="card-rank">#${rank}</span>
        <span class="card-score">${((item.posterior_weight ?? 0) * 100).toFixed(1)}%</span>
      </div>
      <div class="card-coords">${Number(cand.latitude).toFixed(5)}, ${Number(cand.longitude).toFixed(5)}</div>
      <div class="card-sub">Paris, France • Match ID: ${cand.match_id || "N/A"}</div>
    `;

    card.addEventListener("click", () => {
      document.querySelectorAll(".candidate-card").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      if (liveMap) {
        liveMap.flyTo({ center: [cand.longitude, cand.latitude], zoom: 14, duration: 1000 });
      }
    });

    container.appendChild(card);
  });
}

function renderLiveMap(result) {
  ensureLiveMap();
  const fusion = result?.result?.fusion;
  const candidates = Array.isArray(fusion?.candidates) ? fusion.candidates : [];
  
  if (!fusion || candidates.length === 0) return;

  const features = candidates.slice(0, 20).map((item, idx) => ({
    type: "Feature",
    geometry: { type: "Point", coordinates: [item.candidate.longitude, item.candidate.latitude] },
    properties: { rank: idx + 1 }
  }));

  const ringRadius = fusion.uncertainty_radius_m || (fusion.ellipse?.major_axis_m ? fusion.ellipse.major_axis_m * 0.6 : null);
  const ringFeature = (fusion.mean_latitude && fusion.mean_longitude && ringRadius) ? {
    type: "Feature",
    geometry: { type: "Polygon", coordinates: [circlePolygon(fusion.mean_latitude, fusion.mean_longitude, ringRadius)] }
  } : null;

  liveMapReady.then(() => {
    liveMap.getSource("candidates").setData({ type: "FeatureCollection", features });
    liveMap.getSource("ring").setData({ type: "FeatureCollection", features: ringFeature ? [ringFeature] : [] });
    liveMap.getSource("mean").setData({
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: { type: "Point", coordinates: [fusion.mean_longitude, fusion.mean_latitude] } }]
    });

    liveMap.easeTo({ center: [fusion.mean_longitude, fusion.mean_latitude], zoom: 12, duration: 800 });
  });
}

function renderSummary(result) {
  const fusion = result.result.fusion;
  const geo = result.result.geo;

  // Status Section
  const statusEl = byId("result-status");
  if (statusEl) statusEl.style.display = "flex";
  
  const thumb = byId("source-thumb");
  if (thumb) thumb.src = result.image_data || "";
  
  const topLat = fusion?.mean_latitude || result.result.candidates?.[0]?.latitude;
  const topLon = fusion?.mean_longitude || result.result.candidates?.[0]?.longitude;
  setMetric("metric-location", topLat ? `${Number(topLat).toFixed(4)}, ${Number(topLon).toFixed(4)}` : "-");
  
  const weight = fusion?.candidates?.[0]?.posterior_weight ?? result.result.candidates?.[0]?.retrieval_score ?? 0;
  setMetric("metric-confidence", `${(weight * 100).toFixed(1)}%`);

  // Diagnostics
  setMetric("diag-backend", result.result?.backend || "-");
  setMetric("diag-worker", result.runtime?.worker_mode || "-");
  setMetric("diag-tier", geo?.confidence_tier || "-");
  setMetric("diag-radius", fusion?.uncertainty_radius_m ? `${fusion.uncertainty_radius_m.toFixed(1)}m` : "-");
  byId("raw-json").textContent = JSON.stringify(result, null, 2);

  renderCandidateList(result);
  renderLiveMap(result);
}

/* --- Logic & Setup --- */

function setupFilePicker() {
  const input = byId("image-file");
  const name = byId("image-file-name");
  if (!input || !name) return;
  input.addEventListener("change", () => {
    name.textContent = input.files?.[0]?.name || "No file selected";
  });
}

function setupAnalysis() {
  const btn = byId("analyze-image");
  const input = byId("image-file");
  const progress = byId("progress");
  if (!btn || !input) return;

  btn.addEventListener("click", async () => {
    if (!input.files?.[0]) return;
    const form = new FormData();
    form.append("image", input.files[0]);
    
    try {
      if (progress) progress.style.display = "block";
      btn.disabled = true;
      const profile = byId("profile-select").value || "paris";
      const result = await postForm(`/analyze/image?profile=${profile}`, form);
      renderSummary(result);
    } catch (err) {
      console.error(err);
    } finally {
      if (progress) progress.style.display = "none";
      btn.disabled = false;
    }
  });
}

function setupDiagnostics() {
  const trigger = byId("diag-trigger");
  const body = byId("diag-accordion");
  if (trigger && body) {
    trigger.addEventListener("click", () => body.classList.toggle("active"));
  }
}

function init() {
  ensureLiveMap();
  setupFilePicker();
  setupAnalysis();
  setupDiagnostics();
  
  // Sync profile select
  const profileSelect = byId("profile-select");
  if (profileSelect) {
    const stored = localStorage.getItem(profileStorageKey);
    if (stored) profileSelect.value = stored;
    profileSelect.addEventListener("change", () => localStorage.setItem(profileStorageKey, profileSelect.value));
  }
}

window.addEventListener("load", init);
