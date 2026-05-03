import { byId, formatUtcNowLabel, normalizeError, postForm } from "./shared.js";

const profileStorageKey = "heimdallProfile";
let activeProfile = "paris";
let liveMap = null;
let liveMapReady = null;
let topLimit = 3;
let lastResult = null;

const initialCenter = [2.3522, 48.8566]; // Paris
const initialZoom = 11;
const emptyFeatureCollection = { type: "FeatureCollection", features: [] };

/* --- Map Core --- */

function getMapPadding() {
  const panel = document.querySelector(".analysis-panel");
  const panelWidth = panel ? panel.offsetWidth : 420;
  return { left: 0, right: panelWidth + 40, top: 0, bottom: 0 };
}

function ensureLiveMap() {
  if (liveMap) return liveMapReady;
  const el = byId("live-map");
  if (!el) {
    console.error("CRITICAL: Map container #live-map not found");
    return Promise.reject("No map container");
  }

  console.log("LOG: Initializing Map sequence with padding...");

  const styleUrl = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
  const padding = getMapPadding();

  try {
    liveMap = new maplibregl.Map({
      container: el,
      style: styleUrl,
      center: initialCenter,
      zoom: initialZoom,
      pitch: initialZoom < 5 ? 0 : 60,
      bearing: -20,
      padding: padding,
      attributionControl: false,
      antialias: true
    });
  } catch (err) {
    console.error("CRITICAL: Map constructor error:", err);
    return Promise.reject(err);
  }

  liveMap.on("zoom", () => {
    const zoom = liveMap.getZoom();
    const currentPitch = liveMap.getPitch();
    const targetPitch = zoom < 5 ? 0 : 60;
    
    if (Math.abs(currentPitch - targetPitch) > 1) {
      liveMap.easeTo({
        pitch: targetPitch,
        padding: getMapPadding(),
        duration: 400
      });
    }
  });

  liveMap.on("error", (e) => {
    console.error("LOG: MapLibre runtime error:", e.error || e);
  });

  liveMapReady = new Promise((resolve) => {
    liveMap.on("load", () => {
      console.log("LOG: Map base load complete");
      
      try {
        if (typeof liveMap.setProjection === "function") {
          liveMap.setProjection({ type: "globe" });
        }
      } catch (e) {
        console.warn("LOG: Globe fallback:", e);
      }

      // Neutralize Background Layer
      const style = liveMap.getStyle();
      const bgLayer = style.layers.find(l => l.type === "background");
      if (bgLayer) {
        liveMap.setPaintProperty(bgLayer.id, "background-color", "#0B0B0B");
      }

      // Neutralize Atmosphere / Fog
      if (typeof liveMap.setFog === "function") {
        liveMap.setFog({
          color: "#0B0B0B",
          "high-color": "#0B0B0B",
          "space-color": "#0B0B0B",
          "horizon-blend": 0.02
        });
      }

      // Fallback Canvas Background
      liveMap.getCanvas().style.backgroundColor = "#0B0B0B";

      const currentPadding = getMapPadding();
      liveMap.setPadding(currentPadding);

      liveMap.easeTo({
        center: initialCenter,
        zoom: initialZoom,
        pitch: initialZoom < 5 ? 0 : 60,
        padding: currentPadding,
        duration: 1000
      });

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

      window.addEventListener("resize", () => {
        liveMap.setPadding(getMapPadding());
        liveMap.resize();
      });

      setTimeout(() => liveMap.resize(), 100);
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
  
  if (sorted.length === 1) {
    const note = document.createElement("div");
    note.className = "card-sub";
    note.style.textAlign = "center";
    note.style.marginBottom = "8px";
    note.textContent = "Only 1 candidate returned";
    container.appendChild(note);
  }

  sorted.slice(0, topLimit).forEach((item, idx) => {
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
        liveMap.flyTo({ 
          center: [cand.longitude, cand.latitude], 
          zoom: 18, 
          pitch: 65,
          bearing: -20,
          duration: 1200 
        });
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

  const features = candidates.slice(0, topLimit).map((item, idx) => ({
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

    if (candidates.length > 0) {
      const top = candidates[0].candidate;
      liveMap.easeTo({ center: [top.longitude, top.latitude], zoom: 15, pitch: 60, duration: 800 });
    }
  });
}

function renderSummary(result) {
  const fusion = result.result.fusion;
  const geo = result.result.geo;

  const statusEl = byId("result-status");
  if (statusEl) statusEl.style.display = "flex";
  
  const topLat = fusion?.mean_latitude || result.result.candidates?.[0]?.latitude;
  const topLon = fusion?.mean_longitude || result.result.candidates?.[0]?.longitude;
  setMetric("metric-location", topLat ? `${Number(topLat).toFixed(4)}, ${Number(topLon).toFixed(4)}` : "-");
  
  const weight = fusion?.candidates?.[0]?.posterior_weight ?? result.result.candidates?.[0]?.retrieval_score ?? 0;
  setMetric("metric-confidence", `${(weight * 100).toFixed(1)}%`);

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
  const trigger = byId("ingest-trigger");
  const filename = byId("ingest-filename");
  const previewWrap = byId("ingest-preview-wrap");
  const previewImg = byId("ingest-preview");
  
  if (!input || !trigger) return;

  trigger.addEventListener("click", () => input.click());

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) {
      filename.textContent = file.name;
      const reader = new FileReader();
      reader.onload = (e) => {
        if (previewImg) previewImg.src = e.target.result;
        if (previewWrap) previewWrap.style.display = "block";
      };
      reader.readAsDataURL(file);
    } else {
      filename.textContent = "No file selected";
      if (previewWrap) previewWrap.style.display = "none";
    }
  });
}

function setupAnalysis() {
  const btn = byId("geolocate-image");
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
      lastResult = result;
      renderSummary(result);
    } catch (err) {
      console.error(err);
    } finally {
      if (progress) progress.style.display = "none";
      btn.disabled = false;
    }
  });
}

function setupToggles() {
  const toggleGroup = byId("top-n-toggle");
  if (!toggleGroup) return;

  toggleGroup.querySelectorAll("button").forEach(btn => {
    btn.addEventListener("click", () => {
      toggleGroup.querySelectorAll("button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      topLimit = parseInt(btn.dataset.value || "3");
      if (lastResult) {
        renderCandidateList(lastResult);
        renderLiveMap(lastResult);
      }
    });
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
  console.log("LOG: App Init Start");
  ensureLiveMap();
  setupFilePicker();
  setupAnalysis();
  setupDiagnostics();
  setupToggles();
  
  const profileSelect = byId("profile-select");
  if (profileSelect) {
    const stored = localStorage.getItem(profileStorageKey);
    if (stored) profileSelect.value = stored;
    profileSelect.addEventListener("change", () => localStorage.setItem(profileStorageKey, profileSelect.value));
  }
  console.log("LOG: App Init End");
}

window.addEventListener("load", init);
