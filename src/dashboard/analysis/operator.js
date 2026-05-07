import { byId, formatUtcNowLabel, normalizeError, postForm } from "./shared.js";

const profileStorageKey = "heimdallProfile";
let activeProfile = "paris";
let liveMap = null;
let liveMapReady = null;
let topLimit = 3;
let lastResult = null;
let selectedIndex = -1;

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

  const styleUrl = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
  const padding = getMapPadding();

  try {
    liveMap = new maplibregl.Map({
      container: el,
      style: styleUrl,
      center: initialCenter,
      zoom: initialZoom,
      pitch: 0, 
      bearing: 0,
      padding: padding,
      attributionControl: false,
      antialias: true
    });
  } catch (err) {
    console.error("CRITICAL: Map constructor error:", err);
    return Promise.reject(err);
  }

  liveMap.on("error", (e) => {
    console.error("LOG: MapLibre runtime error:", e.error || e);
  });

  liveMapReady = new Promise((resolve) => {
    liveMap.on("load", () => {
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

      liveMap.getCanvas().style.backgroundColor = "#0B0B0B";

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
          "circle-color": ["case", ["==", ["get", "rank"], 1], "#10b981", "#EAEAEA"],
          "circle-radius": ["case", ["==", ["get", "rank"], 1], 10, 6],
          "circle-stroke-color": "#000",
          "circle-stroke-width": 1.5
        }
      });
      liveMap.addLayer({
        id: "mean-layer",
        type: "circle",
        source: "mean",
        paint: { "circle-color": "#fff", "circle-radius": 4, "circle-stroke-color": "#10b981", "circle-stroke-width": 2 }
      });

      // Map Interaction: Click Marker to Select Card
      liveMap.on("click", "candidate-layer", (e) => {
        if (e.features.length > 0) {
          const index = e.features[0].properties.index;
          if (index !== undefined) {
            console.log(`LOG: Map marker clicked (Index ${index})`);
            selectCandidate(index);
          }
        }
      });

      liveMap.on("mouseenter", "candidate-layer", () => {
        liveMap.getCanvas().style.cursor = "pointer";
      });
      liveMap.on("mouseleave", "candidate-layer", () => {
        liveMap.getCanvas().style.cursor = "";
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

function selectCandidate(index) {
  const cards = document.querySelectorAll(".candidate-card");
  if (!cards.length) return;
  
  if (index < 0) index = 0;
  if (index >= cards.length) index = cards.length - 1;
  
  selectedIndex = index;
  const card = cards[index];
  
  cards.forEach(c => c.classList.remove("active"));
  card.classList.add("active");
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  
  const lat = parseFloat(card.dataset.lat);
  const lon = parseFloat(card.dataset.lon);
  
  if (liveMap) {
    console.log(`LOG: Inspecting candidate #${index + 1} at ${lat}, ${lon}`);
    liveMap.flyTo({ 
      center: [lon, lat], 
      zoom: 18, 
      pitch: 0,
      bearing: 0,
      padding: getMapPadding(),
      duration: 1200 
    });
  }
}

function renderCandidateList(result) {
  const container = byId("results-list");
  if (!container) return;
  container.replaceChildren();

  const fusion = result?.fused_estimate;
  const candidates = Array.isArray(result?.candidates) ? result.candidates : [];

  if (candidates.length === 0) {
    container.innerHTML = '<div class="empty-state">No candidates found.</div>';
    return;
  }

  const sorted = [...candidates].sort((a, b) => (b.posterior_weight ?? 0) - (a.posterior_weight ?? 0));
  const slice = sorted.slice(0, topLimit);

  slice.forEach((item, idx) => {
    const rank = idx + 1;
    const cand = item || {};
    const lat = cand.display_lat;
    const lon = cand.display_lon;
    const card = document.createElement("div");
    card.className = "candidate-card";
    card.dataset.lat = lat;
    card.dataset.lon = lon;
    card.dataset.index = idx;
    
    const coordString = `${Number(lat).toFixed(6)}, ${Number(lon).toFixed(6)}`;
    const mapsUrl = `https://www.google.com/maps/search/?api=1&query=${lat},${lon}`;
    
    // Check source
    const isExif = cand.match_id === "exif:gps";
    const sourceLabel = isExif ? "IMAGE METADATA (EXIF)" : `MATCH ID: ${cand.match_id || "N/A"}`;

    card.innerHTML = `
      <div class="card-top">
        <span class="card-rank">#${rank}</span>
        <span class="card-score">${((item.posterior_weight ?? 0) * 100).toFixed(1)}%</span>
      </div>
      <div class="card-address">Target Point ${rank}</div>
      <div class="card-sub">Paris, France • ${sourceLabel}</div>
      <div class="card-coords-row">
        <div class="card-coords">${coordString}</div>
        <button class="btn-icon-small copy-coords" title="Copy Coordinates">COPY</button>
      </div>
      <div style="display: flex; gap: 8px; margin-top: 8px;" class="card-actions-wrapper">
       <button class="btn-card-action" onclick="document.getElementById('btn-confirm-cand').click();">CONFIRM</button>
       <button class="btn-card-action" onclick="document.getElementById('btn-reject-cand').click();">REJECT</button>
      </div>
      <button class="btn-card-action open-maps">Open in Google Maps</button>
      <span class="copied-hint">Copied</span>
    `;

    card.querySelector(".copy-coords").addEventListener("click", (e) => {
      e.stopPropagation();
      navigator.clipboard.writeText(coordString).then(() => {
        const hint = card.querySelector(".copied-hint");
        hint.classList.add("visible");
        setTimeout(() => hint.classList.remove("visible"), 2000);
      });
    });

    card.querySelector(".open-maps").addEventListener("click", (e) => {
      e.stopPropagation();
      window.open(mapsUrl, "_blank");
    });

    card.addEventListener("click", () => selectCandidate(idx));

    container.appendChild(card);
  });
  
  selectedIndex = -1;
}

function renderLiveMap(result) {
  ensureLiveMap();
  const fusion = result?.fused_estimate;
  const candidates = Array.isArray(result?.candidates) ? result.candidates : [];
  
  if (!fusion || candidates.length === 0) return;

  const features = candidates.slice(0, topLimit).map((item, idx) => ({
    type: "Feature",
    geometry: { type: "Point", coordinates: [item.display_lon, item.display_lat] },
    properties: { rank: idx + 1, index: idx, lat: item.display_lat, lon: item.display_lon }
  }));

  const ringRadius = fusion.uncertainty_radius_m || (fusion.ellipse?.major_axis_m ? fusion.ellipse.major_axis_m * 0.6 : null) || result.result?.geo?.uncertainty_m;
  const ringFeature = (fusion.display_lat && fusion.display_lon && ringRadius) ? {
    type: "Feature",
    geometry: { type: "Polygon", coordinates: [circlePolygon(fusion.display_lat, fusion.display_lon, ringRadius)] }
  } : null;

  liveMapReady.then(() => {
    liveMap.getSource("candidates").setData({ type: "FeatureCollection", features });
    liveMap.getSource("ring").setData({ type: "FeatureCollection", features: ringFeature ? [ringFeature] : [] });
    liveMap.getSource("mean").setData({
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: { type: "Point", coordinates: [fusion.display_lon, fusion.display_lat] } }]
    });

    if (candidates.length > 0) {
      const top = candidates[0];
      if (top.display_lat !== undefined && top.display_lon !== undefined) {
         liveMap.easeTo({
           center: [top.display_lon, top.display_lat],
           zoom: 15,
           pitch: 0,
           padding: getMapPadding(),
           duration: 800
         });
      }
    }
  });
}

function renderSummary(result) {
  console.log("LOG: Rendering Summary", result);
  const res = result || {};
  const fusion = result.fused_estimate || {};
  const geo = result.fused_estimate || {};

  // Diagnostics
  setMetric("diag-backend", result.source?.filename || "-");
  setMetric("diag-worker", result.runtime?.worker_mode || "-");
  setMetric("diag-tier", geo.tier || "-");
  
  // Radius: prefer non-zero fusion radius, fallback to geo uncertainty
  let radius = "-";
  if (fusion.radius_km !== undefined && fusion.radius_km > 0) {
    radius = `${(fusion.radius_km * 1000).toFixed(1)}
  setMetric("diag-radius", radius);
  
  // Model Status: Check safe_demo or backend name
  const isDemo = false;
  const modelStatus = isDemo ? "DEMO FALLBACK" : "REAL MODEL";
  const modelStatusEl = byId("diag-model-status");
  if (modelStatusEl) {
    modelStatusEl.textContent = modelStatus;
    modelStatusEl.style.color = isDemo ? "#ffc864" : "#10b981";
  }

  byId("raw-json").textContent = JSON.stringify(result, null, 2);

  renderCandidateList(result);
  renderLiveMap(result);
  renderTimeline(result);
  renderClues(result);
}

function showAnalysisAlert(message) {
  const errorContainer = byId("analysis-error");
  if (!errorContainer) return;
  errorContainer.textContent = `RED ALERT\n${message}`;
  errorContainer.style.display = "block";
}

function clearAnalysisResults() {
  const resultsList = byId("results-list");
  if (resultsList) {
    resultsList.innerHTML = '<div class="empty-state">Upload imagery to begin.</div>';
  }
  const rawJson = byId("raw-json");
  if (rawJson) rawJson.textContent = "{}";
  lastResult = null;
}

/* --- Logic & Setup --- */

function setupFilePicker() {
  const input = byId("image-file");
  const trigger = byId("ingest-trigger");
  const previewBlock = byId("preview-block");
  const ingestBlock = byId("ingest-block");
  const thumb = byId("source-thumb");
  const filename = byId("preview-filename");
  const lightboxImg = byId("lightbox-img");
  const lightboxName = byId("lightbox-filename");
  const removeBtn = byId("remove-image");
  const geolocateBtn = byId("geolocate-image");
  const errorContainer = byId("analysis-error");
  
  if (!input || !trigger) return;

  trigger.addEventListener("click", () => input.click());

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) {
      console.log("LOG: Image selected:", file.name);
      filename.textContent = file.name;
      const reader = new FileReader();
      reader.onload = (e) => {
        thumb.src = e.target.result;
        lightboxImg.src = e.target.result;
        lightboxName.textContent = file.name;
        ingestBlock.style.display = "none";
        previewBlock.style.display = "flex";
        if (geolocateBtn) geolocateBtn.disabled = false;
        if (errorContainer) errorContainer.style.display = "none";
      };
      reader.readAsDataURL(file);
    }
  });

  if (removeBtn) {
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      console.log("LOG: Image removed");
      input.value = "";
      ingestBlock.style.display = "flex";
      previewBlock.style.display = "none";
      if (geolocateBtn) geolocateBtn.disabled = true;
      if (errorContainer) errorContainer.style.display = "none";
      clearAnalysisResults();
    });
  }

  filename.addEventListener("click", () => {
    navigator.clipboard.writeText(filename.textContent).then(() => {
      const status = byId("copy-status");
      status.classList.add("visible");
      setTimeout(() => status.classList.remove("visible"), 2000);
    });
  });
}

function setupLightbox() {
  const modal = byId("lightbox");
  const expandBtn = byId("expand-image");
  const closeBtn = byId("close-lightbox");
  const backdrop = byId("lightbox-backdrop");

  const close = () => modal.classList.remove("active");
  expandBtn.addEventListener("click", () => modal.classList.add("active"));
  closeBtn.addEventListener("click", close);
  backdrop.addEventListener("click", close);
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
}

function setupAnalysis() {
  const btn = byId("geolocate-image");
  const input = byId("image-file");
  const progress = byId("progress");
  const errorContainer = byId("analysis-error");
  if (!btn || !input) return;

  if (!input.files?.[0]) btn.disabled = true;

  btn.addEventListener("click", async () => {
    const file = input.files?.[0];
    if (!file) return;

    const profile = byId("profile-select").value || "paris";
    console.log("LOG: Starting pipeline...", { profile, filename: file.name });

    const form = new FormData();
    form.append("image", file);
    
    try {
      if (progress) progress.style.display = "block";
      if (errorContainer) errorContainer.style.display = "none";
      btn.disabled = true;
      
      const startTime = performance.now();

      const devMode = byId("dev-mode-toggle")?.checked ? "1" : "";
      form.append("profile", profile);
      if (devMode) form.append("dev_mode", devMode);
      const result = await postForm(`/api/operator/analyze`, form);

      // Handle the case where the server returns an error JSON with a session attached
      if (result.error && result.session) {
          lastResult = result.session;
          throw new Error(result.error);
      }

      const duration = (performance.now() - startTime) / 1000;
      
      console.log(`LOG: Pipeline finished in ${duration.toFixed(2)}s`);
      if (result.safe_demo) {
        const fallbackReason = result.geo_debug?.fallback_reason || result.fallback_reason || "server returned demo fallback";
        throw new Error(
          `Failure output is fallback/demo and may be random or synthetic. Do not trust this result.\nReason: ${fallbackReason}`
        );
      }
      console.log("LOG: Pipeline returned REAL model results.");

      lastResult = result;
      renderSummary(result);
    } catch (err) {
      const msg = normalizeError(err);
      console.error("LOG: Pipeline error", err);
      clearAnalysisResults();

      if (lastResult) {
          renderSummary(lastResult);
      }
      showAnalysisAlert(`Analysis failed.
${msg}`);

    } finally {
      if (progress) progress.style.display = "none";
      btn.disabled = false;
    }
  });
}

function setupMapControls() {
  byId("map-zoom-in").addEventListener("click", () => liveMap?.zoomIn());
  byId("map-zoom-out").addEventListener("click", () => liveMap?.zoomOut());
  
  byId("map-reset-paris").addEventListener("click", () => {
    liveMap?.easeTo({ center: initialCenter, zoom: 11, pitch: 0, bearing: 0, padding: getMapPadding(), duration: 1400 });
  });
  byId("map-reset-globe").addEventListener("click", () => {
    liveMap?.easeTo({ center: [10, 20], zoom: 1.8, pitch: 0, bearing: 0, padding: getMapPadding(), duration: 1800 });
  });
}

function setupKeyboardNav() {
  const list = byId("results-list");
  list.addEventListener("keydown", (e) => {
    const cards = document.querySelectorAll(".candidate-card");
    if (!cards.length) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      selectCandidate(selectedIndex + 1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selectCandidate(selectedIndex - 1);
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (selectedIndex >= 0) selectCandidate(selectedIndex);
    }
  });
}

function setupToggles() {
  const group = byId("top-n-toggle");
  if (!group) return;
  group.querySelectorAll("button").forEach(btn => {
    btn.addEventListener("click", () => {
      group.querySelectorAll("button").forEach(b => b.classList.remove("active"));
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
  const copySumBtn = byId("copy-diag-all");
  const copyJsonBtn = byId("copy-json");
  const expandJsonBtn = byId("expand-json");
  
  const jsonModal = byId("json-modal");
  const jsonModalText = byId("modal-json-text");
  const closeJsonBtn = byId("close-json-modal");
  const jsonBackdrop = byId("json-modal-backdrop");
  const copyModalBtn = byId("copy-json-modal");

  if (trigger && body) trigger.addEventListener("click", () => body.classList.toggle("active"));

  const flashBtn = (btn, text = "COPIED") => {
    const old = btn.textContent;
    btn.textContent = text;
    setTimeout(() => { btn.textContent = old; }, 2000);
  };

  if (copySumBtn) {
    copySumBtn.addEventListener("click", () => {
      const grid = byId("diag-summary-grid");
      const text = Array.from(grid.children).map(el => el.textContent).join(" ");
      navigator.clipboard.writeText(text).then(() => flashBtn(copySumBtn));
    });
  }

  const copyRawJson = (btn) => {
    const json = byId("raw-json").textContent;
    navigator.clipboard.writeText(json).then(() => flashBtn(btn));
  };

  if (copyJsonBtn) copyJsonBtn.addEventListener("click", () => copyRawJson(copyJsonBtn));
  if (copyModalBtn) copyModalBtn.addEventListener("click", () => copyRawJson(copyModalBtn));

  if (expandJsonBtn && jsonModal) {
    expandJsonBtn.addEventListener("click", () => {
      jsonModalText.textContent = byId("raw-json").textContent;
      jsonModal.classList.add("active");
    });
  }

  const closeJson = () => jsonModal?.classList.remove("active");
  if (closeJsonBtn) closeJsonBtn.addEventListener("click", closeJson);
  if (jsonBackdrop) jsonBackdrop.addEventListener("click", closeJson);
}

function init() {
  console.log("LOG: Operator UI Initializing");
  ensureLiveMap();
  setupFilePicker();
  setupLightbox();
  setupAnalysis();
  setupMapControls();
  setupKeyboardNav();
  setupDiagnostics();
  setupToggles();
  
  const profileSelect = byId("profile-select");
  if (profileSelect) {
    const stored = localStorage.getItem(profileStorageKey);
    if (stored) profileSelect.value = stored;
    profileSelect.addEventListener("change", () => localStorage.setItem(profileStorageKey, profileSelect.value));
  }
}

window.addEventListener("load", init);


function renderTimeline(session) {
  const timelineEl = byId("session-timeline");
  if (!timelineEl) return;
  timelineEl.innerHTML = "";

  if (!session || !session.timeline) return;
  session.timeline.forEach(event => {
     const div = document.createElement("div");
     div.className = `timeline-item ${event.level}`;
     div.innerHTML = `
        <div class="timeline-time">${event.timestamp.split(" ").slice(1).join(" ")}</div>
        <div class="timeline-msg">${event.message}</div>
     `;
     timelineEl.appendChild(div);
  });
}

function renderClues(session) {
  const cluesEl = byId("clues-list");
  if (!cluesEl) return;
  cluesEl.innerHTML = "";

  if (!session || !session.clues || session.clues.length === 0) {
      cluesEl.innerHTML = '<div class="empty-state">No clues extracted.</div>';
      return;
  }

  session.clues.forEach(clue => {
      const div = document.createElement("div");
      div.className = "clue-chip";
      div.innerHTML = `<strong>${clue.name}</strong> <span>(${(clue.score*100).toFixed(0)}%)</span> - ${clue.description}`;
      cluesEl.appendChild(div);
  });
}

function setupOperatorActions() {
    const confirmBtn = byId("btn-confirm-cand");
    const rejectBtn = byId("btn-reject-cand");
    const noteInput = byId("operator-note-input");
    const saveNoteBtn = byId("btn-save-note");
    const exportBtn = byId("btn-export-session");

    if (confirmBtn) {
        confirmBtn.addEventListener("click", () => {
             if (selectedIndex >= 0) {
                 postForm("/api/operator/confirm", JSON.stringify({rank: selectedIndex + 1, action: "confirm"}));
             }
        });
    }
    if (rejectBtn) {
        rejectBtn.addEventListener("click", () => {
             if (selectedIndex >= 0) {
                 postForm("/api/operator/confirm", JSON.stringify({rank: selectedIndex + 1, action: "reject"}));
             }
        });
    }
    if (saveNoteBtn && noteInput) {
        saveNoteBtn.addEventListener("click", () => {
             postForm("/api/operator/note", JSON.stringify({note: noteInput.value}));
        });
    }

    if (exportBtn) {
        exportBtn.addEventListener("click", () => {
             fetch("/api/operator/export.json").then(r=>r.json()).then(data => {
                  const blob = new Blob([JSON.stringify(data, null, 2)], {type: "application/json"});
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = "session_export.json";
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                  URL.revokeObjectURL(url);
             });
        });
    }

    // allow manual map pin
    if (liveMap) {
        liveMap.on('click', (e) => {
             const manualPinMode = byId("manual-pin-mode")?.checked;
             if (manualPinMode) {
                  const lat = e.lngLat.lat;
                  const lon = e.lngLat.lng;
                  postForm("/api/operator/pin", JSON.stringify({lat, lon, label: "Operator Pin"})).then(() => {
                      // refresh UI
                      fetch("/api/operator/session").then(r=>r.json()).then(renderSummary);
                  });
             }
        });
    }
}

// override init to call setupOperatorActions
const oldInit = init;
init = function() {
   oldInit();
   setupOperatorActions();
};
