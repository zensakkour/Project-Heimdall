import { byId, formatUtcNowLabel, normalizeError, postForm } from "./shared.js";

const profileStorageKey = "heimdallProfile";
let activeProfile = "paris";
let liveMap = null;
let liveMapReady = null;
let topLimit = 3;
let lastResult = null;
let selectedIndex = -1;
let candidateMarkers = [];
let selectedLightboxDetectionIndex = 0;

const parisCenter = [2.3522, 48.8566];
const globeCenter = [10, 20];
const globeZoom = 1.8;
const minPitch = 0;
const maxPitch = 70;
const emptyFeatureCollection = { type: "FeatureCollection", features: [] };
const mapStyleUrl = "https://tiles.openfreemap.org/styles/dark";

/* --- Map Core --- */

function getMapPadding() {
  return { left: 0, right: 0, top: 0, bottom: 0 };
}

function firstSymbolLayerId() {
  const layers = liveMap?.getStyle()?.layers || [];
  return layers.find((layer) => layer.type === "symbol")?.id;
}

function setPaintIfPossible(layerId, property, value) {
  try {
    liveMap.setPaintProperty(layerId, property, value);
  } catch {
    // Style variants may not support every paint property.
  }
}

function tuneMapFor3DReadability() {
  const layers = liveMap?.getStyle()?.layers || [];
  layers.forEach((layer) => {
    if (layer.source !== "openmaptiles") return;

    if (layer.type === "symbol") {
      setPaintIfPossible(layer.id, "text-opacity", 0.78);
      setPaintIfPossible(layer.id, "text-color", "#a8a8a8");
      setPaintIfPossible(layer.id, "icon-opacity", 0.58);
      setPaintIfPossible(layer.id, "text-halo-color", "#000000");
      setPaintIfPossible(layer.id, "text-halo-width", 1.6);
    }

    if (layer.type === "line") {
      const id = layer.id.toLowerCase();
      if (id.includes("road") || id.includes("transport") || id.includes("path") || id.includes("rail")) {
        setPaintIfPossible(layer.id, "line-opacity", 0.52);
        setPaintIfPossible(layer.id, "line-color", "#424242");
      }
    }

    if (layer.type === "fill") {
      const id = layer.id.toLowerCase();
      if (id.includes("land") || id.includes("park") || id.includes("place")) {
        setPaintIfPossible(layer.id, "fill-color", "#151515");
        setPaintIfPossible(layer.id, "fill-opacity", 0.95);
      }
      if (id.includes("water")) {
        setPaintIfPossible(layer.id, "fill-color", "#1f1f1f");
        setPaintIfPossible(layer.id, "fill-opacity", 1);
      }
    }

    if (layer.type === "fill" && layer.id.toLowerCase().includes("building")) {
      setPaintIfPossible(layer.id, "fill-opacity", 0.08);
    }
  });
}

function addBuildingExtrusions() {
  if (!liveMap || liveMap.getLayer("heimdall-building-extrusion") || !liveMap.getSource("openmaptiles")) return;
  liveMap.addLayer({
    id: "heimdall-building-extrusion",
    type: "fill-extrusion",
    source: "openmaptiles",
    "source-layer": "building",
    minzoom: 13,
    paint: {
      "fill-extrusion-color": [
        "interpolate",
        ["linear"],
        ["zoom"],
        13,
        "#3f3f3f",
        16,
        "#707070"
      ],
      "fill-extrusion-height": [
        "interpolate",
        ["linear"],
        ["zoom"],
        13,
        0,
        15,
        ["coalesce", ["to-number", ["get", "render_height"]], ["to-number", ["get", "height"]], 12]
      ],
      "fill-extrusion-base": ["coalesce", ["to-number", ["get", "render_min_height"]], ["to-number", ["get", "min_height"]], 0],
      "fill-extrusion-opacity": 1,
      "fill-extrusion-vertical-gradient": true
    }
  }, firstSymbolLayerId());
}

function moveLayerToTop(layerId) {
  if (!liveMap?.getLayer(layerId)) return;
  try {
    liveMap.moveLayer(layerId);
  } catch {
    // Ignore layer-order races during style/data refreshes.
  }
}

function bringCandidateLayersToFront() {
  [
    "ring-fill",
    "ring-layer",
    "mean-layer",
    "candidate-halo-layer",
    "candidate-stem-layer",
    "candidate-layer",
    "candidate-label-layer",
    "candidate-hit-layer"
  ].forEach(moveLayerToTop);
}

function clearCandidateMarkers() {
  candidateMarkers.forEach((marker) => marker.remove());
  candidateMarkers = [];
}

function updateHtmlMarkerSelection(index) {
  candidateMarkers.forEach((marker, idx) => {
    const el = marker.getElement();
    el.classList.toggle("selected", idx === index);
    el.classList.toggle("top", idx === 0);
  });
}

function updatePinScale() {
  if (!liveMap) return;
  const zoom = liveMap.getZoom();
  const t = Math.max(0, Math.min(1, (zoom - 7) / 7));
  const eased = t * t * (3 - 2 * t);
  const scale = 0.08 + eased * 0.92;
  candidateMarkers.forEach((marker) => {
    const el = marker.getElement();
    el.style.setProperty("--pin-scale", scale.toFixed(2));
    el.classList.toggle("pin-dot", scale < 0.22);
  });
}

function renderHtmlCandidateMarkers(candidates) {
  if (!liveMap) return;
  clearCandidateMarkers();
  candidates.slice(0, topLimit).forEach((item, idx) => {
    const lat = candidateLat(item);
    const lon = candidateLon(item);
    if (!Number.isFinite(Number(lat)) || !Number.isFinite(Number(lon))) return;

    const el = document.createElement("button");
    el.type = "button";
    el.className = `geo-pin${idx === 0 ? " top" : ""}${idx === selectedIndex ? " selected" : ""}`;
    el.dataset.index = String(idx);
    el.setAttribute("aria-label", `Select geo candidate ${idx + 1}`);
    el.innerHTML = `<span class="geo-pin-halo"></span><span class="geo-pin-head">${idx + 1}</span><span class="geo-pin-stem"></span>`;
    el.addEventListener("click", (event) => {
      event.stopPropagation();
      selectCandidate(idx);
    });

    const marker = new maplibregl.Marker({ element: el, anchor: "bottom" })
      .setLngLat([Number(lon), Number(lat)])
      .addTo(liveMap);
    candidateMarkers.push(marker);
  });
  updatePinScale();
}

function ensureLiveMap() {
  if (liveMap) return liveMapReady;
  const el = byId("live-map");
  if (!el) {
    console.error("CRITICAL: Map container #live-map not found");
    return Promise.reject("No map container");
  }

  const padding = getMapPadding();

  try {
    liveMap = new maplibregl.Map({
      container: el,
      style: mapStyleUrl,
      center: globeCenter,
      zoom: globeZoom,
      pitch: 0, 
      bearing: 0,
      padding: padding,
      projection: { type: "globe" },
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

      const style = liveMap.getStyle();
      const bgLayer = style.layers.find(l => l.type === "background");
      if (bgLayer) {
        liveMap.setPaintProperty(bgLayer.id, "background-color", "#000000");
      }

      if (typeof liveMap.setFog === "function") {
        liveMap.setFog({
          color: "#000000",
          "high-color": "#000000",
          "space-color": "#000000",
          "horizon-blend": 0.02
        });
      }

      liveMap.getCanvas().style.backgroundColor = "#000000";

      tuneMapFor3DReadability();
      addBuildingExtrusions();

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
        id: "candidate-stem-layer",
        type: "circle",
        source: "candidates",
        paint: {
          "circle-color": ["case", ["==", ["get", "index"], selectedIndex], "#eaeaea", ["==", ["get", "rank"], 1], "#7dd3a4", "#9a9a9a"],
          "circle-radius": ["case", ["==", ["get", "index"], selectedIndex], 6, 5],
          "circle-translate": [0, 20],
          "circle-translate-anchor": "viewport",
          "circle-opacity": 0.88
        }
      });
      liveMap.addLayer({
        id: "candidate-halo-layer",
        type: "circle",
        source: "candidates",
        paint: {
          "circle-color": ["case", ["==", ["get", "index"], selectedIndex], "#d7d7d7", ["==", ["get", "rank"], 1], "#10b981", "#bdbdbd"],
          "circle-opacity": ["case", ["==", ["get", "index"], selectedIndex], 0.22, ["==", ["get", "rank"], 1], 0.16, 0.1],
          "circle-radius": ["case", ["==", ["get", "index"], selectedIndex], 40, ["==", ["get", "rank"], 1], 34, 24],
          "circle-stroke-color": "#ffffff",
          "circle-stroke-opacity": ["case", ["==", ["get", "index"], selectedIndex], 0.18, 0.08],
          "circle-stroke-width": 1
        }
      });
      liveMap.addLayer({
        id: "candidate-layer",
        type: "circle",
        source: "candidates",
        paint: {
          "circle-color": ["case", ["==", ["get", "index"], selectedIndex], "#ffffff", ["==", ["get", "rank"], 1], "#10b981", "#5f6468"],
          "circle-radius": ["case", ["==", ["get", "index"], selectedIndex], 17, ["==", ["get", "rank"], 1], 16, 12],
          "circle-stroke-color": "#f4f4f4",
          "circle-stroke-width": ["case", ["==", ["get", "index"], selectedIndex], 3, ["==", ["get", "rank"], 1], 2.5, 1.8],
          "circle-opacity": 0.96
        }
      });
      liveMap.addLayer({
        id: "candidate-label-layer",
        type: "symbol",
        source: "candidates",
        layout: {
          "text-field": ["to-string", ["get", "rank"]],
          "text-size": ["case", ["==", ["get", "index"], selectedIndex], 17, 14],
          "text-font": ["Open Sans Bold"],
          "text-allow-overlap": true,
          "text-ignore-placement": true
        },
        paint: {
          "text-color": ["case", ["==", ["get", "index"], selectedIndex], "#111111", "#ffffff"],
          "text-halo-color": "rgba(0,0,0,0)",
          "text-halo-width": 0
        }
      });
      liveMap.addLayer({
        id: "candidate-hit-layer",
        type: "circle",
        source: "candidates",
        paint: {
          "circle-color": "#ffffff",
          "circle-opacity": 0.01,
          "circle-radius": 22,
          "circle-stroke-opacity": 0
        }
      });
      liveMap.addLayer({
        id: "mean-layer",
        type: "circle",
        source: "mean",
        paint: { "circle-color": "#fff", "circle-radius": 4, "circle-stroke-color": "#10b981", "circle-stroke-width": 2 }
      });

      // Map Interaction: Click Marker to Select Card
      const handleCandidateMarkerClick = (e) => {
        if (e.features.length > 0) {
          const index = Number(e.features[0].properties.index);
          if (Number.isFinite(index)) {
            console.log(`LOG: Map marker clicked (Index ${index})`);
            selectCandidate(index);
          }
        }
      };

      liveMap.on("click", "candidate-layer", handleCandidateMarkerClick);
      liveMap.on("click", "candidate-hit-layer", handleCandidateMarkerClick);

      const showPointer = () => {
        liveMap.getCanvas().style.cursor = "pointer";
      };
      const clearPointer = () => {
        liveMap.getCanvas().style.cursor = "";
      };
      liveMap.on("mouseenter", "candidate-layer", showPointer);
      liveMap.on("mouseenter", "candidate-hit-layer", showPointer);
      liveMap.on("mouseleave", "candidate-layer", clearPointer);
      liveMap.on("mouseleave", "candidate-hit-layer", clearPointer);
      liveMap.on("zoom", updatePinScale);

      bringCandidateLayersToFront();

      window.addEventListener("resize", () => {
        liveMap.setPadding(getMapPadding());
        liveMap.resize();
      });

      setTimeout(() => liveMap.resize(), 500);
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

function updateSelectedMarker(index) {
  if (!liveMap || !liveMap.getLayer?.("candidate-layer")) return;
  liveMap.setPaintProperty("candidate-stem-layer", "circle-color", [
    "case",
    ["==", ["get", "index"], index],
    "#eaeaea",
    ["==", ["get", "rank"], 1],
    "#7dd3a4",
    "#9a9a9a"
  ]);
  liveMap.setPaintProperty("candidate-stem-layer", "circle-radius", [
    "case",
    ["==", ["get", "index"], index],
    6,
    5
  ]);
  liveMap.setPaintProperty("candidate-halo-layer", "circle-color", [
    "case",
    ["==", ["get", "index"], index],
    "#d7d7d7",
    ["==", ["get", "rank"], 1],
    "#10b981",
    "#bdbdbd"
  ]);
  liveMap.setPaintProperty("candidate-halo-layer", "circle-opacity", [
    "case",
    ["==", ["get", "index"], index],
    0.22,
    ["==", ["get", "rank"], 1],
    0.16,
    0.1
  ]);
  liveMap.setPaintProperty("candidate-halo-layer", "circle-radius", [
    "case",
    ["==", ["get", "index"], index],
    40,
    ["==", ["get", "rank"], 1],
    34,
    24
  ]);
  liveMap.setPaintProperty("candidate-layer", "circle-color", [
    "case",
    ["==", ["get", "index"], index],
    "#ffffff",
    ["==", ["get", "rank"], 1],
    "#10b981",
    "#5f6468"
  ]);
  liveMap.setPaintProperty("candidate-layer", "circle-radius", [
    "case",
    ["==", ["get", "index"], index],
    17,
    ["==", ["get", "rank"], 1],
    16,
    12
  ]);
  liveMap.setPaintProperty("candidate-layer", "circle-stroke-width", [
    "case",
    ["==", ["get", "index"], index],
    3,
    ["==", ["get", "rank"], 1],
    2.5,
    1.8
  ]);
  liveMap.setLayoutProperty("candidate-label-layer", "text-size", [
    "case",
    ["==", ["get", "index"], index],
    17,
    14
  ]);
  liveMap.setPaintProperty("candidate-label-layer", "text-color", [
    "case",
    ["==", ["get", "index"], index],
    "#111111",
    "#ffffff"
  ]);
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
  updateSelectedMarker(index);
  updateHtmlMarkerSelection(index);
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  
  const lat = parseFloat(card.dataset.lat);
  const lon = parseFloat(card.dataset.lon);
  
  if (liveMap) {
    console.log(`LOG: Inspecting candidate #${index + 1} at ${lat}, ${lon}`);
    liveMap.flyTo({ 
      center: [lon, lat], 
      zoom: 18, 
      pitch: Math.max(liveMap.getPitch(), 58),
      bearing: liveMap.getBearing() || -25,
      padding: getMapPadding(),
      duration: 1200 
    });
  }
}

function recordCandidateAction(action, index = selectedIndex) {
  if (index < 0) return Promise.resolve(null);
  return postForm("/api/operator/confirm", JSON.stringify({ rank: index + 1, action }));
}

function candidateWeight(item) {
  return item?.posterior_weight ?? item?.posterior ?? item?.score ?? 0;
}

function candidateLat(item) {
  return item?.display_lat ?? item?.lat ?? item?.candidate?.latitude;
}

function candidateLon(item) {
  return item?.display_lon ?? item?.lon ?? item?.candidate?.longitude;
}

function sortedCandidates(result) {
  const candidates = Array.isArray(result?.candidates) ? result.candidates : [];
  return [...candidates].sort((a, b) => candidateWeight(b) - candidateWeight(a));
}

function renderCandidateList(result) {
  const container = byId("results-list");
  if (!container) return;
  container.replaceChildren();

  const candidates = sortedCandidates(result);

  if (candidates.length === 0) {
    container.innerHTML = '<div class="empty-state">No candidates found.</div>';
    return;
  }

  const slice = candidates.slice(0, topLimit);

  slice.forEach((item, idx) => {
    const rank = idx + 1;
    const cand = item || {};
    const lat = candidateLat(cand);
    const lon = candidateLon(cand);
    const card = document.createElement("div");
    card.className = "candidate-card";
    card.dataset.lat = lat;
    card.dataset.lon = lon;
    card.dataset.index = idx;
    
    const coordString = `${Number(lat).toFixed(6)}, ${Number(lon).toFixed(6)}`;
    const mapsUrl = `https://www.google.com/maps/search/?api=1&query=${lat},${lon}`;
    
    // Check source
    const sourceId = cand.match_id || cand.source || "N/A";
    const isExif = sourceId === "exif:gps";
    const sourceLabel = isExif ? "IMAGE METADATA (EXIF)" : `MATCH ID: ${sourceId}`;

    card.innerHTML = `
      <div class="card-top">
        <span class="card-rank">#${rank}</span>
        <span class="card-score">${(candidateWeight(item) * 100).toFixed(1)}%</span>
      </div>
      <div class="card-address">Target Point ${rank}</div>
      <div class="card-sub-wrap">
        <div class="card-sub" title="Click to copy source">Paris, France - ${sourceLabel}</div>
        <button class="source-more" type="button" hidden>Show more</button>
      </div>
      <div class="card-coords-row">
        <div class="card-coords">${coordString}</div>
        <button class="btn-icon-small copy-coords" title="Copy Coordinates">COPY</button>
      </div>
      <div style="display: flex; gap: 8px; margin-top: 8px;" class="card-actions-wrapper">
       <button class="btn-card-action candidate-action" data-action="confirm" type="button">CONFIRM</button>
       <button class="btn-card-action candidate-action" data-action="reject" type="button">REJECT</button>
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

    const sourceText = `Paris, France - ${sourceLabel}`;
    const subEl = card.querySelector(".card-sub");
    const sourceMore = card.querySelector(".source-more");
    if (subEl && sourceMore) {
      requestAnimationFrame(() => {
        const isOverflowing = subEl.scrollHeight > subEl.clientHeight + 1;
        sourceMore.hidden = !isOverflowing;
      });
      sourceMore.addEventListener("click", (e) => {
        e.stopPropagation();
        const expanded = subEl.classList.toggle("expanded");
        sourceMore.textContent = expanded ? "Show less" : "Show more";
      });
      subEl.addEventListener("click", (e) => {
        e.stopPropagation();
        navigator.clipboard.writeText(sourceText).then(() => {
          const hint = card.querySelector(".copied-hint");
          hint.textContent = "Source copied";
          hint.classList.add("visible");
          setTimeout(() => {
            hint.classList.remove("visible");
            hint.textContent = "Copied";
          }, 2000);
        });
      });
    }

    card.querySelector(".open-maps").addEventListener("click", (e) => {
      e.stopPropagation();
      window.open(mapsUrl, "_blank");
    });

    card.querySelectorAll(".candidate-action").forEach((button) => {
      button.addEventListener("click", (e) => {
        e.stopPropagation();
        selectCandidate(idx);
        recordCandidateAction(button.dataset.action || "confirm", idx);
      });
    });

    card.addEventListener("click", () => selectCandidate(idx));

    container.appendChild(card);
  });
  
  selectedIndex = -1;
}

function renderLiveMap(result) {
  ensureLiveMap();
  const fusion = result?.fused_estimate;
  const candidates = sortedCandidates(result);
  
  if (!fusion || candidates.length === 0) return;

  const features = candidates.slice(0, topLimit).map((item, idx) => ({
    type: "Feature",
    geometry: { type: "Point", coordinates: [candidateLon(item), candidateLat(item)] },
    properties: { rank: idx + 1, index: idx, lat: candidateLat(item), lon: candidateLon(item) }
  }));

  const ringRadius =
    (fusion.radius_km ? fusion.radius_km * 1000 : null) ||
    fusion.uncertainty_radius_m ||
    (fusion.ellipse?.major_axis_m ? fusion.ellipse.major_axis_m * 0.6 : null) ||
    result?.result?.geo?.uncertainty_m;
  const ringFeature = (fusion.display_lat && fusion.display_lon && ringRadius) ? {
    type: "Feature",
    geometry: { type: "Polygon", coordinates: [circlePolygon(fusion.display_lat, fusion.display_lon, ringRadius)] }
  } : null;

  liveMapReady.then(() => {
    liveMap.getSource("candidates").setData({ type: "FeatureCollection", features });
    renderHtmlCandidateMarkers(candidates);
    liveMap.getSource("ring").setData({ type: "FeatureCollection", features: ringFeature ? [ringFeature] : [] });
    liveMap.getSource("mean").setData({
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: { type: "Point", coordinates: [fusion.display_lon, fusion.display_lat] } }]
    });
    bringCandidateLayersToFront();

    if (candidates.length > 0) {
      const top = candidates[0];
      if (candidateLat(top) !== undefined && candidateLon(top) !== undefined) {
         liveMap.easeTo({
           center: [candidateLon(top), candidateLat(top)],
           zoom: 15,
           pitch: Math.max(liveMap.getPitch(), 55),
           bearing: liveMap.getBearing() || -25,
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
  const fusion = res.fused_estimate || {};
  const geo = res.fused_estimate || {};

  // Diagnostics
  setMetric("diag-backend", res.source?.filename || "-");
  setMetric("diag-worker", res.runtime?.worker_mode || "-");
  setMetric("diag-tier", geo.tier || "-");
  
  // Radius: prefer non-zero fusion radius, fallback to geo uncertainty
  let radius = "-";
  if (fusion.radius_km !== undefined && fusion.radius_km > 0) {
    radius = `${(fusion.radius_km * 1000).toFixed(1)}m`;
  } else if (fusion.uncertainty_radius_m !== undefined && fusion.uncertainty_radius_m > 0) {
    radius = `${fusion.uncertainty_radius_m.toFixed(1)}m`;
  } else if (geo.uncertainty_m !== undefined) {
    radius = `${geo.uncertainty_m}m`;
  }
  setMetric("diag-radius", radius);
  
  // Model Status: Check safe_demo or backend name
  const isDemo = Boolean(res.safe_demo);
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
  selectedLightboxDetectionIndex = 0;
  renderLightboxIntel();
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
  clearCandidateMarkers();
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
        selectedLightboxDetectionIndex = 0;
        lightboxImg.onload = renderLightboxIntel;
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
  expandBtn.addEventListener("click", () => {
    modal.classList.add("active");
    renderLightboxIntel();
  });
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
  const tiltHandle = byId("map-tilt-handle");
  const tiltLabel = byId("map-tilt-label");

  const updateTiltLabel = () => {
    if (!tiltLabel || !liveMap) return;
    const pitch = Math.round(liveMap.getPitch());
    const bearing = Math.round(liveMap.getBearing());
    tiltLabel.textContent = `${pitch}/${bearing}`;
  };

  const setViewAngle = ({ pitch = liveMap?.getPitch() || 0, bearing = liveMap?.getBearing() || 0 }, duration = 180) => {
    if (!liveMap) return;
    liveMap.easeTo({
      pitch: Math.max(minPitch, Math.min(maxPitch, pitch)),
      bearing,
      duration
    });
    window.setTimeout(updateTiltLabel, duration + 20);
  };

  if (tiltHandle) {
    let dragging = false;
    let suppressClick = false;
    let startX = 0;
    let startY = 0;
    let startPitch = 0;
    let startBearing = 0;

    tiltHandle.addEventListener("click", () => {
      if (suppressClick) {
        suppressClick = false;
        return;
      }
      if (liveMap?.getPitch() || Math.round(liveMap?.getBearing() || 0) !== 0) {
        setViewAngle({ pitch: 0, bearing: 0 }, 280);
      } else {
        setViewAngle({ pitch: 45, bearing: -25 }, 280);
      }
    });
    tiltHandle.addEventListener("pointerdown", (e) => {
      if (!liveMap) return;
      dragging = false;
      startX = e.clientX;
      startY = e.clientY;
      startPitch = liveMap.getPitch();
      startBearing = liveMap.getBearing();
      tiltHandle.setPointerCapture(e.pointerId);
      tiltHandle.classList.add("dragging");
    });
    tiltHandle.addEventListener("pointermove", (e) => {
      if (!tiltHandle.hasPointerCapture(e.pointerId) || !liveMap) return;
      const deltaX = e.clientX - startX;
      const deltaY = startY - e.clientY;
      if (Math.abs(deltaY) > 2 || Math.abs(deltaX) > 2) dragging = true;
      liveMap.setPitch(Math.max(minPitch, Math.min(maxPitch, startPitch + deltaY * 0.45)));
      liveMap.setBearing(startBearing + deltaX * 0.55);
      updateTiltLabel();
    });
    const endDrag = (e) => {
      if (tiltHandle.hasPointerCapture(e.pointerId)) {
        tiltHandle.releasePointerCapture(e.pointerId);
      }
      tiltHandle.classList.remove("dragging");
      if (dragging) {
        suppressClick = true;
        e.preventDefault();
        e.stopPropagation();
      }
    };
    tiltHandle.addEventListener("pointerup", endDrag);
    tiltHandle.addEventListener("pointercancel", endDrag);
  }

  liveMapReady?.then(updateTiltLabel);
  
  byId("map-reset-paris").addEventListener("click", () => {
    liveMap?.easeTo({ center: parisCenter, zoom: 11, pitch: 0, bearing: 0, padding: getMapPadding(), duration: 1400 });
    window.setTimeout(updateTiltLabel, 1420);
  });
  byId("map-reset-globe").addEventListener("click", () => {
    liveMap?.easeTo({ center: globeCenter, zoom: globeZoom, pitch: 0, bearing: 0, padding: getMapPadding(), duration: 1800 });
    window.setTimeout(updateTiltLabel, 1820);
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
  setupOperatorActions();
  
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

function getImageDetections() {
  const direct = Array.isArray(lastResult?.detections) ? lastResult.detections : [];
  const nested = Array.isArray(lastResult?.result?.detections) ? lastResult.result.detections : [];
  return direct.length ? direct : nested;
}

function detectionPoints(det) {
  const obb = Array.isArray(det?.obb) ? det.obb : [];
  if (obb.length !== 4) return [];
  const points = obb
    .map((pt) => [Number(pt?.[0]), Number(pt?.[1])])
    .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  return points.length === 4 ? points : [];
}

function formatDetectionDetail(det) {
  const score = Number.isFinite(Number(det?.confidence)) ? `${Math.round(Number(det.confidence) * 100)}%` : "-";
  const heading = Number.isFinite(Number(det?.heading_deg)) ? `heading ${Number(det.heading_deg).toFixed(1)} deg` : null;
  const shadow = Number.isFinite(Number(det?.shadow_azimuth_deg)) ? `shadow ${Number(det.shadow_azimuth_deg).toFixed(1)} deg` : null;
  return [score, heading, shadow].filter(Boolean).join(" | ");
}

function renderLightboxIntel() {
  const img = byId("lightbox-img");
  const overlay = byId("lightbox-overlay");
  const intel = byId("lightbox-intel");
  if (!img || !overlay || !intel) return;

  const detections = getImageDetections();
  const clues = Array.isArray(lastResult?.clues) ? lastResult.clues : [];
  const width = img.naturalWidth || 1;
  const height = img.naturalHeight || 1;

  overlay.setAttribute("viewBox", `0 0 ${width} ${height}`);
  overlay.replaceChildren();

  if (detections.length && selectedLightboxDetectionIndex >= detections.length) {
    selectedLightboxDetectionIndex = 0;
  }

  const activeDetection = detections[selectedLightboxDetectionIndex];
  const activePoints = detectionPoints(activeDetection);
  if (activePoints.length) {
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("points", activePoints.map(([x, y]) => `${x},${y}`).join(" "));
    polygon.setAttribute("class", "det-obb active");
    overlay.appendChild(polygon);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    const x = Math.min(...activePoints.map(([px]) => px));
    const y = Math.min(...activePoints.map(([, py]) => py));
    label.setAttribute("x", String(Math.max(8, x)));
    label.setAttribute("y", String(Math.max(18, y - 8)));
    label.setAttribute("class", "det-label");
    label.textContent = `${selectedLightboxDetectionIndex + 1}. ${activeDetection?.label || "object"}`;
    overlay.appendChild(label);
  }

  intel.replaceChildren();
  const title = document.createElement("div");
  title.className = "intel-title";
  title.textContent = detections.length ? `Detected Objects (${detections.length})` : "Detected Objects";
  intel.appendChild(title);

  const addIntelRow = (name, detail, idx = null) => {
    const row = document.createElement(idx === null ? "div" : "button");
    row.className = `intel-row${idx === selectedLightboxDetectionIndex ? " active" : ""}`;
    if (idx !== null) {
      row.type = "button";
      row.addEventListener("click", () => {
        selectedLightboxDetectionIndex = idx;
        renderLightboxIntel();
      });
    }
    const strong = document.createElement("strong");
    strong.textContent = name;
    const span = document.createElement("span");
    span.textContent = detail;
    row.append(strong, span);
    intel.appendChild(row);
  };

  detections.forEach((det, idx) => {
    addIntelRow(`${idx + 1}. ${det?.label || "object"}`, formatDetectionDetail(det), idx);
  });

  if (!detections.length && clues.length) {
    clues.forEach((clue, idx) => {
      const score = Number.isFinite(Number(clue?.score)) ? `${Math.round(Number(clue.score) * 100)}%` : "-";
      addIntelRow(`${idx + 1}. ${clue?.name || "clue"}`, `${score} | ${clue?.description || ""}`);
    });
  }

  if (!detections.length && !clues.length) {
    const empty = document.createElement("div");
    empty.className = "intel-empty";
    empty.textContent = "No object detections available for this image.";
    intel.appendChild(empty);
  }
}

function setupOperatorActions() {
    const confirmBtn = byId("btn-confirm-cand");
    const rejectBtn = byId("btn-reject-cand");
    const noteInput = byId("operator-note-input");
    const saveNoteBtn = byId("btn-save-note");
    const exportBtn = byId("btn-export-session");

    if (confirmBtn) {
        confirmBtn.addEventListener("click", () => {
             recordCandidateAction("confirm");
        });
    }
    if (rejectBtn) {
        rejectBtn.addEventListener("click", () => {
             recordCandidateAction("reject");
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
