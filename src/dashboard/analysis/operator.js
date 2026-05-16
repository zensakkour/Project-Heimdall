import { byId, formatUtcNowLabel, normalizeError, postForm } from "./shared.js";

const profileStorageKey = "heimdallProfile";
let isDroppingPin = false;
let isStreetWalkMode = false;
let droppedPinLocation = null;
let currentManualPinMarker = null;
let operatorPinMarkers = [];

// Street View interactive viewer state
let svState = {
    imageId: null, lat: null, lon: null, heading: null,
    sequence: null, seqPos: null, seqTotal: null,
    candidates: [], minimap: null, minimapReady: null, minimapMarkers: [],
};
const SV_CAND_COLORS = ["#f2f2f2", "#c9c9c9", "#9f9f9f", "#777777", "#555555"];
const profileOptions = [
  { value: "paris", label: "Paris (Standard)" },
  { value: "paris_test", label: "Paris (Test)" },
];

let loadedSessionId = null;

let noteMarkers = [];
let activeProfile = "paris";
let liveMap = null;
let liveMapReady = null;
let topLimit = 3;
let lastResult = null;
let selectedIndex = -1;
let candidateMarkers = [];
let activeCandidateItems = [];
let candidatePinsPopulated = false;
let selectedLightboxDetectionIndex = 0;

const parisCenter = [2.3522, 48.8566];
const globeCenter = [10, 20];
const globeZoom = 1.8;
const minPitch = 0;
const maxPitch = 70;
const htmlPinMinZoom = 11;
const maxCandidateLayers = 10;
const emptyFeatureCollection = { type: "FeatureCollection", features: [] };
const mapStyleUrl = "https://tiles.openfreemap.org/styles/dark";
const globePitchResetZoom = 4;
const candidateClusterRadiusPx = 42;

/* --- Map Core --- */

function getMapPadding() {
  return { left: 0, right: 0, top: 0, bottom: 0 };
}

function centeredCameraOptions(options = {}) {
  return {
    ...options,
    padding: getMapPadding(),
    offset: [0, 0]
  };
}

function easeToCentered(options) {
  if (!liveMap) return;
  liveMap.setPadding(getMapPadding());
  liveMap.easeTo(centeredCameraOptions(options));
}

function flyToCentered(options) {
  if (!liveMap) return;
  liveMap.setPadding(getMapPadding());
  liveMap.flyTo(centeredCameraOptions(options));
}

function zoomCentered(delta) {
  if (!liveMap) return;
  const currentZoom = liveMap.getZoom();
  const targetZoom = Math.max(liveMap.getMinZoom(), Math.min(liveMap.getMaxZoom(), currentZoom + delta));
  easeToCentered({
    center: liveMap.getCenter(),
    zoom: targetZoom,
    pitch: targetZoom <= globePitchResetZoom ? 0 : liveMap.getPitch(),
    duration: 280
  });
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
    "candidate-hit-layer",
    ...candidateRankLayerIds()
  ].forEach(moveLayerToTop);
}

function clearCandidateMarkers() {
  candidateMarkers.forEach((marker) => marker.remove());
  candidateMarkers = [];
  candidatePinsPopulated = false;
}

function updateHtmlMarkerSelection(index) {
  candidateMarkers.forEach((marker) => {
    const el = marker.getElement();
    const idx = Number(el.dataset.index);
    const isSelected = idx === index;
    const color = CAND_COLORS[idx] || "#4a5568";
    el.style.background = isSelected ? "#ffffff" : color;
    el.style.color = isSelected ? color : "#ffffff";
    el.style.borderColor = isSelected ? color : "rgba(255,255,255,0.85)";
    el.style.boxShadow = isSelected
      ? `0 0 0 3px ${color}55, 0 2px 8px rgba(0,0,0,0.55)`
      : "0 2px 8px rgba(0,0,0,0.55)";
  });
}

function updatePinScale() {
  // No HTML markers — GeoJSON layers handle all rendering. No-op.
}

function fusedMapCoord(result) {
  const fusion = result?.fused_estimate || {};
  const lat = Number(fusion.display_lat ?? fusion.lat);
  const lon = Number(fusion.display_lon ?? fusion.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}

function operatorMapCoord(item) {
  return numericCandidateCoord(item);
}

const CAND_COLORS = ["#10b981", "#6366f1", "#f59e0b", "#ef4444", "#8b5cf6"];

function renderCandidatePins(candidateEntries) {
  clearCandidateMarkers();
  if (!liveMap) return;

  candidateEntries.slice(0, topLimit + 1).forEach((entry, idx) => {
    const item = entry.item || entry;
    const listIndex = Number.isFinite(entry.listIndex) ? entry.listIndex : idx;
    const rank = Number.isFinite(entry.rank) ? entry.rank : listIndex + 1;
    const coord = numericCandidateCoord(item);
    if (!coord) return;

    const color = entry.temporary ? "#ffffff" : CAND_COLORS[listIndex] || "#4a5568";
    const size = entry.temporary || listIndex === 0 ? 28 : 24;

    const el = document.createElement("div");
    el.className = "candidate-pin-marker";
    el.dataset.index = String(listIndex);
    Object.assign(el.style, {
      width: `${size}px`,
      height: `${size}px`,
      background: color,
      border: entry.temporary ? "3px solid #10b981" : "2.5px solid rgba(255,255,255,0.85)",
      borderRadius: "50%",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: listIndex === 0 ? "13px" : "11px",
      fontWeight: "700",
      color: entry.temporary ? "#111111" : "#ffffff",
      cursor: "pointer",
      boxShadow: entry.temporary ? "0 0 0 4px rgba(16,185,129,0.28), 0 2px 10px rgba(0,0,0,0.6)" : "0 2px 8px rgba(0,0,0,0.55)",
      userSelect: "none",
    });
    el.textContent = String(rank);
    el.title = entry.temporary ? `Selected candidate ${rank}` : `Candidate ${rank}`;

    el.addEventListener("click", (e) => {
      e.stopPropagation();
      selectCandidate(listIndex);
    });

    const marker = new maplibregl.Marker({ element: el, anchor: "center" })
      .setLngLat([coord.lon, coord.lat])
      .addTo(liveMap);
    candidateMarkers.push(marker);
  });
}

function refreshCandidateMarkers() {
  // Markers are permanent — nothing to refresh.
}

function markerCandidateIndices(el) {
  return (el.dataset.indices || "")
    .split(",")
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
}

function numericCandidateCoord(item) {
  const lat = Number(candidateMapLat(item));
  const lon = Number(candidateMapLon(item));
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}

function candidateRankLayerIds() {
  const ids = [];
  for (let i = 0; i < maxCandidateLayers; i += 1) {
    ids.push(`candidate-rank-halo-${i}`, `candidate-rank-dot-${i}`, `candidate-rank-label-${i}`, `candidate-rank-hit-${i}`);
  }
  return ids;
}

function rankLayerStyle(index, selected = selectedIndex) {
  const isSelected = index === selected;
  const isTop = index === 0;
  return {
    dotColor: isSelected ? "#ffffff" : isTop ? "#10b981" : "#5f6468",
    dotText: isSelected ? "#111111" : "#ffffff",
    haloColor: isSelected ? "#d7d7d7" : isTop ? "#10b981" : "#bdbdbd",
    dotRadius: isSelected ? 9 : isTop ? 8 : 7,
    haloRadius: isSelected ? 18 : isTop ? 16 : 14,
    strokeWidth: isSelected ? 3 : isTop ? 2.5 : 1.8,
    haloOpacity: isSelected ? 0.28 : isTop ? 0.22 : 0.14
  };
}

function addCandidateRankLayers() {
  for (let i = 0; i < maxCandidateLayers; i += 1) {
    const offset = [0, 0];
    const style = rankLayerStyle(i);
    const filter = ["==", ["get", "index"], i];

    liveMap.addLayer({
      id: `candidate-rank-halo-${i}`,
      type: "circle",
      source: "candidates",
      filter,
      paint: {
        "circle-color": style.haloColor,
        "circle-opacity": style.haloOpacity,
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 1.5, style.haloRadius, 14, style.haloRadius + 8],
        "circle-stroke-color": "#ffffff",
        "circle-stroke-opacity": 0.1,
        "circle-stroke-width": 1,
        "circle-translate": offset,
        "circle-translate-anchor": "viewport"
      }
    });
    liveMap.addLayer({
      id: `candidate-rank-dot-${i}`,
      type: "circle",
      source: "candidates",
      filter,
      paint: {
        "circle-color": style.dotColor,
        "circle-opacity": 0.96,
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 1.5, style.dotRadius, 14, style.dotRadius + 7],
        "circle-stroke-color": "#f4f4f4",
        "circle-stroke-width": style.strokeWidth,
        "circle-translate": offset,
        "circle-translate-anchor": "viewport"
      }
    });
    liveMap.addLayer({
      id: `candidate-rank-label-${i}`,
      type: "symbol",
      source: "candidates",
      filter,
      layout: {
        "text-field": ["to-string", ["get", "rank"]],
        "text-size": i === selectedIndex ? 15 : 13,
        "text-font": ["Open Sans Bold"],
        "text-allow-overlap": true,
        "text-ignore-placement": true
      },
      paint: {
        "text-color": style.dotText,
        "text-halo-color": "rgba(0,0,0,0)",
        "text-halo-width": 0,
        "text-opacity": ["interpolate", ["linear"], ["zoom"], 2.5, 0, 4, 1],
        "text-translate": offset,
        "text-translate-anchor": "viewport"
      }
    });
    liveMap.addLayer({
      id: `candidate-rank-hit-${i}`,
      type: "circle",
      source: "candidates",
      filter,
      paint: {
        "circle-color": "#ffffff",
        "circle-opacity": 0.01,
        "circle-radius": 24,
        "circle-translate": offset,
        "circle-translate-anchor": "viewport"
      }
    });
  }
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
          "circle-radius": 0,
          "circle-opacity": 0
        }
      });
      liveMap.addLayer({
        id: "candidate-halo-layer",
        type: "circle",
        source: "candidates",
        paint: {
          "circle-color": ["case", ["==", ["get", "index"], selectedIndex], "#d7d7d7", ["==", ["get", "rank"], 1], "#10b981", "#bdbdbd"],
          "circle-opacity": 0,
          "circle-radius": 0,
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
          "circle-radius": 0,
          "circle-stroke-color": "#f4f4f4",
          "circle-stroke-width": ["case", ["==", ["get", "index"], selectedIndex], 3, ["==", ["get", "rank"], 1], 2.5, 1.8],
          "circle-opacity": 0
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
          "text-halo-width": 0,
          "text-opacity": 0
        }
      });
      liveMap.addLayer({
        id: "candidate-hit-layer",
        type: "circle",
        source: "candidates",
        paint: {
          "circle-color": "#ffffff",
          "circle-opacity": 0.01,
          "circle-radius": 0,
          "circle-stroke-opacity": 0
        }
      });
      addCandidateRankLayers();
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
      for (let i = 0; i < maxCandidateLayers; i += 1) {
        liveMap.on("click", `candidate-rank-dot-${i}`, handleCandidateMarkerClick);
        liveMap.on("click", `candidate-rank-hit-${i}`, handleCandidateMarkerClick);
      }

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
      for (let i = 0; i < maxCandidateLayers; i += 1) {
        liveMap.on("mouseenter", `candidate-rank-dot-${i}`, showPointer);
        liveMap.on("mouseenter", `candidate-rank-hit-${i}`, showPointer);
        liveMap.on("mouseleave", `candidate-rank-dot-${i}`, clearPointer);
        liveMap.on("mouseleave", `candidate-rank-hit-${i}`, clearPointer);
      }
      liveMap.on("zoomend", () => {
        if (liveMap.getZoom() <= globePitchResetZoom && liveMap.getPitch() !== 0) {
          easeToCentered({ center: liveMap.getCenter(), pitch: 0, duration: 180 });
        }
      });

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
  liveMap.setPaintProperty("candidate-stem-layer", "circle-radius", 0);
  liveMap.setPaintProperty("candidate-stem-layer", "circle-opacity", 0);
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
    0.28,
    ["==", ["get", "rank"], 1],
    0.22,
    0.14
  ]);
  liveMap.setPaintProperty("candidate-halo-layer", "circle-radius", [
    "interpolate",
    ["linear"],
    ["zoom"],
    1.5,
    ["case", ["==", ["get", "index"], index], 10, ["==", ["get", "rank"], 1], 9, 8],
    5,
    ["case", ["==", ["get", "index"], index], 13, ["==", ["get", "rank"], 1], 12, 10],
    14,
    ["case", ["==", ["get", "index"], index], 22, ["==", ["get", "rank"], 1], 20, 16]
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
    "interpolate",
    ["linear"],
    ["zoom"],
    1.5,
    ["case", ["==", ["get", "index"], index], 5.5, ["==", ["get", "rank"], 1], 5, 4.5],
    5,
    ["case", ["==", ["get", "index"], index], 7.5, ["==", ["get", "rank"], 1], 7, 6],
    14,
    ["case", ["==", ["get", "index"], index], 16, ["==", ["get", "rank"], 1], 15, 12]
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

  for (let i = 0; i < maxCandidateLayers; i += 1) {
    if (!liveMap.getLayer(`candidate-rank-dot-${i}`)) continue;
    const style = rankLayerStyle(i, index);
    liveMap.setPaintProperty(`candidate-rank-halo-${i}`, "circle-color", style.haloColor);
    liveMap.setPaintProperty(`candidate-rank-halo-${i}`, "circle-opacity", style.haloOpacity);
    liveMap.setPaintProperty(`candidate-rank-halo-${i}`, "circle-radius", ["interpolate", ["linear"], ["zoom"], 1.5, style.haloRadius, 14, style.haloRadius + 8]);
    liveMap.setPaintProperty(`candidate-rank-dot-${i}`, "circle-color", style.dotColor);
    liveMap.setPaintProperty(`candidate-rank-dot-${i}`, "circle-radius", ["interpolate", ["linear"], ["zoom"], 1.5, style.dotRadius, 14, style.dotRadius + 7]);
    liveMap.setPaintProperty(`candidate-rank-dot-${i}`, "circle-stroke-width", style.strokeWidth);
    liveMap.setLayoutProperty(`candidate-rank-label-${i}`, "text-size", i === index ? 15 : 13);
    liveMap.setPaintProperty(`candidate-rank-label-${i}`, "text-color", style.dotText);
  }
}

function loadNoteForTarget(targetType, rankOrLat, lon) {
    const noteInput = byId("operator-note-input");
    if (!noteInput) return;

    noteInput.value = ""; // Clear by default

    if (lastResult && lastResult.notes) {
        const note = lastResult.notes.find(n => {
            if (targetType === "candidate" && n.target_type === "candidate") {
                return n.rank === rankOrLat;
            } else if (targetType === "manual_pin" && n.target_type === "manual_pin") {
                return Math.abs(n.lat - rankOrLat) < 0.0001 && Math.abs(n.lon - lon) < 0.0001;
            } else if (targetType === "note_id") {
                return n.note_id === rankOrLat;
            }
            return false;
        });

        if (note) {
            noteInput.value = note.text;
            return;
        }
    }

    fetch("/api/operator/session").then(r => r.json()).then(data => {
        if (!data.notes) return;

        const note = data.notes.find(n => {
            if (targetType === "candidate" && n.target_type === "candidate") {
                return n.rank === rankOrLat;
            } else if (targetType === "manual_pin" && n.target_type === "manual_pin") {
                // approximate matching for lat/lon floats
                return Math.abs(n.lat - rankOrLat) < 0.0001 && Math.abs(n.lon - lon) < 0.0001;
            } else if (targetType === "note_id") {
                return n.note_id === rankOrLat;
            }
            return false;
        });

        if (note) {
            noteInput.value = note.text;
        }
    }).catch(err => console.error("Failed to fetch session for notes:", err));
}

function selectCandidate(index) {
  droppedPinLocation = null;
  const cards = document.querySelectorAll(".candidate-card");
  if (!cards.length) return;
  
  if (index < 0) index = 0;
  if (index >= cards.length) index = cards.length - 1;
  
  const card = cards[index];
  const isAlreadySelected = card.classList.contains("active");

  cards.forEach(c => c.classList.remove("active"));

  if (isAlreadySelected) {
      selectedIndex = -1;
      if (lastResult) renderLiveMap(lastResult);
      updateSelectedMarker(-1);
      updateHtmlMarkerSelection(-1);
      const noteInput = byId("operator-note-input");
      if (noteInput) noteInput.value = "";
      return;
  }

  selectedIndex = index;
  if (lastResult) renderLiveMap(lastResult);
  card.classList.add("active");
  updateSelectedMarker(index);
  updateHtmlMarkerSelection(index);
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  
  const lat = parseFloat(card.dataset.lat);
  const lon = parseFloat(card.dataset.lon);
  
  if (liveMap) {
    console.log(`LOG: Inspecting candidate #${index + 1} at ${lat}, ${lon}`);
    flyToCentered({
      center: [lon, lat], 
      zoom: 18, 
      pitch: 0,
      bearing: 0,
      duration: 1200 
    });
  }

  loadNoteForTarget("candidate", index + 1);
}

function recordCandidateAction(action, index = selectedIndex) {
  if (index < 0) return Promise.resolve(null);
  return postForm("/api/operator/confirm", JSON.stringify({ index, rank: index + 1, action }));
}

function candidateWeight(item) {
  return item?.posterior_weight ?? item?.posterior ?? item?.score ?? 0;
}

function candidateAccepted(item) {
  return item?.accepted === true || item?.status === "accepted" || item?.operator_status === "accepted";
}

function reviewableSessionCandidates(data) {
  const candidates = sortedCandidates(data);
  const accepted = candidates.filter(candidateAccepted);
  return accepted.length > 0 ? accepted : candidates;
}

function candidateLat(item) {
  return item?.lat ?? item?.candidate?.latitude ?? item?.display_lat;
}

function candidateLon(item) {
  return item?.lon ?? item?.candidate?.longitude ?? item?.display_lon;
}

function candidateDisplayLat(item) {
  return item?.display_lat ?? candidateLat(item);
}

function candidateDisplayLon(item) {
  return item?.display_lon ?? candidateLon(item);
}

function candidateMapLat(item) {
  return candidateLat(item) ?? item?.display_lat;
}

function candidateMapLon(item) {
  return candidateLon(item) ?? item?.display_lon;
}

function sortedCandidates(result) {
  const candidates = Array.isArray(result?.candidates) ? result.candidates : [];
  return [...candidates].sort((a, b) => candidateWeight(b) - candidateWeight(a));
}

function mapCandidateEntries(candidates) {
  const entries = candidates.map((item, listIndex) => ({
    item,
    listIndex,
    rank: listIndex + 1,
    temporary: false
  }));

  if (!loadedSessionId) return entries.slice(0, topLimit);

  const accepted = entries.filter(entry => candidateAccepted(entry.item));
  const acceptedEntries = (accepted.length > 0 ? accepted : entries).slice(0, topLimit);
  const selectedEntry = Number.isFinite(selectedIndex) && selectedIndex >= 0 ? entries[selectedIndex] : null;
  if (selectedEntry && !acceptedEntries.some(entry => entry.listIndex === selectedEntry.listIndex)) {
    acceptedEntries.push({ ...selectedEntry, temporary: true });
  }
  return acceptedEntries;
}

function visibleSessionEntries() {
  return Array.from(_overlayMarkers.entries())
    .filter(([sessionId, entry]) => {
      if (!entry?.visible || !entry?.data) return false;
      return sessionId !== loadedSessionId || isCurrentSessionHidden();
    })
    .map(([sessionId, entry]) => ({ sessionId, ...entry }));
}

function sessionDisplayName(sessionId, data = {}) {
  const summary = _allSessions.find(s => s.session_id === sessionId) || {};
  return summary.custom_name || data.custom_name || summary.display_name || data.display_name || (sessionId || "").slice(0, 10);
}

function appendSessionCandidateCards(container) {
  const entries = visibleSessionEntries();
  if (!entries.length) return 0;

  let appended = 0;
  entries.forEach(({ sessionId, data, color }) => {
    const candidates = reviewableSessionCandidates(data).slice(0, topLimit);
    if (!candidates.length) return;

    const divider = document.createElement("div");
    divider.className = "session-candidate-divider";
    divider.style.setProperty("--session-color", color);
    divider.innerHTML = `<span></span><strong>${sessionDisplayName(sessionId, data)}</strong>`;
    container.appendChild(divider);

    candidates.forEach((cand, idx) => {
      const coord = numericCandidateCoord(cand);
      if (!coord) return;
      const rank = idx + 1;
      const card = document.createElement("div");
      card.className = "candidate-card session-candidate-card";
      card.style.setProperty("--session-color", color);
      card.dataset.lat = coord.lat;
      card.dataset.lon = coord.lon;

      const coordString = `${coord.lat.toFixed(6)}, ${coord.lon.toFixed(6)}`;
      const mapsUrl = `https://www.google.com/maps/search/?api=1&query=${coord.lat},${coord.lon}`;
      const sourceId = cand.match_id || cand.source || "N/A";
      card.innerHTML = `
        <div class="card-top">
          <span class="card-rank">S${rank}</span>
          <span class="card-score">${candidateAccepted(cand) ? "ACCEPTED" : `${(candidateWeight(cand) * 100).toFixed(1)}%`}</span>
        </div>
        <div class="card-address">Session Point ${rank}</div>
        <div class="card-sub">Session ${sessionDisplayName(sessionId, data)} - ${sourceId}</div>
        <div class="card-coords-row">
          <div class="card-coords">${coordString}</div>
          <button class="btn-icon-small copy-coords" title="Copy Coordinates">COPY</button>
        </div>
        <div class="card-action-row">
          <button class="btn-card-action open-maps">Open in Google Maps</button>
          <button class="btn-card-action street-view-btn street-view-btn-accent" data-lat="${coord.lat}" data-lon="${coord.lon}">Street View</button>
        </div>
        <div class="candidate-card-actions">
          <button class="btn-accept-candidate" type="button" title="Accept this session point">
            <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>
            ${candidateAccepted(cand) ? "Accepted" : "Accept"}
          </button>
          <button class="btn-refuse-candidate" type="button" title="Remove this point from the session">
            <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
            Remove
          </button>
        </div>
      `;

      card.querySelector(".copy-coords")?.addEventListener("click", (e) => {
        e.stopPropagation();
        navigator.clipboard.writeText(coordString);
      });
      card.querySelector(".open-maps")?.addEventListener("click", (e) => {
        e.stopPropagation();
        window.open(mapsUrl, "_blank");
      });
      card.querySelector(".btn-accept-candidate")?.addEventListener("click", (e) => {
        e.stopPropagation();
        acceptSessionCandidate(sessionId, idx, rank);
      });
      card.querySelector(".btn-refuse-candidate")?.addEventListener("click", (e) => {
        e.stopPropagation();
        removeSessionCandidate(sessionId, idx, rank);
      });
      card.addEventListener("click", () => {
        document.querySelectorAll(".candidate-card").forEach(c => c.classList.remove("active"));
        card.classList.add("active");
        flyToCentered({ center: [coord.lon, coord.lat], zoom: 16, pitch: 0, bearing: 0, duration: 900 });
      });
      container.appendChild(card);
      appended += 1;
    });
  });
  return appended;
}

function renderCandidateList(result) {
  const container = byId("results-list");
  if (!container) return;
  container.replaceChildren();

  const candidates = sortedCandidates(result);
  const currentHidden = isCurrentSessionHidden();
  const hasVisibleSessions = visibleSessionEntries().length > 0;

  if ((currentHidden || candidates.length === 0) && !hasVisibleSessions) {
    container.innerHTML = '<div class="empty-state">No candidates found.</div>';
    return;
  }

  const slice = currentHidden ? [] : candidates.slice(0, topLimit);

  slice.forEach((item, idx) => {
    const rank = idx + 1;
    const cand = item || {};
    const mapCoord = operatorMapCoord(cand);
    const lat = mapCoord?.lat ?? candidateMapLat(cand);
    const lon = mapCoord?.lon ?? candidateMapLon(cand);
    const card = document.createElement("div");
    const isAccepted = candidateAccepted(cand);
    card.className = `candidate-card${isAccepted ? " accepted-candidate" : ""}`;
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
        <span class="card-score">${isAccepted ? "ACCEPTED" : `${(candidateWeight(item) * 100).toFixed(1)}%`}</span>
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
      <div class="card-action-row">
        <button class="btn-card-action open-maps">Open in Google Maps</button>
        <button class="btn-card-action street-view-btn street-view-btn-accent" data-lat="${lat}" data-lon="${lon}">Street View</button>
      </div>
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

    // Refuse candidate row
    const actionsRow = document.createElement("div");
    actionsRow.className = "candidate-card-actions";
    actionsRow.innerHTML = `
      <button class="btn-accept-candidate" type="button" title="Accept this geotag">
        <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>
        ${isAccepted ? "Accepted" : "Accept"}
      </button>
      <button class="btn-refuse-candidate" type="button" title="Remove this candidate from the analysis">
        <svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
        Remove
      </button>
    `;
    actionsRow.querySelector(".btn-refuse-candidate").addEventListener("click", (e) => {
      e.stopPropagation();
      refuseCandidate(idx, rank);
    });
    actionsRow.querySelector(".btn-accept-candidate").addEventListener("click", (e) => {
      e.stopPropagation();
      acceptCandidate(idx, rank);
    });
    card.appendChild(actionsRow);

    container.appendChild(card);
  });

  appendSessionCandidateCards(container);

  // Attach street view handlers
  container.querySelectorAll(".street-view-btn").forEach(btn => {
      btn.addEventListener("click", (e) => {
          e.stopPropagation();
          const lat = parseFloat(btn.dataset.lat);
          const lon = parseFloat(btn.dataset.lon);
          openStreetView(lat, lon);
      });
  });

  selectedIndex = -1;
  const noteInput = byId("operator-note-input");
  if (noteInput) noteInput.value = "";
}

// ── Street View Math Helpers ─────────────────────────────────────

function svBearingTo(fromLat, fromLon, toLat, toLon) {
    const dLon = (toLon - fromLon) * Math.PI / 180;
    const lat1 = fromLat * Math.PI / 180;
    const lat2 = toLat * Math.PI / 180;
    const y = Math.sin(dLon) * Math.cos(lat2);
    const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
    return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

function svHaversineKm(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ── Street View State Helpers ─────────────────────────────────────

function svSetLoading(active) {
    const loading = byId("sv-loading");
    const img = byId("sv-img");
    const err = byId("sv-error");
    if (loading) loading.style.display = active ? "flex" : "none";
    if (img) img.style.visibility = active ? "hidden" : "visible";
    if (err) err.style.display = "none";
}

function svSetError(msg) {
    const loading = byId("sv-loading");
    const img = byId("sv-img");
    const err = byId("sv-error");
    if (loading) loading.style.display = "none";
    if (img) img.style.visibility = "hidden";
    if (err) { err.style.display = "flex"; err.textContent = msg; }
    // Clear canvas
    const canvas = byId("sv-overlay-canvas");
    if (canvas) { const ctx = canvas.getContext("2d"); ctx?.clearRect(0, 0, canvas.width, canvas.height); }
}

function svClose() {
    byId("street-view-modal")?.classList.remove("active");
    svResetZoom();
    if (svState.minimapMarkers) {
        svState.minimapMarkers.forEach(m => m.remove());
        svState.minimapMarkers = [];
    }
}

// ── Zoom + Pan ───────────────────────────────────────────────────
let svZoom = 1;
let svPanX = 0; // offset from natural center (in CSS px)
let svPanY = 0;
let svIsDragging = false;
let svDragStartX = 0, svDragStartY = 0, svDragStartPanX = 0, svDragStartPanY = 0;
const SV_ZOOM_MIN = 0.5, SV_ZOOM_MAX = 5, SV_ZOOM_STEP = 0.25;

function svSetTransform(animated) {
    const img = byId("sv-img");
    if (!img) return;
    img.style.transition = animated ? "transform 0.12s ease" : "none";
    img.style.transform = `translate(${svPanX}px, ${svPanY}px) scale(${svZoom})`;
    const lbl = byId("sv-zoom-label");
    if (lbl) lbl.textContent = `${Math.round(svZoom * 100)}%`;
}

// Zoom toward a point. cursorX/Y are in wrap-relative px; omit to zoom to center.
function svApplyZoom(newZ, cursorX, cursorY) {
    const clamped = Math.max(SV_ZOOM_MIN, Math.min(SV_ZOOM_MAX, newZ));
    if (cursorX !== undefined && cursorY !== undefined) {
        const wrap = byId("sv-image-wrap");
        if (wrap) {
            const wRect = wrap.getBoundingClientRect();
            // cursor relative to wrap center (= natural image center)
            const cx = cursorX - wRect.width  / 2;
            const cy = cursorY - wRect.height / 2;
            // same content-space point stays under cursor after zoom
            const contentX = (cx - svPanX) / svZoom;
            const contentY = (cy - svPanY) / svZoom;
            svPanX = cx - contentX * clamped;
            svPanY = cy - contentY * clamped;
        }
    }
    svZoom = clamped;
    svSetTransform(cursorX === undefined); // animate only for button/key presses
}

function svResetZoom() {
    svZoom = 1; svPanX = 0; svPanY = 0;
    svSetTransform(true);
}

// ── Street View Core ──────────────────────────────────────────────

function openStreetView(lat, lon) {
    const modal = byId("street-view-modal");
    if (!modal) return;

    // Collect top-5 candidates for overlay rendering
    const rawCands = sortedCandidates(lastResult).slice(0, 5);
    svState.candidates = rawCands.map((c, i) => ({
        rank: i + 1,
        lat: parseFloat(candidateMapLat(c)),
        lon: parseFloat(candidateMapLon(c)),
        score: candidateWeight(c),
        color: SV_CAND_COLORS[i] || "#888",
    }));

    modal.classList.add("active");
    svNavigateTo({ lat, lon });
}

async function svNavigateTo({ lat, lon, imageId, preferHeading } = {}) {
    svSetLoading(true);

    const params = new URLSearchParams();
    if (imageId) params.set("image_id", imageId);
    if (lat != null) params.set("lat", lat);
    if (lon != null) params.set("lon", lon);
    if (preferHeading != null) params.set("prefer_heading", preferHeading);

    try {
        const resp = await fetch(`/api/operator/street_view/neighbors?${params}`);
        if (!resp.ok) {
            const errorPayload = await resp.json().catch(() => ({}));
            throw new Error(errorPayload.message || errorPayload.error || "No imagery found");
        }
        const data = await resp.json();
        svApplyState(data);
    } catch (err) {
        svSetError(err?.message || "No imagery available here.");
        console.error("Street view navigate error:", err);
    }
}

function svApplyState(data) {
    const cur = data.current;
    if (!cur) { svSetError("No imagery returned."); return; }
    const imageUrl = cur.url || cur.image_url || cur.image_path;
    if (!imageUrl) { svSetError("Street imagery was found, but no displayable image URL was returned."); return; }

    svState.imageId = cur.image_id;
    svState.lat = cur.lat;
    svState.lon = cur.lon;
    svState.heading = cur.heading;
    svState.sequence = cur.sequence;
    svState.seqPos = data.sequence_position;
    svState.seqTotal = data.sequence_total;

    // Reset zoom/pan on each new image load
    svZoom = 1; svPanX = 0; svPanY = 0;
    svSetTransform(false);

    // Load image
    const img = byId("sv-img");
    if (img) {
        img.removeAttribute("src");
        img.style.visibility = "hidden";
        img.onload = () => {
            svSetLoading(false);
            svDrawCandidateOverlays();
        };
        img.onerror = () => svSetError("Image failed to load.");
        img.src = imageUrl;
        if (img.complete && img.naturalWidth > 0) {
            svSetLoading(false);
            requestAnimationFrame(svDrawCandidateOverlays);
        }
    }

    // Meta text
    const meta = byId("sv-meta");
    if (meta) {
        const ts = cur.captured_at ? parseInt(cur.captured_at) : null;
        const dateStr = ts && !isNaN(ts) ? new Date(ts).toLocaleDateString() : (cur.captured_at || "");
        const distStr = cur.distance_km != null ? `${(cur.distance_km * 1000).toFixed(0)}m from target` : "";
        meta.textContent = [cur.provider, dateStr, distStr].filter(Boolean).join(" · ");
    }

    // Sequence indicator
    const seqInd = byId("sv-seq-indicator");
    if (seqInd) seqInd.textContent = `${data.sequence_position} / ${data.sequence_total}`;

    // Heading label
    const headLabel = byId("sv-heading-label");
    if (headLabel) headLabel.textContent = cur.heading != null ? `${Math.round(cur.heading)}°` : "--°";

    // Sequence buttons
    const prevBtn = byId("sv-btn-prev");
    const nextBtn = byId("sv-btn-next");
    if (prevBtn) { prevBtn.disabled = !data.prev; prevBtn.dataset.navId = data.prev?.image_id || ""; }
    if (nextBtn) { nextBtn.disabled = !data.next; nextBtn.dataset.navId = data.next?.image_id || ""; }

    // Directional nav buttons in the right panel
    const fwdArrow = byId("sv-arrow-fwd");
    const backArrow = byId("sv-arrow-back");
    if (fwdArrow) {
        fwdArrow.classList.toggle("hidden", !data.next);
        fwdArrow.dataset.navId = data.next?.image_id || "";
    }
    if (backArrow) {
        backArrow.classList.toggle("hidden", !data.prev);
        backArrow.dataset.navId = data.prev?.image_id || "";
    }

    svRenderCandidateList();
    svUpdateMinimap();
}

function svDrawCandidateOverlays() {
    const canvas = byId("sv-overlay-canvas");
    const img = byId("sv-img");
    if (!canvas || !img || svState.heading == null) return;

    // Position canvas exactly over the rendered image (which is flex-centered in sv-image-wrap)
    const iL = img.offsetLeft;
    const iT = img.offsetTop;
    const iW = img.offsetWidth;
    const iH = img.offsetHeight;
    if (!iW || !iH) return;

    canvas.style.left = iL + "px";
    canvas.style.top = iT + "px";
    canvas.style.width = iW + "px";
    canvas.style.height = iH + "px";
    canvas.style.inset = "auto"; // override CSS inset:0
    canvas.width = iW;
    canvas.height = iH;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, iW, iH);

    if (svState.candidates.length === 0) return;

    const FOV = 90;
    const camHeading = svState.heading;

    svState.candidates.forEach(cand => {
        if (cand.lat == null || cand.lon == null || isNaN(cand.lat)) return;

        const bearing = svBearingTo(svState.lat, svState.lon, cand.lat, cand.lon);
        let relAngle = (bearing - camHeading + 360) % 360;
        if (relAngle > 180) relAngle -= 360; // -180 to +180

        if (Math.abs(relAngle) > FOV / 2) return;

        const x = (0.5 + relAngle / FOV) * iW;
        // Pin drops from upper-middle of the image
        const pinY = iH * 0.42;
        const distKm = svHaversineKm(svState.lat, svState.lon, cand.lat, cand.lon);

        // Dashed ground line from pin bottom to bottom of image
        ctx.save();
        ctx.setLineDash([3, 4]);
        ctx.strokeStyle = cand.color + "66";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, pinY + 20);
        ctx.lineTo(x, iH - 6);
        ctx.stroke();
        ctx.restore();

        // Outer glow
        ctx.beginPath();
        ctx.arc(x, pinY, 24, 0, Math.PI * 2);
        ctx.fillStyle = cand.color + "1a";
        ctx.fill();

        // Pin circle
        ctx.beginPath();
        ctx.arc(x, pinY, 17, 0, Math.PI * 2);
        ctx.fillStyle = cand.color + "cc";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Rank number
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 12px 'Courier New', monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(`#${cand.rank}`, x, pinY);

        // Distance badge below pin
        const label = `${(distKm * 1000).toFixed(0)}m`;
        ctx.font = "9px monospace";
        const bW = ctx.measureText(label).width + 8;
        const bX = x - bW / 2;
        const bY = pinY + 22;
        ctx.fillStyle = "rgba(0,0,0,0.65)";
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(bX, bY, bW, 14, 3);
        } else {
            ctx.rect(bX, bY, bW, 14);
        }
        ctx.fill();
        ctx.fillStyle = cand.color;
        ctx.textBaseline = "middle";
        ctx.fillText(label, x, bY + 7);
    });
}

function svRenderCandidateList() {
    const list = byId("sv-cand-list");
    if (!list) return;

    list.innerHTML = '<div class="sv-cand-title">Geo Candidates</div>';

    if (!svState.lat || svState.heading == null) return;

    const FOV = 90;
    svState.candidates.forEach(cand => {
        if (cand.lat == null || isNaN(cand.lat)) return;

        const bearing = svBearingTo(svState.lat, svState.lon, cand.lat, cand.lon);
        let relAngle = (bearing - svState.heading + 360) % 360;
        if (relAngle > 180) relAngle -= 360;
        const inView = Math.abs(relAngle) <= FOV / 2;
        const distKm = svHaversineKm(svState.lat, svState.lon, cand.lat, cand.lon);
        const bearingLabel = inView
            ? "IN VIEW"
            : relAngle > 0
                ? `${Math.round(relAngle)}° right`
                : `${Math.round(-relAngle)}° left`;

        const item = document.createElement("div");
        item.className = `sv-cand-item${inView ? " in-view" : ""}`;
        item.innerHTML = `
            <div class="sv-cand-dot" style="background:${cand.color}"></div>
            <div class="sv-cand-info">
                <div class="sv-cand-rank">#${cand.rank} &nbsp;${(cand.score * 100).toFixed(0)}%</div>
                <div class="sv-cand-bearing">${bearingLabel} · ${(distKm * 1000).toFixed(0)}m</div>
            </div>`;
        list.appendChild(item);
    });
}

function svInitMinimap() {
    const container = byId("sv-minimap");
    if (!container || svState.minimap) return;

    try {
        svState.minimap = new maplibregl.Map({
            container: "sv-minimap",
            style: mapStyleUrl,
            center: parisCenter,
            zoom: 15,
            interactive: false,
            attributionControl: false,
        });
        svState.minimapReady = new Promise(resolve => svState.minimap.on("load", resolve));
    } catch (e) {
        // Mini-map is enhancement only — ignore failures
    }
}

function svUpdateMinimap() {
    if (!svState.minimap) svInitMinimap();
    if (!svState.minimapReady || svState.lat == null) return;

    svState.minimapReady.then(() => {
        svState.minimap.setCenter([svState.lon, svState.lat]);
        svState.minimap.setZoom(16);

        // Clear old markers
        (svState.minimapMarkers || []).forEach(m => m.remove());
        svState.minimapMarkers = [];

        // Camera position marker (white dot with heading arrow)
        const camEl = document.createElement("div");
        camEl.style.cssText = `width:14px;height:14px;border-radius:50%;background:#fff;border:2px solid #10b981;box-shadow:0 0 6px rgba(16,185,129,0.6);`;
        const camMarker = new maplibregl.Marker({ element: camEl })
            .setLngLat([svState.lon, svState.lat])
            .addTo(svState.minimap);
        svState.minimapMarkers.push(camMarker);

        // Candidate markers on mini-map
        svState.candidates.forEach(cand => {
            if (cand.lat == null || isNaN(cand.lat)) return;
            const el = document.createElement("div");
            el.style.cssText = `width:10px;height:10px;border-radius:50%;background:${cand.color};border:1.5px solid rgba(255,255,255,0.5);`;
            const marker = new maplibregl.Marker({ element: el })
                .setLngLat([cand.lon, cand.lat])
                .addTo(svState.minimap);
            svState.minimapMarkers.push(marker);
        });
    }).catch(() => {});
}

function renderLiveMap(result, { resetView = false } = {}) {
  ensureLiveMap();

  // Always clear existing markers and sources first
  clearCandidateMarkers();
  liveMapReady.then(() => {
    liveMap.getSource("candidates")?.setData(emptyFeatureCollection);
    liveMap.getSource("ring")?.setData({ type: "FeatureCollection", features: [] });
    liveMap.getSource("mean")?.setData(emptyFeatureCollection);
  });

  const fusion = result?.fused_estimate;
  const candidates = sortedCandidates(result);
  const mapEntries = mapCandidateEntries(candidates);

  if (isCurrentSessionHidden() || !fusion || mapEntries.length === 0) return;

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
    renderCandidatePins(mapEntries);
    liveMap.getSource("ring")?.setData({ type: "FeatureCollection", features: ringFeature ? [ringFeature] : [] });
    bringCandidateLayersToFront();
    if (resetView) {
      easeToCentered({
        center: globeCenter,
        zoom: globeZoom,
        pitch: 0,
        bearing: 0,
        duration: 600
      });
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
  renderLiveMap(result, { resetView: true });
  renderOperatorPinMarkers(result);
  renderTimeline(result);
  renderClues(result);
  selectedLightboxDetectionIndex = 0;
  // Only draw overlay if image is already decoded; doLoadSession sets onload for the async case
  const lbImg = byId("lightbox-img");
  if (lbImg && lbImg.naturalWidth > 0) renderLightboxIntel();
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
  clearOperatorPinMarkers();
  activeCandidateItems = [];
  candidatePinsPopulated = false;
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

    loadedSessionId = null;
    localStorage.removeItem("heimdallSessionId");

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
  byId("map-zoom-in").addEventListener("click", () => zoomCentered(1));
  byId("map-zoom-out").addEventListener("click", () => zoomCentered(-1));
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
    easeToCentered({
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
    easeToCentered({ center: parisCenter, zoom: 11, pitch: 0, bearing: 0, duration: 1400 });
    window.setTimeout(updateTiltLabel, 1420);
  });
  byId("map-reset-globe").addEventListener("click", () => {
    easeToCentered({ center: globeCenter, zoom: globeZoom, pitch: 0, bearing: 0, duration: 1800 });
    window.setTimeout(updateTiltLabel, 1820);
  });

  const streetWalkBtn = byId("map-street-walk");
  if (streetWalkBtn) {
      streetWalkBtn.addEventListener("click", () => {
          isStreetWalkMode = true;
          streetWalkBtn.style.color = "var(--accent)";
          liveMap.getCanvas().style.cursor = "crosshair";
      });
  }
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

  // Street view keyboard navigation (WASD + arrows + ESC)
  window.addEventListener("keydown", (e) => {
    if (!byId("street-view-modal")?.classList.contains("active")) return;
    if (e.key === "Escape") {
      e.preventDefault();
      svClose();
    } else if (e.key === "ArrowUp" || e.key === "w" || e.key === "W") {
      // Forward: next in sequence
      e.preventDefault();
      byId("sv-arrow-fwd")?.click();
    } else if (e.key === "ArrowDown" || e.key === "s" || e.key === "S") {
      // Backward: prev in sequence
      e.preventDefault();
      byId("sv-arrow-back")?.click();
    } else if (e.key === "ArrowLeft" || e.key === "a" || e.key === "A") {
      // Look left
      e.preventDefault();
      byId("sv-arrow-left-img")?.click();
    } else if (e.key === "ArrowRight" || e.key === "d" || e.key === "D") {
      // Look right
      e.preventDefault();
      byId("sv-arrow-right-img")?.click();
    } else if (e.key === "+" || e.key === "=") {
      e.preventDefault();
      svApplyZoom(svZoom + SV_ZOOM_STEP);
    } else if (e.key === "-") {
      e.preventDefault();
      svApplyZoom(svZoom - SV_ZOOM_STEP);
    } else if (e.key === "r" || e.key === "R") {
      e.preventDefault();
      svResetZoom();
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
  const copySumBtn = byId("copy-diag-all");
  const copyJsonBtn = byId("copy-json");
  const expandJsonBtn = byId("expand-json");
  
  const jsonModal = byId("json-modal");
  const jsonModalText = byId("modal-json-text");
  const closeJsonBtn = byId("close-json-modal");
  const jsonBackdrop = byId("json-modal-backdrop");
  const copyModalBtn = byId("copy-json-modal");

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

const SVG_CHEVRON_RIGHT = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>`;
const SVG_CHEVRON_DOWN  = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`;

function setupCollapsiblePanel(triggerId, bodyId) {
  const trigger = byId(triggerId);
  const body = byId(bodyId);
  if (!trigger || !body) return;

  const icon = trigger.querySelector(".toggle-chevron");
  const syncIcon = () => {
    if (icon) icon.innerHTML = body.classList.contains("active") ? SVG_CHEVRON_DOWN : SVG_CHEVRON_RIGHT;
  };

  trigger.addEventListener("click", () => {
    body.classList.toggle("active");
    syncIcon();
  });
  syncIcon();
}

function setupPanelAccordions() {
  setupCollapsiblePanel("timeline-trigger", "timeline-accordion");
  setupCollapsiblePanel("clues-trigger", "clues-accordion");
  setupCollapsiblePanel("notes-trigger", "notes-accordion");
  setupCollapsiblePanel("diag-trigger", "diag-accordion");
}

function loadSessionList() {
    const select = byId("load-session-select");
    if (!select) return Promise.resolve();
    return fetch("/api/operator/sessions")
        .then(r => r.json())
        .then(data => {
            select.innerHTML = '<option value="">— Load Session —</option>';
            if (data.sessions && data.sessions.length > 0) {
                data.sessions.forEach(session => {
                    const option = document.createElement("option");
                    option.value = session.session_id;
                    const fileStr = session.source_filename ? ` — ${session.source_filename}` : "";
                    option.textContent = `${session.display_name}${fileStr} [${session.status}]`;
                    select.appendChild(option);
                });
            }
            if (loadedSessionId) {
                select.value = loadedSessionId;
            }
        })
        .catch(err => console.error("Failed to load session list:", err));
}

function doLoadSession(sid) {
    if (!sid) return;
    fetch(`/api/operator/sessions/${sid}`)
        .then(r => {
            if (!r.ok) throw new Error(`Session load failed (HTTP ${r.status})`);
            return r.json();
        })
        .then(data => {
            if (data.error) throw new Error(data.error);

            loadedSessionId = data.session_id || sid;
            localStorage.setItem("heimdallSessionId", loadedSessionId);

            lastResult = {
                ...data,
                fused_estimate: data.fused_estimate,
                candidates: Array.isArray(data.candidates) ? data.candidates : [],
                clues: Array.isArray(data.clues) ? data.clues : [],
                detections: Array.isArray(data.detections) ? data.detections : [],
                notes: Array.isArray(data.notes) ? data.notes : [],
                operator_pins: Array.isArray(data.operator_pins) ? data.operator_pins : [],
            };

            // Restore source image into the left-panel preview area.
            // Use the binary image endpoint for reliability instead of base64 data URL.
            const source = data.source || {};
            // Add cache buster to prevent stale image loads from session storage
            const imgEndpoint = `/api/operator/sessions/${loadedSessionId}/image?t=${Date.now()}`;
            const imgUrl = source.image_data_url
                || (source.has_session_image || source.image_file ? imgEndpoint : null);
            const thumb = byId("source-thumb");
            const filenameLbl = byId("preview-filename");
            const previewBlock = byId("preview-block");
            const ingestBlock = byId("ingest-block");
            const lightboxImg = byId("lightbox-img");
            const lightboxName = byId("lightbox-filename");
            const geolocateBtn = byId("geolocate-image");

            if (source.filename || imgUrl) {
                if (filenameLbl) filenameLbl.textContent = source.filename || "Session image";
                if (lightboxName) lightboxName.textContent = source.filename || "";
                if (previewBlock) previewBlock.style.display = "flex";
                if (ingestBlock) ingestBlock.style.display = "none";
            }

            if (imgUrl) {
                console.log("LOG: Constructing image load for URL:", imgUrl);
                if (geolocateBtn) geolocateBtn.disabled = false;
                
                // Fetch image as blob to get better error reporting
                fetch(imgUrl)
                    .then(r => {
                        if (!r.ok) throw new Error(`Image fetch failed (HTTP ${r.status})`);
                        return r.blob();
                    })
                    .then(blob => {
                        const blobUrl = URL.createObjectURL(blob);
                        if (lightboxImg) {
                            selectedLightboxDetectionIndex = 0;
                            lightboxImg.removeAttribute("width");
                            lightboxImg.removeAttribute("height");
                            lightboxImg.onload = () => {
                                console.log("LOG: Lightbox image loaded via blob successfully");
                                renderLightboxIntel();
                                // Clean up blob URL after load
                                // URL.revokeObjectURL(blobUrl); 
                            };
                            lightboxImg.src = blobUrl;
                        }
                        if (thumb) {
                            thumb.src = blobUrl;
                        }
                    })
                    .catch(err => {
                        console.error("LOG: Image blob load failed:", err);
                        showAnalysisAlert(`Failed to load source image from session storage.\nReason: ${err.message}\nURL: ${imgUrl}`);
                    });
            }

            renderSummary(data);
            renderNoteMarkers();
            renderNotesList();
            if (data.operator_notes) {
                const noteInput = byId("operator-note-input");
                if (noteInput) noteInput.value = data.operator_notes;
            }
        })
        .catch(err => console.error("Failed to load session:", err));
}

const SVG_CHEVRON_LEFT  = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>`;
const SVG_CHEVRON_RIGHT2 = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>`;

function setupPanelResize() {
  const shell = document.querySelector(".app-shell");
  if (!shell) return;

  function attachDrag(handle, side) {
    if (!handle) return;
    handle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      const varName = side === "left" ? "--left-w" : "--right-w";
      const startX = e.clientX;
      const startW = parseInt(getComputedStyle(shell).getPropertyValue(varName)) || (side === "left" ? 350 : 380);
      handle.classList.add("resizing");

      const onMove = (ev) => {
        const dx = ev.clientX - startX;
        const newW = Math.max(48, Math.min(640, side === "left" ? startW + dx : startW - dx));
        shell.style.setProperty(varName, newW + "px");
        if (liveMap) liveMap.resize();
      };
      const onUp = () => {
        handle.classList.remove("resizing");
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });
  }

  attachDrag(document.querySelector(".left-resize-handle"), "left");
  attachDrag(document.querySelector(".right-resize-handle"), "right");
}

function setupPanelCollapse() {
  const shell = document.querySelector(".app-shell");
  if (!shell) return;

  function attachCollapse(panelEl, btn, side) {
    if (!panelEl || !btn) return;
    const varName = side === "left" ? "--left-w" : "--right-w";
    const defaultW = side === "left" ? "350px" : "380px";

    const syncIcon = () => {
      const collapsed = panelEl.classList.contains("collapsed");
      btn.innerHTML = side === "left"
        ? (collapsed ? SVG_CHEVRON_RIGHT2 : SVG_CHEVRON_LEFT)
        : (collapsed ? SVG_CHEVRON_LEFT  : SVG_CHEVRON_RIGHT2);
      btn.title = collapsed ? "Expand panel" : "Collapse panel";
    };

    btn.addEventListener("click", () => {
      const collapsed = panelEl.classList.toggle("collapsed");
      if (collapsed) {
        panelEl.dataset.prevW = getComputedStyle(shell).getPropertyValue(varName).trim();
        shell.style.setProperty(varName, "48px");
      } else {
        shell.style.setProperty(varName, panelEl.dataset.prevW || defaultW);
      }
      if (liveMap) liveMap.resize();
      syncIcon();
    });

    syncIcon();
  }

  attachCollapse(document.querySelector(".left-panel"),  byId("left-panel-collapse-btn"),  "left");
  attachCollapse(document.querySelector(".right-panel"), byId("right-panel-collapse-btn"), "right");
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
  setupPanelAccordions();
  setupToggles();
  setupOperatorActions();
  setupPanelResize();
  setupPanelCollapse();

  // On every page load: check if a session was queued to auto-load (set before reload).
  // If yes, load it. If no, reset server so there's no stale state.
  const pendingSessionId = localStorage.getItem("heimdallPendingLoad");
  localStorage.removeItem("heimdallPendingLoad");
  loadedSessionId = null;
  localStorage.removeItem("heimdallSessionId");

  if (pendingSessionId) {
    loadSessionList().then(() => {
      const sel = byId("load-session-select");
      if (sel) sel.value = pendingSessionId;
      doLoadSession(pendingSessionId);
    });
  } else {
    fetch("/api/operator/reset", { method: "POST" }).catch(() => {});
    loadSessionList();
  }

  const tabGeotags = document.getElementById("tab-geotags");
  const tabNotes = document.getElementById("tab-notes");
  const geotagsView = document.getElementById("geotags-view");
  const notesView = document.getElementById("notes-view");

  if (tabGeotags && tabNotes && geotagsView && notesView) {
      tabGeotags.addEventListener("click", () => {
          tabGeotags.classList.add("active");
          tabNotes.classList.remove("active");
          geotagsView.style.display = "flex";
          notesView.style.display = "none";

          // Clear Notes tab selection when switching back to Geotags
          const notesList = document.getElementById("notes-list");
          if (notesList) {
              notesList.querySelectorAll(".result-card").forEach(c => c.classList.remove("active"));
          }

          if (selectedIndex === -1) {
              const noteInput = byId("operator-note-input");
              if (noteInput) noteInput.value = "";
          } else {
              loadNoteForTarget("candidate", selectedIndex + 1);
          }
      });
      tabNotes.addEventListener("click", () => {
          tabNotes.classList.add("active");
          tabGeotags.classList.remove("active");
          notesView.style.display = "flex";
          geotagsView.style.display = "none";

          // Clear Geotags selection when switching to Notes
          selectedIndex = -1;
          const cards = document.querySelectorAll(".candidate-card");
          cards.forEach(c => c.classList.remove("active"));

          const noteInput = byId("operator-note-input");
          if (noteInput) noteInput.value = "";

          renderNotesList();
          renderNoteMarkers();
      });
  }

  const profileSelect = byId("profile-select");
  if (profileSelect) {
    if (!profileSelect.options.length) {
      profileOptions.forEach(({ value, label }) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        profileSelect.appendChild(option);
      });
    }
    const stored = localStorage.getItem(profileStorageKey);
    if (stored) profileSelect.value = stored;
    if (!profileSelect.value && profileSelect.options.length) profileSelect.value = profileOptions[0].value;
    profileSelect.addEventListener("change", () => localStorage.setItem(profileStorageKey, profileSelect.value));
  }

  initCaseSidebar();
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
  const heading = Number.isFinite(Number(det?.heading_deg)) ? `head ${Number(det.heading_deg).toFixed(0)} deg` : null;
  const shadow = Number.isFinite(Number(det?.shadow_azimuth_deg)) ? `shadow ${Number(det.shadow_azimuth_deg).toFixed(0)} deg` : null;
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
    const row = document.createElement("div");
    row.className = `intel-row${idx === selectedLightboxDetectionIndex ? " active" : ""}`;
    if (idx !== null) {
      row.tabIndex = 0;
      row.setAttribute("role", "button");
      row.addEventListener("click", () => {
        selectedLightboxDetectionIndex = idx;
        renderLightboxIntel();
      });
      row.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
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
    // Street View navigation
    const closeStreetViewBtn = byId("close-street-view-modal");
    if (closeStreetViewBtn) closeStreetViewBtn.addEventListener("click", svClose);

    const svBackdrop = byId("street-view-backdrop");
    if (svBackdrop) svBackdrop.addEventListener("click", svClose);

    const svPrevBtn = byId("sv-btn-prev");
    if (svPrevBtn) svPrevBtn.addEventListener("click", () => {
        const id = svPrevBtn.dataset.navId;
        if (id) svNavigateTo({ imageId: id });
    });

    const svNextBtn = byId("sv-btn-next");
    if (svNextBtn) svNextBtn.addEventListener("click", () => {
        const id = svNextBtn.dataset.navId;
        if (id) svNavigateTo({ imageId: id });
    });

    const svLeftBtn = byId("sv-btn-left");
    if (svLeftBtn) svLeftBtn.addEventListener("click", () => {
        const ph = ((svState.heading || 0) - 90 + 360) % 360;
        svNavigateTo({ lat: svState.lat, lon: svState.lon, preferHeading: ph });
    });

    const svRightBtn = byId("sv-btn-right");
    if (svRightBtn) svRightBtn.addEventListener("click", () => {
        const ph = ((svState.heading || 0) + 90) % 360;
        svNavigateTo({ lat: svState.lat, lon: svState.lon, preferHeading: ph });
    });

    // Right-panel directional buttons
    const svFwdArrow = byId("sv-arrow-fwd");
    if (svFwdArrow) svFwdArrow.addEventListener("click", () => {
        const id = svFwdArrow.dataset.navId;
        if (id) svNavigateTo({ imageId: id });
    });

    const svBackArrow = byId("sv-arrow-back");
    if (svBackArrow) svBackArrow.addEventListener("click", () => {
        const id = svBackArrow.dataset.navId;
        if (id) svNavigateTo({ imageId: id });
    });

    const svLeftImg = byId("sv-arrow-left-img");
    if (svLeftImg) svLeftImg.addEventListener("click", () => {
        const ph = ((svState.heading || 0) - 90 + 360) % 360;
        svNavigateTo({ lat: svState.lat, lon: svState.lon, preferHeading: ph });
    });

    const svRightImg = byId("sv-arrow-right-img");
    if (svRightImg) svRightImg.addEventListener("click", () => {
        const ph = ((svState.heading || 0) + 90) % 360;
        svNavigateTo({ lat: svState.lat, lon: svState.lon, preferHeading: ph });
    });

    // Zoom buttons (zoom to center)
    byId("sv-zoom-in")?.addEventListener("click", () => svApplyZoom(svZoom + SV_ZOOM_STEP));
    byId("sv-zoom-out")?.addEventListener("click", () => svApplyZoom(svZoom - SV_ZOOM_STEP));
    byId("sv-zoom-reset")?.addEventListener("click", () => svResetZoom());

    // Mouse wheel zoom toward cursor
    const svWrap = byId("sv-image-wrap");
    svWrap?.addEventListener("wheel", (e) => {
        if (!byId("street-view-modal")?.classList.contains("active")) return;
        e.preventDefault();
        const rect = svWrap.getBoundingClientRect();
        svApplyZoom(
            svZoom + (e.deltaY < 0 ? SV_ZOOM_STEP : -SV_ZOOM_STEP),
            e.clientX - rect.left,
            e.clientY - rect.top
        );
    }, { passive: false });

    // Drag to pan when zoomed in
    svWrap?.addEventListener("mousedown", (e) => {
        if (svZoom <= 1 || e.button !== 0) return;
        svIsDragging = true;
        svDragStartX = e.clientX;
        svDragStartY = e.clientY;
        svDragStartPanX = svPanX;
        svDragStartPanY = svPanY;
        svWrap.style.cursor = "grabbing";
        e.preventDefault();
    });
    document.addEventListener("mousemove", (e) => {
        if (!svIsDragging) return;
        svPanX = svDragStartPanX + (e.clientX - svDragStartX);
        svPanY = svDragStartPanY + (e.clientY - svDragStartY);
        svSetTransform(false);
    });
    document.addEventListener("mouseup", () => {
        if (!svIsDragging) return;
        svIsDragging = false;
        if (svWrap) svWrap.style.cursor = "";
    });

    window.addEventListener("resize", () => {
        if (byId("street-view-modal")?.classList.contains("active")) svDrawCandidateOverlays();
    });

    const confirmBtn = byId("btn-confirm-cand");
    const rejectBtn = byId("btn-reject-cand");
    const noteInput = byId("operator-note-input");
    const saveNoteBtn = byId("btn-save-note");
    const saveSessionBtn = byId("btn-save-session");
    const newSessionBtn = byId("btn-new-session");
    const dropPinBtn = byId("btn-drop-pin");

    const sessionSaveModal = byId("session-save-modal");
    const btnUpdateSession = byId("btn-modal-update-session");
    const btnSaveNewSession = byId("btn-modal-save-new-session");
    const btnCancelSession = byId("btn-modal-cancel-session");

    if (btnCancelSession && sessionSaveModal) {
        btnCancelSession.addEventListener("click", () => {
            sessionSaveModal.classList.remove("active");
        });
    }

    function afterSessionSaved(data) {
        const savedId = data.session_id;
        if (!savedId) { window.location.reload(); return; }
        localStorage.setItem("heimdallPendingLoad", savedId);

        // Auto-attach to active case if one is open (backend also does this on save)
        if (_activeCaseId) {
            fetch("/api/cases/active/sessions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: savedId }),
            }).catch(() => {});
        }
        window.location.reload();

        // (Legacy post-save modal code disabled — sessions auto-attach to active case)
        if (false) {
        const modal = document.getElementById("post-save-modal");
        const sel = document.getElementById("post-save-case-select");
        if (modal && sel) {
            sel.innerHTML = '<option value="">— No case (skip) —</option>';
            (_cases || []).forEach(c => {
                const opt = document.createElement("option");
                opt.value = c.case_id;
                opt.textContent = c.name;
                sel.appendChild(opt);
            });
            modal.style.display = "flex";

            const doFinish = async (caseId) => {
                modal.style.display = "none";
                if (caseId) {
                    await fetch(`/api/cases/${caseId}/sessions`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ session_id: savedId }),
                    }).catch(() => {});
                }
                window.location.reload();
            };

            document.getElementById("btn-post-save-assign")?.addEventListener("click", () => {
                doFinish(sel.value);
            }, { once: true });
            document.getElementById("btn-post-save-skip")?.addEventListener("click", () => {
                doFinish("");
            }, { once: true });
            document.getElementById("post-save-backdrop")?.addEventListener("click", () => {
                doFinish("");
            }, { once: true });
        } else {
            window.location.reload();
        }
        } // end if (false)
    }

    if (btnUpdateSession && sessionSaveModal) {
        btnUpdateSession.addEventListener("click", () => {
            sessionSaveModal.classList.remove("active");
            postForm("/api/operator/save", JSON.stringify({ save_as_new: false }))
                .then(afterSessionSaved)
                .catch(err => console.error("Save failed:", err));
        });
    }

    if (btnSaveNewSession && sessionSaveModal) {
        btnSaveNewSession.addEventListener("click", () => {
            sessionSaveModal.classList.remove("active");
            const sessionName = prompt("Enter a name for this session (optional):");
            if (sessionName !== null) {
                postForm("/api/operator/save", JSON.stringify({ name: sessionName.trim(), save_as_new: true }))
                    .then(afterSessionSaved)
                    .catch(err => console.error("Save failed:", err));
            }
        });
    }

    if (newSessionBtn) {
        newSessionBtn.addEventListener("click", () => {
            localStorage.removeItem("heimdallPendingLoad");
            localStorage.removeItem("heimdallSessionId");
            window.location.reload();
        });
    }

    if (saveSessionBtn) {
        saveSessionBtn.addEventListener("click", () => {
            if (loadedSessionId && sessionSaveModal) {
                sessionSaveModal.classList.add("active");
            } else {
                const sessionName = prompt("Enter a name for this session (optional):");
                if (sessionName !== null) {
                    postForm("/api/operator/save", JSON.stringify({ name: sessionName.trim(), save_as_new: true }))
                        .then(afterSessionSaved)
                        .catch(err => console.error("Save failed:", err));
                }
            }
        });
    }

    if (dropPinBtn) {
        dropPinBtn.addEventListener("click", () => {
            isDroppingPin = true;
            dropPinBtn.textContent = "Click on map...";
            dropPinBtn.style.color = "var(--accent)";
        });
    }

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
             const oldWarn = document.getElementById("note-save-warn");
             if (oldWarn) oldWarn.remove();

             // Wait, what if we selected a note from the notes list?
             // It's allowed to update the selected note or candidate. Let's rely on checking if there's any active selection.
             // The prompt logic: "Select a candidate or drop/select a note pin to save this note."
             const isNoteSelected = document.querySelector("#notes-list .result-card.active") !== null;

             if (selectedIndex === -1 && !droppedPinLocation && !isNoteSelected) {
                 const warn = document.createElement("div");
                 warn.id = "note-save-warn";
                 warn.style.color = "#ef4444";
                 warn.style.fontSize = "11px";
                 warn.style.marginTop = "4px";
                 warn.textContent = "Select a candidate or drop/select a note pin to save this note.";
                 noteInput.parentElement.appendChild(warn);
                 return;
             }

             let targetData = {};
             // We can check which tab is active to determine precedence
             const isNotesTabActive = document.getElementById("notes-view")?.style.display === "flex";

             if (isNotesTabActive && isNoteSelected) {
                 const activeCard = document.querySelector("#notes-list .result-card.active");
                 if (activeCard && activeCard.dataset.noteId) {
                     targetData = {
                         target_type: "note_id",
                         note_id: activeCard.dataset.noteId
                     };
                 } else if (activeCard?.dataset.lat && activeCard?.dataset.lon) {
                     targetData = {
                         target_type: "manual_pin",
                         pin_id: activeCard.dataset.pinId || undefined,
                         lat: Number(activeCard.dataset.lat),
                         lon: Number(activeCard.dataset.lon)
                     };
                 }
             } else if (selectedIndex !== -1 && lastResult && lastResult.candidates) {
                 const cand = sortedCandidates(lastResult)[selectedIndex];
                 targetData = {
                     target_type: "candidate",
                     rank: selectedIndex + 1,
                     source: cand.source
                 };
             } else if (droppedPinLocation) {
                 targetData = {
                     target_type: "manual_pin",
                     pin_id: droppedPinLocation.pin_id,
                     lat: droppedPinLocation.lat,
                     lon: droppedPinLocation.lon
                 };
             }

             postForm("/api/operator/note", JSON.stringify({note: noteInput.value, ...targetData}))
                 .then((data) => {
                     if (!lastResult) lastResult = {};
                     if (Array.isArray(data.notes)) lastResult.notes = data.notes;
                     if (Array.isArray(data.operator_pins)) {
                         lastResult.operator_pins = data.operator_pins;
                         renderOperatorPinMarkers(lastResult);
                     }
                     saveNoteBtn.textContent = "Saved";
                     noteInput.blur();

                     // Reset drop pin location after save
                     if (droppedPinLocation && dropPinBtn) {
                         droppedPinLocation = null;
                         dropPinBtn.textContent = "Drop Pin";
                     }

                     // Refresh notes list and map markers immediately
                     renderNotesList();
                     renderNoteMarkers();

                     setTimeout(() => {
                         saveNoteBtn.textContent = "Save Note";
                     }, 2000);
                 });
        });
    }


    const sessionSelect = byId("load-session-select");
    const loadSessionBtn = byId("btn-load-session");

    if (loadSessionBtn && sessionSelect) {
        loadSessionBtn.addEventListener("click", () => {
            const sid = sessionSelect.value;
            if (!sid) {
                const oldWarn = document.getElementById("session-load-warn");
                if (oldWarn) oldWarn.remove();
                const warn = document.createElement("span");
                warn.id = "session-load-warn";
                warn.style.color = "#ef4444";
                warn.style.fontSize = "11px";
                warn.style.marginLeft = "8px";
                warn.textContent = "Select a session first.";
                loadSessionBtn.parentElement.appendChild(warn);
                setTimeout(() => warn.remove(), 3000);
                return;
            }
            localStorage.setItem("heimdallPendingLoad", sid);
            window.location.reload();
        });
    }

    if (liveMap) {
        liveMap.on('click', (e) => {
             if (isStreetWalkMode) {
                  const lat = e.lngLat.lat;
                  const lon = e.lngLat.lng;
                  isStreetWalkMode = false;

                  const streetWalkBtn = byId("map-street-walk");
                  if (streetWalkBtn) {
                      streetWalkBtn.style.color = "";
                  }
                  liveMap.getCanvas().style.cursor = "";

                  openStreetView(lat, lon);
                  return;
             }

             if (isDroppingPin) {
                  const lat = e.lngLat.lat;
                  const lon = e.lngLat.lng;
                  droppedPinLocation = { lat, lon };
                  isDroppingPin = false;

                  if (dropPinBtn) {
                      dropPinBtn.textContent = "Dropped";
                      dropPinBtn.style.color = "";
                  }

                  // Deselect candidate if dropping a manual pin
                  selectedIndex = -1;
                  document.querySelectorAll(".candidate-card").forEach(c => c.classList.remove("active"));

                  const noteInput = byId("operator-note-input");

                  if (currentManualPinMarker) {
                      currentManualPinMarker.remove();
                  }

                  const el = document.createElement("div");
                  el.className = "manual-pin-marker";
                  el.style.width = "16px";
                  el.style.height = "16px";
                  el.style.backgroundColor = "#ff4444";
                  el.style.border = "2px solid white";
                  el.style.borderRadius = "50%";
                  el.style.boxShadow = "0 0 8px rgba(255, 0, 0, 0.8)";

                  el.addEventListener("click", (evt) => {
                      evt.stopPropagation();
                      loadNoteForTarget("manual_pin", lat, lon);
                  });

                  currentManualPinMarker = new maplibregl.Marker({ element: el })
                      .setLngLat([lon, lat])
                      .addTo(liveMap);

                  const label = (noteInput?.value || "").trim() || "Custom Pin";
                  fetch("/api/operator/pin", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ lat, lon, label }),
                  })
                    .then(r => r.ok ? r.json() : Promise.reject(r))
                    .then(data => {
                        if (!lastResult) lastResult = {};
                        lastResult.operator_pins = data.operator_pins || [...(lastResult.operator_pins || []), data.pin].filter(Boolean);
                        if (data.pin?.pin_id) droppedPinLocation = { lat, lon, pin_id: data.pin.pin_id };
                        if (currentManualPinMarker) {
                            currentManualPinMarker.remove();
                            currentManualPinMarker = null;
                        }
                        renderOperatorPinMarkers(lastResult);
                        renderNotesList();
                    })
                    .catch(err => console.error("drop pin failed:", err));
             }
        });
    }
}




function renderNoteMarkers() {
    if (!liveMap) return;

    // Clear old markers
    noteMarkers.forEach(m => m.remove());
    noteMarkers = [];

    fetch("/api/operator/session").then(r => r.json()).then(data => {
        if (!data.notes || data.notes.length === 0) return;

        data.notes.forEach(note => {
            let lat = null, lon = null;
            if (note.target_type === "candidate") {
                const cand = sortedCandidates(data)[note.rank - 1];
                if (cand) {
                    lat = candidateMapLat(cand);
                    lon = candidateMapLon(cand);
                }
            } else if (note.target_type === "manual_pin") {
                lat = note.lat;
                lon = note.lon;
            }

            if (lat !== null && lon !== null) {
                const el = document.createElement("div");
                el.className = "note-marker";
                el.style.width = "18px";
                el.style.height = "18px";
                el.style.backgroundColor = "#ffb84d";
                el.style.border = "2px solid #333";
                el.style.borderRadius = "50%";
                el.style.boxShadow = "0 0 8px rgba(255, 184, 77, 0.8)";
                el.title = note.text;

                el.addEventListener("click", (evt) => {
                    evt.stopPropagation();
                    const noteInput = byId("operator-note-input");
                    if (noteInput) {
                        noteInput.value = note.text;
                    }
                });

                const marker = new maplibregl.Marker({ element: el })
                    .setLngLat([lon, lat])
                    .addTo(liveMap);

                noteMarkers.push(marker);
            }
        });
    });
}

function clearOperatorPinMarkers() {
    operatorPinMarkers.forEach(m => m.remove());
    operatorPinMarkers = [];
}

function renderOperatorPinMarkers(data = lastResult) {
    if (!liveMap) return;
    clearOperatorPinMarkers();
    const pins = Array.isArray(data?.operator_pins) ? data.operator_pins : [];
    pins.forEach((pin) => {
        const lat = Number(pin.lat);
        const lon = Number(pin.lon);
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;

        const el = document.createElement("div");
        el.className = "manual-pin-marker";
        el.title = pin.label || "Custom Pin";
        el.addEventListener("click", (evt) => {
            evt.stopPropagation();
            droppedPinLocation = { lat, lon, pin_id: pin.pin_id };
            selectedIndex = -1;
            document.querySelectorAll(".candidate-card").forEach(c => c.classList.remove("active"));
            const noteInput = byId("operator-note-input");
            if (noteInput) noteInput.value = pin.label || "";
            loadNoteForTarget("manual_pin", lat, lon);
        });

        const marker = new maplibregl.Marker({ element: el, anchor: "center" })
            .setLngLat([lon, lat])
            .addTo(liveMap);
        operatorPinMarkers.push(marker);
    });
}

function noteMatchesPin(note, pin) {
    if (!note || !pin || note.target_type !== "manual_pin") return false;
    if (note.pin_id && pin.pin_id) return note.pin_id === pin.pin_id;
    const noteLat = Number(note.lat);
    const noteLon = Number(note.lon);
    const pinLat = Number(pin.lat);
    const pinLon = Number(pin.lon);
    return Number.isFinite(noteLat) && Number.isFinite(noteLon)
        && Number.isFinite(pinLat) && Number.isFinite(pinLon)
        && Math.abs(noteLat - pinLat) < 0.0001
        && Math.abs(noteLon - pinLon) < 0.0001;
}

function applyAnnotationPayload(data) {
    if (!lastResult) lastResult = {};
    if (Array.isArray(data?.notes)) lastResult.notes = data.notes;
    if (Array.isArray(data?.operator_pins)) lastResult.operator_pins = data.operator_pins;
    renderOperatorPinMarkers(lastResult);
    renderNoteMarkers();
    renderNotesList();
}

async function deleteOperatorNote(noteId) {
    if (!noteId) return;
    if (!window.confirm("Delete this note?")) return;
    const res = await fetch(`/api/operator/notes/${noteId}`, { method: "DELETE" });
    if (!res.ok) throw new Error(await res.text());
    applyAnnotationPayload(await res.json());
}

async function deleteOperatorPin(pinId) {
    if (!pinId) return;
    if (!window.confirm("Delete this custom pin and its note?")) return;
    const res = await fetch(`/api/operator/pins/${pinId}`, { method: "DELETE" });
    if (!res.ok) throw new Error(await res.text());
    applyAnnotationPayload(await res.json());
}

function renderNotesList() {
    const list = document.getElementById("notes-list");
    if (!list) return;

    fetch("/api/operator/session").then(r => r.json()).then(data => {
        const notes = Array.isArray(data.notes) ? data.notes : [];
        const pins = Array.isArray(data.operator_pins) ? data.operator_pins : [];
        if (notes.length === 0 && pins.length === 0) {
            list.innerHTML = '<div class="empty-state">No notes saved.</div>';
            return;
        }

        list.innerHTML = "";
        pins.forEach(pin => {
            const lat = Number(pin.lat);
            const lon = Number(pin.lon);
            if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
            const note = notes.find(n => noteMatchesPin(n, pin));
            const card = document.createElement("div");
            card.className = "result-card pin-result-card";
            card.dataset.pinId = pin.pin_id || "";
            card.dataset.lat = String(lat);
            card.dataset.lon = String(lon);
            if (note?.note_id) card.dataset.noteId = note.note_id;
            card.innerHTML = `
                <div class="pin-card-kicker">Custom Pin</div>
                <div class="card-sub note-text" style="color: #fff; font-size: 12px;">${note?.text || pin.label || "Custom Pin"}</div>
                <div class="card-coords-row">
                  <div class="card-coords">${lat.toFixed(6)}, ${lon.toFixed(6)}</div>
                  <button class="btn-icon-small copy-coords" type="button">COPY</button>
                  <button class="note-delete-btn" type="button" title="Delete custom pin">DELETE</button>
                </div>
            `;
            card.querySelector(".copy-coords")?.addEventListener("click", (e) => {
                e.stopPropagation();
                navigator.clipboard.writeText(`${lat.toFixed(6)}, ${lon.toFixed(6)}`);
            });
            card.querySelector(".note-delete-btn")?.addEventListener("click", (e) => {
                e.stopPropagation();
                deleteOperatorPin(pin.pin_id).catch(err => console.error("delete pin failed:", err));
            });
            card.addEventListener("click", () => {
                list.querySelectorAll(".result-card").forEach(c => c.classList.remove("active"));
                card.classList.add("active");
                droppedPinLocation = { lat, lon, pin_id: pin.pin_id };
                selectedIndex = -1;
                if (liveMap) flyToCentered({ center: [lon, lat], zoom: 16, pitch: 0, bearing: 0, duration: 900 });
                const noteInput = byId("operator-note-input");
                if (noteInput) noteInput.value = note?.text || pin.label || "";
                loadNoteForTarget("manual_pin", lat, lon);
            });
            list.appendChild(card);
        });

        notes.forEach(note => {
            if (note.target_type === "manual_pin" && pins.some(pin => noteMatchesPin(note, pin))) return;
            const card = document.createElement("div");
            card.className = "result-card";
            card.style.marginBottom = "8px";
            card.style.cursor = "pointer";

            let targetHtml = "";
            let lat = 0, lon = 0;
            if (note.target_type === "candidate") {
                targetHtml = `<div style="color: var(--accent); font-size: 10px; margin-bottom: 4px;">Candidate Rank ${note.rank} • ${note.source}</div>`;
                const cand = sortedCandidates(data)[note.rank - 1];
                if (cand) {
                    lat = candidateMapLat(cand);
                    lon = candidateMapLon(cand);
                }
            } else if (note.target_type === "manual_pin") {
                lat = note.lat;
                lon = note.lon;
                targetHtml = `<div style="color: #ff4444; font-size: 10px; margin-bottom: 4px;">Manual Pin • ${lat.toFixed(4)}, ${lon.toFixed(4)}</div>`;
            }

            const timeStr = note.timestamp ? note.timestamp.split(" ").slice(1).join(" ") : "";

            card.innerHTML = `
                ${targetHtml}
                <div class="card-sub-wrap note-text-wrap" style="margin-bottom: 6px;">
                  <div class="card-sub note-text" style="color: #fff; font-size: 12px;">${note.text}</div>
                  <button class="source-more note-more" type="button" hidden>Show more</button>
                </div>
                <div class="note-card-footer">
                  <span style="font-size: 9px; color: var(--text-secondary); border-top: none;">${timeStr}</span>
                  <button class="note-delete-btn" type="button" title="Delete note">DELETE</button>
                </div>
            `;
            if (note.note_id) card.dataset.noteId = note.note_id;
            card.querySelector(".note-delete-btn")?.addEventListener("click", (e) => {
                e.stopPropagation();
                deleteOperatorNote(note.note_id).catch(err => console.error("delete note failed:", err));
            });

            card.addEventListener("click", () => {
                list.querySelectorAll(".result-card").forEach(c => c.classList.remove("active"));
                card.classList.add("active");
                if (lat && lon && liveMap) {
                    flyToCentered({ center: [lon, lat], zoom: 16 });
                }
                const noteInput = byId("operator-note-input");
                if (noteInput) {
                    noteInput.value = note.text;
                }
            });

            list.appendChild(card);

            const subEl = card.querySelector(".note-text");
            const sourceMore = card.querySelector(".note-more");
            if (subEl && sourceMore) {
              requestAnimationFrame(() => {
                const isOverflowing = subEl.scrollHeight > subEl.clientHeight + 1;
                sourceMore.hidden = !isOverflowing;
              });

              sourceMore.addEventListener("click", (e) => {
                e.stopPropagation();
                const isExp = subEl.classList.toggle("expanded");
                sourceMore.textContent = isExp ? "Show less" : "Show more";
              });
            }

        });
    });
}

// ─── REFUSE CANDIDATE ────────────────────────────────────────────────────────
async function acceptCandidate(index, rank = index + 1) {
  if (!lastResult) return;
  try {
    const target = sortedCandidates(lastResult)[index];
    if (target) target.accepted = true;
    renderCandidateList(lastResult);
    renderLiveMap(lastResult);

    const res = await fetch("/api/operator/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index, rank, action: "accept" }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if (Array.isArray(data.candidates)) {
      lastResult.candidates = data.candidates;
      renderCandidateList(lastResult);
      renderLiveMap(lastResult);
    }
  } catch (err) {
    console.error("acceptCandidate error:", err);
  }
}

function rebuildSessionOverlayMarkers(sessionId) {
  const entry = _overlayMarkers.get(sessionId);
  if (!entry || !liveMap) return;
  entry.markers.forEach(marker => marker.remove());
  entry.markers = [];
  reviewableSessionCandidates(entry.data).forEach(c => {
    const coord = numericCandidateCoord(c);
    if (!coord) return;
    const el = document.createElement("div");
    el.className = "overlay-session-marker";
    el.style.setProperty("--marker-color", entry.color);
    el.title = `[${(sessionId || "").slice(0, 6)}] ${c.address || ""}`;
    const marker = new maplibregl.Marker({ element: el })
      .setLngLat([coord.lon, coord.lat])
      .addTo(liveMap);
    if (!entry.visible) marker.getElement().style.display = "none";
    entry.markers.push(marker);
  });
}

async function acceptSessionCandidate(sessionId, index, rank = index + 1) {
  const entry = _overlayMarkers.get(sessionId);
  if (!entry) return;
  try {
    const res = await fetch("/api/operator/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, index, rank, action: "accept" }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if (Array.isArray(data.candidates)) {
      entry.data.candidates = data.candidates;
      rebuildSessionOverlayMarkers(sessionId);
      renderCandidateList(lastResult);
    }
  } catch (err) {
    console.error("acceptSessionCandidate error:", err);
  }
}

async function removeSessionCandidate(sessionId, index, rank = index + 1) {
  const entry = _overlayMarkers.get(sessionId);
  if (!entry) return;
  try {
    const res = await fetch("/api/operator/refuse-candidate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, index, rank }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if (Array.isArray(data.candidates)) {
      entry.data.candidates = data.candidates;
      rebuildSessionOverlayMarkers(sessionId);
      renderCandidateList(lastResult);
    }
  } catch (err) {
    console.error("removeSessionCandidate error:", err);
  }
}

async function refuseCandidate(index, rank = index + 1) {
  if (!lastResult) return;
  const target = sortedCandidates(lastResult)[index];
  lastResult.candidates = (lastResult.candidates || [])
    .filter(c => c !== target && c.rank !== rank)
    .map((c, i) => ({ ...c, rank: i + 1 }));
  if (selectedIndex === index) selectedIndex = -1;
  renderCandidateList(lastResult);
  renderLiveMap(lastResult);
  renderNoteMarkers();

  try {
    const payload = { index, rank };
    const res = await fetch("/api/operator/refuse-candidate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data.candidates)) {
        lastResult.candidates = data.candidates;
      }
      renderCandidateList(lastResult);
      renderLiveMap(lastResult);
      renderNoteMarkers();
    } else throw new Error(await res.text());
  } catch (err) {
    console.error("refuseCandidate error:", err);
  }
}

// ─── CASE SIDEBAR STATE ───────────────────────────────────────────────────────
const SESSION_COLORS = [
  "#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6",
  "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
];
let _cases = [];
let _allSessions = [];
let _activeCaseId = null;
let _activeCase = null;
let _overlayMarkers = new Map();
let _pendingPhotoNoteId = null;
let _hiddenCaseSessions = new Set();

function isCurrentSessionHidden() {
  return Boolean(loadedSessionId && _hiddenCaseSessions.has(loadedSessionId));
}

function sessionColor(idx) {
  return SESSION_COLORS[idx % SESSION_COLORS.length];
}

function relativeTime(isoStr) {
  if (!isoStr) return "";
  const diff = Date.now() - new Date(isoStr).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

// ─── DATA LOADING ─────────────────────────────────────────────────────────────
async function loadCases() {
  try {
    const res = await fetch("/api/cases");
    if (res.ok) _cases = (await res.json()).cases || [];
  } catch (e) { console.error("loadCases:", e); }
}

async function loadAllSessions() {
  try {
    const res = await fetch("/api/operator/sessions?include_case=1");
    if (res.ok) _allSessions = (await res.json()).sessions || [];
  } catch (e) { console.error("loadAllSessions:", e); }
}

async function loadActiveCase() {
  try {
    const res = await fetch("/api/cases/active");
    if (res.ok) {
      const data = await res.json();
      _activeCase = data.active_case || null;
      _activeCaseId = _activeCase ? _activeCase.case_id : null;
    }
  } catch (e) { console.error("loadActiveCase:", e); }
}

async function refreshSidebar() {
  await Promise.all([loadCases(), loadAllSessions(), loadActiveCase()]);
  renderSidebar();
  updateActiveCaseBadge();
}

// ─── SIDEBAR RENDER ───────────────────────────────────────────────────────────
function renderSidebar() {
  const body = document.getElementById("case-sidebar-body");
  if (!body) return;
  const query = normalizeCaseSearch(document.getElementById("case-search")?.value || "");
  body.innerHTML = "";

  if (!_activeCase) {
    body.innerHTML = '<div class="cs-empty">No case open.<br>Click <b>Open Case</b> to start.</div>';
    return;
  }

  const caseHeader = document.createElement("div");
  caseHeader.className = "cs-active-case-header";
  caseHeader.textContent = _activeCase.name;
  body.appendChild(caseHeader);

  const sessionIds = _activeCase.sessions || [];
  const sessions = sessionIds
    .map(sid => _allSessions.find(s => s.session_id === sid) || {
      session_id: sid,
      display_name: `Missing session ${(sid || "").slice(0, 8)}`,
      missing: true
    })
    .filter(Boolean);

  const filteredSessions = query ? sessions.filter(sess => sessionMatchesQuery(sess, query)) : sessions;
  const caseMatches = query && caseMatchesQuery(_activeCase, query);

  if (sessions.length === 0 && !caseMatches) {
    const empty = document.createElement("div");
    empty.className = "cs-empty";
    empty.style.paddingTop = "10px";
    empty.textContent = "No sessions yet. Save a session to add it here.";
    body.appendChild(empty);
    return;
  }

  if (query && !caseMatches && filteredSessions.length === 0) {
    const empty = document.createElement("div");
    empty.className = "cs-empty";
    empty.style.paddingTop = "10px";
    empty.textContent = "No matching sessions in this case.";
    body.appendChild(empty);
    return;
  }

  filteredSessions.forEach((sess, si) => {
    body.appendChild(buildSessionItem(sess, si, _activeCaseId));
  });
}

function normalizeCaseSearch(value) {
  return String(value || "").trim().toLowerCase();
}

function sessionSearchText(session) {
  return [
    session?.custom_name,
    session?.display_name,
    session?.source_filename,
    session?.session_id,
    session?.status,
  ].filter(Boolean).join(" ").toLowerCase();
}

function sessionMatchesQuery(session, query) {
  return !query || sessionSearchText(session).includes(query);
}

function caseMatchesQuery(caseData, query) {
  return !query || [
    caseData?.name,
    caseData?.description,
    caseData?.case_id,
  ].filter(Boolean).join(" ").toLowerCase().includes(query);
}

function buildCaseGroup(c, ci, query) {
  const group = document.createElement("div");
  group.className = "cs-case-group";
  const isActive = c.case_id === _activeCaseId;

  const row = document.createElement("div");
  row.className = "cs-case-row" + (isActive ? " active" : "");

  const chevron = document.createElement("span");
  chevron.className = "cs-case-chevron";
  chevron.textContent = "▶";

  const nameEl = document.createElement("span");
  nameEl.className = "cs-case-name";
  nameEl.textContent = c.name;

  const countEl = document.createElement("span");
  countEl.className = "cs-case-count";
  countEl.textContent = (c.sessions || []).length;

  const actions = document.createElement("div");
  actions.className = "cs-case-actions";
  const delBtn = document.createElement("button");
  delBtn.className = "cs-btn-icon";
  delBtn.title = "Delete case";
  delBtn.textContent = "✕";
  delBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!confirm(`Delete case "${c.name}"?`)) return;
    await fetch(`/api/cases/${c.case_id}`, { method: "DELETE" });
    if (_activeCaseId === c.case_id) { _activeCaseId = null; updateFooterBtn(); }
    refreshSidebar();
  });
  actions.appendChild(delBtn);
  row.append(chevron, nameEl, countEl, actions);

  const sessionsList = document.createElement("div");
  sessionsList.className = "cs-sessions-list";
  let open = isActive;

  const setOpen = (val) => {
    open = val;
    sessionsList.classList.toggle("open", open);
    chevron.style.transform = open ? "rotate(90deg)" : "";
  };

  row.addEventListener("click", (e) => {
    if (e.target.closest(".cs-case-actions")) return;
    if (!open) {
      _activeCaseId = c.case_id;
      updateActiveCaseBadge();
      updateFooterBtn();
    }
    setOpen(!open);
  });

  setOpen(open);

  (c.sessions || []).forEach((sid, si) => {
    const sess = _allSessions.find(s => s.session_id === sid);
    if (!sess) return;
    if (query && !sessionMatchesQuery(sess, query)) return;
    sessionsList.appendChild(buildSessionItem(sess, si, c.case_id));
  });

  group.appendChild(row);
  group.appendChild(sessionsList);
  return group;
}

function buildSessionItem(session, colorIdx, caseId) {
  const sid = session.session_id;
  const color = sessionColor(colorIdx);
  const overlay = _overlayMarkers.get(sid);
  const isVisible = !_hiddenCaseSessions.has(sid) && (overlay?.visible || loadedSessionId === sid);
  const isLoaded = loadedSessionId === sid;

  const item = document.createElement("div");
  item.className = "cs-session-item" + (isLoaded ? " active-session" : "") + (session.missing ? " missing-session" : "");
  item.dataset.sessionId = sid;

  const vis = document.createElement("div");
  vis.className = "cs-session-vis" + (isVisible ? " visible" : "");
  vis.style.setProperty("--dot-color", color);
  vis.title = isVisible ? "Hide from map" : "Show on map";
  vis.addEventListener("click", (e) => {
    e.stopPropagation();
    if (session.missing) return;
    toggleSessionOverlay(sid, color);
  });

  const info = document.createElement("div");
  info.className = "cs-session-info";

  const nameEl = document.createElement("span");
  nameEl.className = "cs-session-name";
  nameEl.textContent = session.custom_name || session.display_name || (sid || "").slice(0, 14);

  const meta = document.createElement("span");
  meta.className = "cs-session-meta";
  meta.textContent = session.missing ? "missing saved data" : relativeTime(session.updated_at || session.created_at);

  info.append(nameEl, meta);
  item.append(vis, info);

  if (caseId) {
    const removeBtn = document.createElement("button");
    removeBtn.className = "cs-session-remove";
    removeBtn.title = "Remove from case";
    removeBtn.setAttribute("aria-label", "Remove session from case");
    removeBtn.textContent = "x";
    removeBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const sessionName = session.custom_name || session.display_name || (sid || "").slice(0, 14);
      if (!window.confirm(`Remove "${sessionName}" from this case?`)) return;
      try {
        const res = await fetch(`/api/cases/${caseId}/sessions/${sid}`, { method: "DELETE" });
        if (!res.ok) throw new Error(`DELETE returned ${res.status}`);
        const overlay = _overlayMarkers.get(sid);
        if (overlay) {
          overlay.markers.forEach(marker => marker.remove());
          _overlayMarkers.delete(sid);
        }
        _hiddenCaseSessions.delete(sid);
        if (_activeCase?.sessions) {
          _activeCase.sessions = _activeCase.sessions.filter(sessionId => sessionId !== sid);
        }
        await refreshSidebar();
        renderCandidateList(lastResult);
      } catch (err) {
        console.error("Failed to remove session from case", err);
        window.alert("Could not remove that session from the case.");
      }
    });
    item.appendChild(removeBtn);
  }

  item.addEventListener("click", (e) => {
    if (e.target.closest(".cs-session-vis") || e.target.closest(".cs-session-remove")) return;
    if (session.missing) return;
    loadSessionFromSidebar(sid);
  });

  return item;
}

// ─── OVERLAY MARKERS ──────────────────────────────────────────────────────────
async function toggleSessionOverlay(sessionId, color) {
  const willHide = !_hiddenCaseSessions.has(sessionId) && (_overlayMarkers.get(sessionId)?.visible || loadedSessionId === sessionId);
  if (willHide) {
    _hiddenCaseSessions.add(sessionId);
    const entry = _overlayMarkers.get(sessionId);
    if (entry) {
      entry.visible = false;
      entry.markers.forEach(m => { m.getElement().style.display = "none"; });
    }
    if (loadedSessionId === sessionId) {
      renderCandidateList(lastResult);
      renderLiveMap(lastResult);
    } else {
      renderCandidateList(lastResult);
    }
    renderSidebar();
    return;
  }

  _hiddenCaseSessions.delete(sessionId);
  if (_overlayMarkers.has(sessionId)) {
    const entry = _overlayMarkers.get(sessionId);
    entry.visible = true;
    entry.markers.forEach(m => {
      m.getElement().style.display = "";
    });
    if (loadedSessionId === sessionId) renderLiveMap(lastResult);
    renderSidebar();
    renderCandidateList(lastResult);
    return;
  }
  try {
    const res = await fetch(`/api/operator/sessions/${sessionId}`);
    if (!res.ok) return;
    const data = await res.json();
    const markers = [];
    reviewableSessionCandidates(data).forEach(c => {
      const coord = numericCandidateCoord(c);
      if (!coord) return;
      const el = document.createElement("div");
      el.className = "overlay-session-marker";
      el.style.setProperty("--marker-color", color);
      el.title = `[${(sessionId || "").slice(0, 6)}] ${c.address || ""}`;
      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([coord.lon, coord.lat])
        .addTo(liveMap);
      markers.push(marker);
    });
    _overlayMarkers.set(sessionId, { markers, visible: true, color, data });
    if (loadedSessionId === sessionId) renderLiveMap(lastResult);
    renderSidebar();
    renderCandidateList(lastResult);
  } catch (e) { console.error("toggleSessionOverlay:", e); }
}

// ─── LOAD SESSION FROM SIDEBAR ────────────────────────────────────────────────
function loadSessionFromSidebar(sid) {
  if (!sid) return;
  _hiddenCaseSessions.delete(sid);
  doLoadSession(sid);
  setTimeout(() => renderSidebar(), 900);
}

// ─── ACTIVE CASE BADGE ────────────────────────────────────────────────────────
function updateActiveCaseBadge() {
  const badge = document.getElementById("active-case-badge");
  if (!badge) return;
  if (_activeCase) {
    badge.textContent = _activeCase.name;
    badge.style.display = "";
  } else {
    badge.textContent = "";
    badge.style.display = "none";
  }
}

// ─── INIT CASE SIDEBAR ────────────────────────────────────────────────────────
function initCaseSidebar() {
  const shell = document.querySelector(".app-shell");
  const collapseBtn = document.getElementById("btn-collapse-sidebar");

  const syncCollapseIcon = () => {
    if (!collapseBtn) return;
    const collapsed = shell?.classList.contains("sidebar-collapsed");
    collapseBtn.innerHTML = collapsed ? SVG_CHEVRON_RIGHT2 : SVG_CHEVRON_LEFT;
    collapseBtn.title = collapsed ? "Expand sidebar" : "Collapse sidebar";
  };

  const toggleSidebar = () => {
    shell?.classList.toggle("sidebar-collapsed");
    syncCollapseIcon();
  };

  collapseBtn?.addEventListener("click", toggleSidebar);

  // ── Open Case modal ──────────────────────────────────────────────
  const openCaseModal = () => {
    const modal = document.getElementById("open-case-modal");
    if (!modal) return;
    const list = document.getElementById("open-case-list");
    if (list) {
      list.innerHTML = "";
      if (_cases.length === 0) {
        list.innerHTML = '<div class="cs-empty" style="padding:12px 0 4px;">No cases yet. Create one below.</div>';
      } else {
        _cases.forEach(c => {
          const item = document.createElement("div");
          item.className = "open-case-item" + (c.case_id === _activeCaseId ? " active-item" : "");
          item.innerHTML = `<span style="flex:1;font-weight:500;">${c.name}</span><span style="font-size:10px;color:var(--cs-text-dim);">${(c.sessions || []).length} sessions</span>`;
          item.addEventListener("click", async () => {
            const res = await fetch("/api/cases/active", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ case_id: c.case_id }),
            });
            if (res.ok) {
              modal.style.display = "none";
              await refreshSidebar();
            }
          });
          list.appendChild(item);
        });
      }
    }
    modal.style.display = "flex";
    document.getElementById("new-case-name")?.focus();
  };

  document.getElementById("btn-open-case")?.addEventListener("click", openCaseModal);
  document.getElementById("btn-open-case-footer")?.addEventListener("click", openCaseModal);

  document.getElementById("btn-create-case-confirm")?.addEventListener("click", async () => {
    const nameEl = document.getElementById("new-case-name");
    const name = nameEl?.value.trim();
    if (!name) { nameEl?.focus(); return; }
    const res = await fetch("/api/cases/active", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (res.ok) {
      document.getElementById("open-case-modal").style.display = "none";
      if (nameEl) nameEl.value = "";
      await refreshSidebar();
    }
  });

  document.getElementById("btn-open-case-cancel")?.addEventListener("click", () => {
    document.getElementById("open-case-modal").style.display = "none";
  });
  document.getElementById("open-case-backdrop")?.addEventListener("click", () => {
    document.getElementById("open-case-modal").style.display = "none";
  });

  document.getElementById("case-search")?.addEventListener("input", () => renderSidebar());

  // ── Note photo modal ─────────────────────────────────────────────
  document.getElementById("btn-attach-photo-confirm")?.addEventListener("click", async () => {
    const fileInput = document.getElementById("note-photo-file");
    if (!fileInput?.files.length || !_pendingPhotoNoteId) return;
    const fd = new FormData();
    fd.append("session_id", loadedSessionId || "");
    fd.append("note_id", _pendingPhotoNoteId);
    fd.append("photo", fileInput.files[0]);
    const res = await fetch("/api/operator/note-photo", { method: "POST", body: fd });
    if (res.ok) {
      document.getElementById("note-photo-modal").style.display = "none";
      fileInput.value = "";
    }
  });

  document.getElementById("btn-note-photo-cancel")?.addEventListener("click", () => {
    document.getElementById("note-photo-modal").style.display = "none";
  });

  syncCollapseIcon();
  refreshSidebar();
}
