import { byId, formatUtcNowLabel, normalizeError, postForm } from "./shared.js";

const profileStorageKey = "heimdallProfile";
let isDroppingPin = false;
let isStreetWalkMode = false;
let droppedPinLocation = null;
let currentManualPinMarker = null;

// Street View interactive viewer state
let svState = {
    imageId: null, lat: null, lon: null, heading: null,
    sequence: null, seqPos: null, seqTotal: null,
    candidates: [], minimap: null, minimapReady: null, minimapMarkers: [],
};
const SV_CAND_COLORS = ["#f2f2f2", "#c9c9c9", "#9f9f9f", "#777777", "#555555"];

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
}

function updateHtmlMarkerSelection(index) {
  candidateMarkers.forEach((marker) => {
    const el = marker.getElement();
    const indices = markerCandidateIndices(el);
    el.classList.toggle("selected", indices.includes(index));
    el.classList.toggle("top", indices.includes(0));
  });
}

function updatePinScale() {
  if (!liveMap) return;
  const zoom = liveMap.getZoom();
  const t = Math.max(0, Math.min(1, (zoom - 2) / 12));
  const eased = t * t * (3 - 2 * t);
  const scale = 0.72 + eased * 0.42;
  candidateMarkers.forEach((marker) => {
    const el = marker.getElement();
    el.style.setProperty("--pin-scale", scale.toFixed(2));
    el.classList.remove("pin-hidden", "pin-dot");
  });
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

function renderHtmlCandidateMarkers(candidates) {
  activeCandidateItems = buildCandidateMarkerItems(candidates);
  refreshCandidateMarkers();
}

function buildCandidateMarkerItems(candidates) {
  return candidates.slice(0, topLimit).flatMap((item, idx) => {
    const coord = operatorMapCoord(item);
    if (!coord) return [];
    return [{ item, index: idx, coord }];
  });
}

function refreshCandidateMarkers() {
  clearCandidateMarkers();
  if (!liveMap) return;

  clusterCandidateItems(activeCandidateItems).forEach((cluster) => {
    const isCluster = cluster.items.length > 1;
    const topItem = cluster.items[0];
    const topIndex = topItem.index;

    const el = document.createElement("button");
    el.type = "button";
    el.className = `geo-pin compact${isCluster ? " cluster" : ""}${cluster.items.some(({ index }) => index === 0) ? " top" : ""}${cluster.items.some(({ index }) => index === selectedIndex) ? " selected" : ""}`;
    el.dataset.indices = cluster.items.map(({ index }) => index).join(",");
    el.setAttribute(
      "aria-label",
      isCluster ? `Zoom to ${cluster.items.length} grouped geo candidates` : `Select geo candidate ${topIndex + 1}`
    );
    el.innerHTML = `<span class="geo-pin-head">${isCluster ? cluster.items.length : topIndex + 1}</span>`;
    el.addEventListener("click", (event) => {
      event.stopPropagation();
      if (isCluster) {
        if (liveMap.getZoom() >= 16) {
          selectCandidate(topIndex);
        } else {
          easeToCentered({
            center: [cluster.coord.lon, cluster.coord.lat],
            zoom: Math.min(liveMap.getZoom() + 2.2, 16),
            pitch: 0,
            bearing: 0,
            duration: 650
          });
        }
      } else {
        selectCandidate(topIndex);
      }
    });

    const marker = new maplibregl.Marker({
      element: el,
      anchor: "center",
      offset: [0, 0]
    })
      .setLngLat([cluster.coord.lon, cluster.coord.lat])
      .addTo(liveMap);
    candidateMarkers.push(marker);
  });

  updatePinScale();
}

function clusterCandidateItems(items) {
  if (!liveMap) return [];
  const clusters = [];

  items.forEach((item) => {
    const point = liveMap.project([item.coord.lon, item.coord.lat]);
    let target = null;
    let targetDistance = Infinity;

    clusters.forEach((cluster) => {
      const distance = Math.hypot(point.x - cluster.point.x, point.y - cluster.point.y);
      if (distance < candidateClusterRadiusPx && distance < targetDistance) {
        target = cluster;
        targetDistance = distance;
      }
    });

    if (!target) {
      clusters.push({
        items: [item],
        point,
        coord: { ...item.coord }
      });
      return;
    }

    target.items.push(item);
    target.point = averageClusterPoint(target.items);
    target.coord = averageClusterCoord(target.items);
  });

  return clusters;
}

function averageClusterPoint(items) {
  const sum = items.reduce((acc, item) => {
    const point = liveMap.project([item.coord.lon, item.coord.lat]);
    return { x: acc.x + point.x, y: acc.y + point.y };
  }, { x: 0, y: 0 });
  return { x: sum.x / items.length, y: sum.y / items.length };
}

function averageClusterCoord(items) {
  const sum = items.reduce((acc, item) => ({
    lat: acc.lat + item.coord.lat,
    lon: acc.lon + item.coord.lon
  }), { lat: 0, lon: 0 });
  return { lat: sum.lat / items.length, lon: sum.lon / items.length };
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
      liveMap.on("zoom", refreshCandidateMarkers);
      liveMap.on("move", refreshCandidateMarkers);
      liveMap.on("zoomend", () => {
        if (liveMap.getZoom() <= globePitchResetZoom && liveMap.getPitch() !== 0) {
          easeToCentered({ center: liveMap.getCenter(), pitch: 0, duration: 180 });
        }
        refreshCandidateMarkers();
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
      updateSelectedMarker(-1);
      updateHtmlMarkerSelection(-1);
      const noteInput = byId("operator-note-input");
      if (noteInput) noteInput.value = "";
      return;
  }

  selectedIndex = index;
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
  return postForm("/api/operator/confirm", JSON.stringify({ rank: index + 1, action }));
}

function candidateWeight(item) {
  return item?.posterior_weight ?? item?.posterior ?? item?.score ?? 0;
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
    const mapCoord = operatorMapCoord(cand);
    const lat = mapCoord?.lat ?? candidateMapLat(cand);
    const lon = mapCoord?.lon ?? candidateMapLon(cand);
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

    container.appendChild(card);
  });
  
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
    svApplyZoom(1);
    if (svState.minimapMarkers) {
        svState.minimapMarkers.forEach(m => m.remove());
        svState.minimapMarkers = [];
    }
}

// ── Zoom ─────────────────────────────────────────────────────────
let svZoom = 1;
const SV_ZOOM_MIN = 0.5, SV_ZOOM_MAX = 5, SV_ZOOM_STEP = 0.25;

function svApplyZoom(z) {
    svZoom = Math.max(SV_ZOOM_MIN, Math.min(SV_ZOOM_MAX, z));
    const img = byId("sv-img");
    if (img) img.style.transform = `scale(${svZoom})`;
    const lbl = byId("sv-zoom-label");
    if (lbl) lbl.textContent = `${Math.round(svZoom * 100)}%`;
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
  const fusion = result?.fused_estimate;
  const candidates = sortedCandidates(result);
  
  if (!fusion || candidates.length === 0) return;

  const visibleCandidates = candidates.slice(0, topLimit);

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
    liveMap.getSource("candidates").setData(emptyFeatureCollection);
    renderHtmlCandidateMarkers(visibleCandidates);
    liveMap.getSource("ring").setData({ type: "FeatureCollection", features: ringFeature ? [ringFeature] : [] });
    liveMap.getSource("mean").setData(emptyFeatureCollection);
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
  activeCandidateItems = [];
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
      svApplyZoom(1);
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
    if (!select) return;
    fetch("/api/operator/sessions")
        .then(r => r.json())
        .then(data => {
            select.innerHTML = '<option value="">-- Load Session --</option>';
            if (data.sessions && data.sessions.length > 0) {
                data.sessions.forEach(session => {
                    const option = document.createElement("option");
                    option.value = session.session_id;
                    const fileStr = session.source_filename ? ` - ${session.source_filename}` : "";
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

  // Do not auto-load session on page load — pins and notes only appear
  // when a session is explicitly selected and loaded by the operator.
  localStorage.removeItem("heimdallSessionId");
  loadedSessionId = null;

  loadSessionList();

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

    // Zoom buttons
    byId("sv-zoom-in")?.addEventListener("click", () => svApplyZoom(svZoom + SV_ZOOM_STEP));
    byId("sv-zoom-out")?.addEventListener("click", () => svApplyZoom(svZoom - SV_ZOOM_STEP));
    byId("sv-zoom-reset")?.addEventListener("click", () => svApplyZoom(1));

    // Mouse wheel zoom on the image area
    byId("sv-image-wrap")?.addEventListener("wheel", (e) => {
        if (!byId("street-view-modal")?.classList.contains("active")) return;
        e.preventDefault();
        svApplyZoom(svZoom + (e.deltaY < 0 ? SV_ZOOM_STEP : -SV_ZOOM_STEP));
    }, { passive: false });

    window.addEventListener("resize", () => {
        if (byId("street-view-modal")?.classList.contains("active")) svDrawCandidateOverlays();
    });

    const confirmBtn = byId("btn-confirm-cand");
    const rejectBtn = byId("btn-reject-cand");
    const noteInput = byId("operator-note-input");
    const saveNoteBtn = byId("btn-save-note");
    const exportBtn = byId("btn-export-session");
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

    if (btnUpdateSession && sessionSaveModal) {
        btnUpdateSession.addEventListener("click", () => {
            sessionSaveModal.classList.remove("active");
            postForm("/api/operator/save", JSON.stringify({ save_as_new: false }))
                .then(r => r.json())
                .then(data => {
                    if (data.session_id) {
                        loadedSessionId = data.session_id;
                        localStorage.setItem("heimdallSessionId", data.session_id);
                    }
                    loadSessionList();
                    alert("Session updated successfully.");
                });
        });
    }

    if (btnSaveNewSession && sessionSaveModal) {
        btnSaveNewSession.addEventListener("click", () => {
            sessionSaveModal.classList.remove("active");
            const sessionName = prompt("Enter a new custom name for this session (optional):");
            if (sessionName !== null) {
                postForm("/api/operator/save", JSON.stringify({ name: sessionName.trim(), save_as_new: true }))
                    .then(r => r.json())
                    .then(data => {
                        if (data.session_id) {
                            loadedSessionId = data.session_id;
                            localStorage.setItem("heimdallSessionId", data.session_id);
                        }
                        loadSessionList();
                        alert("Session saved as new.");
                    });
            }
        });
    }

    if (newSessionBtn) {
        newSessionBtn.addEventListener("click", () => {
            fetch("/api/operator/reset", { method: "POST" })
                .then(() => {
                    loadedSessionId = null;
                    localStorage.removeItem("heimdallSessionId");
                    setMetric("diag-backend", "-");
                    setMetric("diag-worker", "-");
                    setMetric("diag-tier", "-");
                    setMetric("diag-radius", "-");
                    lastResult = null;
                    selectedIndex = -1;
                    droppedPinLocation = null;

                    const progress = byId("progress");
                    const errorContainer = byId("analysis-error");
                    const previewBlock = byId("preview-block");
                    const ingestBlock = byId("ingest-block");

                    if (progress) progress.style.display = "none";
                    if (errorContainer) errorContainer.style.display = "none";
                    if (previewBlock) previewBlock.style.display = "none";
                    if (ingestBlock) ingestBlock.style.display = "block";

                    const geolocateBtn = byId("geolocate-image");
                    if (geolocateBtn) geolocateBtn.disabled = true;

                    renderCandidateList(null);
                    renderLiveMap(null);
                    renderTimeline(null);
                    renderClues(null);
                    renderNoteMarkers();
                    renderNotesList();

                    const noteInput = byId("operator-note-input");
                    if (noteInput) noteInput.value = "";
                });
        });
    }

    if (saveSessionBtn) {
        saveSessionBtn.addEventListener("click", () => {
            if (loadedSessionId && sessionSaveModal) {
                sessionSaveModal.classList.add("active");
            } else {
                const sessionName = prompt("Enter a custom name for this session (optional):");
                if (sessionName !== null) {
                    postForm("/api/operator/save", JSON.stringify({ name: sessionName.trim(), save_as_new: true }))
                        .then(r => r.json())
                        .then(data => {
                            if (data.session_id) {
                                loadedSessionId = data.session_id;
                                localStorage.setItem("heimdallSessionId", data.session_id);
                            }
                            loadSessionList();
                            alert("Session saved as new.");
                        });
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
                 }
             } else if (selectedIndex !== -1 && lastResult && lastResult.candidates) {
                 const cand = lastResult.candidates[selectedIndex];
                 targetData = {
                     target_type: "candidate",
                     rank: cand.rank,
                     source: cand.source
                 };
             } else if (droppedPinLocation) {
                 targetData = {
                     target_type: "manual_pin",
                     lat: droppedPinLocation.lat,
                     lon: droppedPinLocation.lon
                 };
             }

             postForm("/api/operator/note", JSON.stringify({note: noteInput.value, ...targetData}))
                 .then(() => {
                     saveNoteBtn.textContent = "Saved";
                     noteInput.blur();

                     // Reset drop pin location after save
                     if (droppedPinLocation && dropPinBtn) {
                         droppedPinLocation = null;
                         dropPinBtn.textContent = "📍 Drop Note Pin";
                     }

                     setTimeout(() => {
                         saveNoteBtn.textContent = "Save Note";
                     }, 2000);
                 });
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
                warn.textContent = "Please select a session first.";
                loadSessionBtn.parentElement.appendChild(warn);
                setTimeout(() => warn.remove(), 3000);
                return;
            }
            fetch(`/api/operator/sessions/${sid}`)
                .then(r => r.json())
                .then(data => {
                     loadedSessionId = data.session_id || sid;
                     localStorage.setItem("heimdallSessionId", loadedSessionId);

                     // Full mock of lastResult structure needed by UI renderers
                     lastResult = {
                         geo: data.fused_estimate,
                         candidates: data.candidates,
                         clues: data.clues,
                         detections: data.detections
                     };
                     renderSummary(data);
                     renderCandidateList(lastResult);
                     renderLiveMap(lastResult);
                     renderTimeline(data);
                     renderClues(data);
                     renderNoteMarkers();
                     renderNotesList();
                     if (data.operator_notes) {
                         const noteInput = byId("operator-note-input");
                         if (noteInput) noteInput.value = data.operator_notes;
                     }
                })
                .catch(err => console.error("Failed to load session:", err));
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
                      dropPinBtn.textContent = "📍 Dropped";
                      dropPinBtn.style.color = "";
                  }

                  // Deselect candidate if dropping a manual pin
                  selectedIndex = -1;
                  renderCandidates();

                  const noteInput = byId("operator-note-input");
                  if (noteInput) noteInput.value = "";

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
                if (data.candidates && data.candidates[note.rank - 1]) {
                    lat = data.candidates[note.rank - 1].lat;
                    lon = data.candidates[note.rank - 1].lon;
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

function renderNotesList() {
    const list = document.getElementById("notes-list");
    if (!list) return;

    fetch("/api/operator/session").then(r => r.json()).then(data => {
        if (!data.notes || data.notes.length === 0) {
            list.innerHTML = '<div class="empty-state">No notes saved.</div>';
            return;
        }

        list.innerHTML = "";
        data.notes.forEach(note => {
            const card = document.createElement("div");
            card.className = "result-card";
            card.style.marginBottom = "8px";
            card.style.cursor = "pointer";

            let targetHtml = "";
            let lat = 0, lon = 0;
            if (note.target_type === "candidate") {
                targetHtml = `<div style="color: var(--accent); font-size: 10px; margin-bottom: 4px;">Candidate Rank ${note.rank} • ${note.source}</div>`;
                if (data.candidates && data.candidates[note.rank - 1]) {
                    lat = data.candidates[note.rank - 1].lat;
                    lon = data.candidates[note.rank - 1].lon;
                }
            } else if (note.target_type === "manual_pin") {
                lat = note.lat;
                lon = note.lon;
                targetHtml = `<div style="color: #ff4444; font-size: 10px; margin-bottom: 4px;">Manual Pin • ${lat.toFixed(4)}, ${lon.toFixed(4)}</div>`;
            }

            const timeStr = note.timestamp ? note.timestamp.split(" ").slice(1).join(" ") : "";

            card.innerHTML = `
                ${targetHtml}
                <div style="font-size: 12px; color: #fff; line-height: 1.4; margin-bottom: 6px;">${note.text}</div>
                <div style="font-size: 9px; color: var(--text-secondary); text-align: right;">${timeStr}</div>
            `;
            if (note.note_id) card.dataset.noteId = note.note_id;

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
        });
    });
}
