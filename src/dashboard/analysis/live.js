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
  if (progress) {
    progress.classList.toggle("active", active);
    if (text && progressText) progressText.textContent = text;
  }
  if (analyzeImage) analyzeImage.disabled = active;
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
  const summary = byId("summary");
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
  summary.textContent = `Score: ${result.result.score.toFixed(3)} | Geo tier: ${
    geo?.confidence_tier || tierFromFusion() || tierFromCandidates() || "-"
  }${geoConfText} | Detections: ${result.result.detections.length}${fusionText}${debugText}`;
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
  } catch (err) {
    byId("summary").textContent = `Error: ${err.message || err}`;
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

// Initialize the globe immediately so it is visible before analysis runs.
window.addEventListener("load", () => {
  ensureLiveMap();
  if (liveMap) {
    setTimeout(() => liveMap.resize(), 0);
  }
});

