# Project-Heimdall
Watchman of the Gods - Sovereign Intelligence

## 1. Core Architecture: The "Heimdall Vision"

Project Heimdall leverages high-performance C++ and Python to bridge the gap between "noisy" social media feeds (Telegram etc) and high-accuracy situational awareness.

### **The Technical Stack**

- **Detection:** **YOLOv11-OBB** (Oriented Bounding Boxes). Unlike standard boxes, OBB detects the exact rotation of a target, providing a "heading" that is essential for map-matching.
- **Geolocation:** **GeoFT / GeoCLIP**. A pixel-to-coordinate engine that treats geolocation as a retrieval problem, matching ground-level features against a global satellite database.
- **Intelligence:** **Telethon / Pyrogram** for high-speed, asynchronous monitoring of specific "war-reporting" channels.
- **Deep Tech Integration:** **PennyLane**. A hybrid Quantum-Classical circuit used to enhance classification confidence in low-resolution or obfuscated imagery.

---

## 2. Phase-by-Phase Setup

### **Phase 1: Foundation (Weeks 1-4)**

- **Environment:** Ubuntu 22.04+, CUDA 12.x, Python 3.10+.
- **Dataset Setup:** * **DOTA v1.0:** Primary training set for oriented targets (tanks, ships, aircraft).
    - **MSTAR:** Specialized dataset for Synthetic Aperture Radar (SAR) recognition benchmarks.
- **Labeling:** Use **Roboflow** for OBB annotations, converting traditional DOTA formats to the `class_index x1 y1 x2 y2 x3 y3 x4 y4` YOLO format.

### **Phase 2: The "Heimdall" Geolocation Core (Weeks 5-8)**

- **The Engine:** Fork **GeoFT** and replace its backbone with a **C3k2-optimized** feature extractor to improve small-object detection in wide landscapes.
- **Shadow Verification (Chronolocation):** Implement a Python module using **SunCalc**. It matches the shadow angles of detected vehicles to the sun's position at the estimated GPS coordinates and time to provide "verified" confirmation.
- **Topographic Matching:** Cross-reference background mountain silhouettes with **NASA SRTM** data to ensure the horizon matches the predicted coordinates.

### **Phase 3: Live Mapping & Scale (Weeks 9-12)**

- **Ingestion:** Set up a **Telethon** bot to scrape images and videos from 20+ priority Telegram channels.
- **Heimdall Score:** Develop a proprietary "Confirmation Score" that weights visual matches, shadow analysis, and topographic verification.
- **Output:** A live **Mapbox** dashboard showing "Verified Threats" with a timestamp and high-confidence location pins.

---

## 3. VC & Defense Strategy

To secure funding (e.g., from defense-focused firms like **Anduril** or **Helsing**), Project Heimdall emphasizes **Sovereign Intelligence**.

- **GPS-Denied Capability:** Visual geolocation is the primary fallback when GPS is jammed or spoofed.
- **Explainability:** Unlike "black box" AI, Heimdall outputs the specific visual landmarks used for the location, providing a "why" for human intelligence analysts.
- **Computational Efficiency:** By optimizing the C++ core for **TensorRT**, Heimdall is designed to run on-device for autonomous drones or tactical ruggedized tablets.

---

### **Project Structure**

`Heimdall/
├── core/
│   ├── detection/ (YOLOv11-OBB models)
│   ├── geo/       (GeoFT/GeoCLIP alignment logic)
│   └── logic/     (Chronolocation and Terrain verification)
├── ingestion/
│   └── telegram/  (Telethon scraper and parsers)
├── data/
│   ├── dota/      (Aerial oriented images)
│   └── weights/   (Trained yolo11-obb.pt)
└── dashboard/     (Mapbox live visualization)`
