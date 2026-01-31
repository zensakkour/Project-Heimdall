"""
GeoFT/GeoCLIP geolocation adapter stub.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.core.logic.types import GeoEstimate
from src.core.logic.image_meta import extract_gps


class GeoLocator:
    def __init__(
        self,
        model_path: str | None = None,
        use_sidecar: bool = True,
        use_exif: bool = True,
    ) -> None:
        self.model_path = model_path
        self.use_sidecar = use_sidecar
        self.use_exif = use_exif
        # TODO: load model when integrating real model.

    def predict(self, image_path: str) -> Optional[GeoEstimate]:
        if self.use_sidecar:
            sidecar = _load_sidecar_geo(image_path)
            if sidecar is not None:
                return sidecar

        if self.use_exif:
            gps = extract_gps(image_path)
            if gps is not None:
                latitude, longitude = gps
                if -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0:
                    return GeoEstimate(
                        latitude=latitude,
                        longitude=longitude,
                        confidence=0.6,
                        landmarks=["exif:gps"],
                        uncertainty_m=50.0,
                    )
        return None


def _load_sidecar_geo(image_path: str) -> Optional[GeoEstimate]:
    path = Path(image_path)
    candidates = [
        Path(str(path) + ".geo.json"),
        path.with_suffix(".geo.json"),
        Path(str(path) + ".geoloc.json"),
        path.with_suffix(".geoloc.json"),
    ]
    sidecar = next((p for p in candidates if p.exists()), None)
    if sidecar is None:
        return None

    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(raw, dict):
        return None

    latitude = raw.get("latitude")
    longitude = raw.get("longitude")
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return None
    if not (-90.0 <= float(latitude) <= 90.0 and -180.0 <= float(longitude) <= 180.0):
        return None

    confidence = raw.get("confidence", 0.6)
    if not isinstance(confidence, (int, float)):
        confidence = 0.6

    landmarks = raw.get("landmarks")
    if landmarks is not None and not isinstance(landmarks, list):
        landmarks = None
    uncertainty = raw.get("uncertainty_m")
    if uncertainty is not None and not isinstance(uncertainty, (int, float)):
        uncertainty = None

    return GeoEstimate(
        latitude=float(latitude),
        longitude=float(longitude),
        confidence=float(confidence),
        landmarks=landmarks,
        uncertainty_m=None if uncertainty is None else float(uncertainty),
    )


