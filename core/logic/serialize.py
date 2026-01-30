"""
Serialization helpers for pipeline outputs.
"""
from __future__ import annotations

from typing import Any

from .types import Assessment


def assessment_to_dict(result: Assessment) -> dict[str, Any]:
    return {
        "detections": [
            {
                "label": d.label,
                "confidence": d.confidence,
                "obb": d.obb,
                "heading_deg": d.heading_deg,
                "shadow_azimuth_deg": d.shadow_azimuth_deg,
                "shadow_length_ratio": d.shadow_length_ratio,
            }
            for d in result.detections
        ],
        "geo": None
        if result.geo is None
        else {
            "latitude": result.geo.latitude,
            "longitude": result.geo.longitude,
            "confidence": result.geo.confidence,
            "landmarks": result.geo.landmarks,
            "uncertainty_m": result.geo.uncertainty_m,
            "uncertainty_radius_m": result.geo.uncertainty_m,
            "confidence_tier": _geo_confidence_tier(result.geo),
        },
        "verification": None
        if result.verification is None
        else {
            "shadow_ok": result.verification.shadow_ok,
            "topo_ok": result.verification.topo_ok,
            "notes": result.verification.notes,
        },
        "score": result.score,
    }


def _geo_confidence_tier(geo: "GeoEstimate") -> str:
    score = max(0.0, min(1.0, geo.confidence))
    if geo.uncertainty_m is not None:
        if geo.uncertainty_m > 1000:
            score *= 0.4
        elif geo.uncertainty_m > 200:
            score *= 0.6
        elif geo.uncertainty_m > 50:
            score *= 0.85
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"
