"""
Lightweight domain types for the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    # Oriented bounding box: 4 points (x, y)
    obb: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    # Optional metadata for downstream heuristics
    heading_deg: Optional[float] = None
    shadow_azimuth_deg: Optional[float] = None
    shadow_length_ratio: Optional[float] = None


@dataclass(frozen=True)
class GeoEstimate:
    latitude: float
    longitude: float
    confidence: float
    # Optional metadata for explainability
    landmarks: Optional[List[str]] = None
    uncertainty_m: Optional[float] = None


@dataclass(frozen=True)
class Verification:
    shadow_ok: bool
    topo_ok: bool
    notes: Optional[str] = None


@dataclass(frozen=True)
class Assessment:
    detections: List[Detection]
    geo: Optional[GeoEstimate]
    verification: Optional[Verification]
    score: float
