"""
Lightweight domain types for the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
class GeoCandidate:
    latitude: float
    longitude: float
    retrieval_score: float
    match_id: Optional[str] = None


@dataclass(frozen=True)
class Evidence:
    retrieval_score: float
    shadow_residual_deg: Optional[float]
    terrain_residual: Optional[float]
    likelihoods: Dict[str, float]
    posterior_weight: float
    explanation: str


@dataclass(frozen=True)
class FusionCandidate:
    candidate: GeoCandidate
    posterior_weight: float
    evidence: Evidence


@dataclass(frozen=True)
class UncertaintyEllipse:
    major_axis_m: float
    minor_axis_m: float
    orientation_deg: float


@dataclass(frozen=True)
class FusionResult:
    candidates: List[FusionCandidate]
    mean_latitude: float
    mean_longitude: float
    covariance_m: Tuple[Tuple[float, float], Tuple[float, float]]
    ellipse: UncertaintyEllipse
    uncertainty_radius_m: float


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
    candidates: List[GeoCandidate] = field(default_factory=list)
    fusion: Optional["FusionResult"] = None


