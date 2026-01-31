from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, Field

from .evidence import EvidenceModel
from .geo_candidate import GeoCandidateModel


class FusedCandidateModel(BaseModel):
    candidate: GeoCandidateModel
    posterior_weight: float = Field(..., ge=0.0, le=1.0)
    evidence: EvidenceModel


class UncertaintyEllipseModel(BaseModel):
    major_axis_m: float
    minor_axis_m: float
    orientation_deg: float


class FusionOutputModel(BaseModel):
    candidates: List[FusedCandidateModel]
    mean_latitude: float
    mean_longitude: float
    covariance_m: Tuple[Tuple[float, float], Tuple[float, float]]
    ellipse: UncertaintyEllipseModel
    uncertainty_radius_m: float


