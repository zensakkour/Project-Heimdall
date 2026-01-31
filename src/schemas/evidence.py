from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class EvidenceModel(BaseModel):
    retrieval_score: float
    shadow_residual_deg: Optional[float] = None
    terrain_residual: Optional[float] = None
    likelihoods: Dict[str, float]
    posterior_weight: float
    explanation: str


