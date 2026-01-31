from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GeoCandidateModel(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    retrieval_score: float = Field(..., description="Raw retrieval score")
    match_id: Optional[str] = None


