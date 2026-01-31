from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from .fusion import FusionOutputModel


class TrackObservationModel(BaseModel):
    image_id: str
    fused: FusionOutputModel
    timestamp: Optional[str] = None


class TrackModel(BaseModel):
    track_id: str
    observations: List[TrackObservationModel]


