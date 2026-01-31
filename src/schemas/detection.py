from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class DetectionModel(BaseModel):
    label: str = Field(..., description="Class label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    obb: List[Tuple[float, float]] = Field(..., description="4-point oriented bounding box")
    heading_deg: Optional[float] = None
    embedding: Optional[List[float]] = None


class DetectionBatch(BaseModel):
    detections: List[DetectionModel]


