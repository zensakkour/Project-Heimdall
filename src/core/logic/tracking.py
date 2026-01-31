"""
Tracking association stubs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .types import FusionResult


@dataclass(frozen=True)
class TrackState:
    track_id: str
    fused_history: List[FusionResult]


def associate_track(
    existing: List[TrackState],
    fused: FusionResult,
    max_distance_m: float = 5000.0,
) -> Optional[TrackState]:
    for track in existing:
        last = track.fused_history[-1]
        if _rough_distance_m(last, fused) <= max_distance_m:
            return track
    return None


def _rough_distance_m(a: FusionResult, b: FusionResult) -> float:
    lat0 = a.mean_latitude
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * __import__("math").cos(__import__("math").radians(lat0))
    dx = (b.mean_longitude - a.mean_longitude) * meters_per_deg_lon
    dy = (b.mean_latitude - a.mean_latitude) * meters_per_deg_lat
    return (dx * dx + dy * dy) ** 0.5


