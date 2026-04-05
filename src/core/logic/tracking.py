"""
Tracking association stubs.
"""
from __future__ import annotations

import math
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
        allowed = _association_distance_threshold_m(last, fused, max_distance_m)
        if _rough_distance_m(last, fused) <= allowed:
            return track
    return None


def _rough_distance_m(a: FusionResult, b: FusionResult) -> float:
    return _haversine_m(a.mean_latitude, a.mean_longitude, b.mean_latitude, b.mean_longitude)


def _association_distance_threshold_m(a: FusionResult, b: FusionResult, base_max_m: float) -> float:
    base = max(100.0, float(base_max_m))
    a_unc = max(0.0, float(a.uncertainty_radius_m))
    b_unc = max(0.0, float(b.uncertainty_radius_m))
    # Expand allowable track-join radius when either estimate is uncertain.
    return base + 0.5 * (a_unc + b_unc)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    term = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    arc = 2.0 * math.atan2(math.sqrt(term), math.sqrt(max(0.0, 1.0 - term)))
    return radius_m * arc
