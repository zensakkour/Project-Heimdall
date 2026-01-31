"""
Scoring logic (placeholder). Replace with the real Heimdall score.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .config import ScoreConfig
from .types import Detection, GeoEstimate, Verification


def compute_score(
    detections: Sequence[Detection],
    geo: Optional[GeoEstimate],
    verification: Optional[Verification],
    config: Optional[ScoreConfig] = None,
) -> float:
    """Compute a basic score from available signals.

    This is a stub to be replaced with the real weighting strategy.
    """
    cfg = config or ScoreConfig()
    components: list[tuple[float, float]] = []

    if detections:
        top = sorted((d.confidence for d in detections), reverse=True)[: max(1, cfg.detection_top_k)]
        det_score = sum(max(0.0, min(1.0, c)) for c in top) / len(top)
        components.append((det_score, cfg.detection_weight))

    if geo is not None:
        geo_score = max(0.0, min(1.0, geo.confidence))
        if geo.uncertainty_m is not None:
            if geo.uncertainty_m <= 50:
                geo_score *= 1.0
            elif geo.uncertainty_m <= 200:
                geo_score *= 0.85
            elif geo.uncertainty_m <= 1000:
                geo_score *= 0.6
            else:
                geo_score *= 0.4
        components.append((geo_score, cfg.geo_weight))

    if verification is not None:
        components.append((1.0 if verification.shadow_ok else 0.0, cfg.shadow_weight))
        components.append((1.0 if verification.topo_ok else 0.0, cfg.topo_weight))

    if not components:
        return 0.0

    total_weight = sum(weight for _, weight in components if weight > 0)
    if total_weight <= 0:
        return 0.0

    weighted = sum(value * weight for value, weight in components)
    return round(weighted / total_weight, 3)


