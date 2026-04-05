"""Combine multiple candidate providers."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from src.core.logic.types import GeoCandidate


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    return radius_m * c


def _is_valid_candidate(cand: GeoCandidate, min_score: float) -> bool:
    if not isinstance(cand.latitude, (int, float)) or not isinstance(cand.longitude, (int, float)):
        return False
    if not isinstance(cand.retrieval_score, (int, float)):
        return False
    if not math.isfinite(cand.latitude) or not math.isfinite(cand.longitude):
        return False
    if not math.isfinite(cand.retrieval_score):
        return False
    if not (-90.0 <= cand.latitude <= 90.0 and -180.0 <= cand.longitude <= 180.0):
        return False
    return cand.retrieval_score >= min_score


def _clamp_score(value: float) -> float:
    return max(0.0, min(0.999, float(value)))


@dataclass
class _CandidateCluster:
    lat_weighted_sum: float = 0.0
    lon_weighted_sum: float = 0.0
    total_weight: float = 0.0
    scores: List[float] = field(default_factory=list)
    best: Optional[GeoCandidate] = None
    first_non_null_match_id: Optional[str] = None

    def add(self, cand: GeoCandidate) -> None:
        weight = max(_clamp_score(cand.retrieval_score), 0.05)
        self.lat_weighted_sum += cand.latitude * weight
        self.lon_weighted_sum += cand.longitude * weight
        self.total_weight += weight
        self.scores.append(_clamp_score(cand.retrieval_score))
        if self.best is None or cand.retrieval_score > self.best.retrieval_score:
            self.best = cand
        if self.first_non_null_match_id is None and cand.match_id:
            self.first_non_null_match_id = cand.match_id

    @property
    def centroid_lat(self) -> float:
        if self.total_weight <= 0.0:
            return 0.0
        return self.lat_weighted_sum / self.total_weight

    @property
    def centroid_lon(self) -> float:
        if self.total_weight <= 0.0:
            return 0.0
        return self.lon_weighted_sum / self.total_weight

    def to_candidate(self) -> GeoCandidate:
        # Independent-evidence style merge: repeated near-identical matches increase confidence.
        score = 1.0
        for s in self.scores:
            score *= 1.0 - s
        merged_score = max(0.0, min(1.0, 1.0 - score))
        match_id = self.best.match_id if self.best and self.best.match_id else self.first_non_null_match_id
        return GeoCandidate(
            latitude=self.centroid_lat,
            longitude=self.centroid_lon,
            retrieval_score=merged_score,
            match_id=match_id,
        )


class MultiCandidateProvider:
    def __init__(
        self,
        providers: Iterable[object],
        dedupe_radius_m: float = 300.0,
        max_candidates: int = 80,
        min_score: float = 1e-3,
    ) -> None:
        self.providers = [p for p in providers if p is not None]
        self.dedupe_radius_m = max(0.0, float(dedupe_radius_m))
        self.max_candidates = max(1, int(max_candidates))
        self.min_score = max(0.0, float(min_score))
        self.last_error: Optional[str] = None

    def candidates(self, image_path: str) -> List[GeoCandidate]:
        gathered: List[GeoCandidate] = []
        errors = []
        for provider in self.providers:
            try:
                items = provider.candidates(image_path) or []
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(str(exc))
                continue
            for item in items:
                if _is_valid_candidate(item, self.min_score):
                    gathered.append(item)
            err = getattr(provider, "last_error", None)
            if err:
                errors.append(str(err))
        self.last_error = "; ".join(errors) if errors else None
        if not gathered:
            return []
        merged = self._dedupe_candidates(gathered)
        merged.sort(key=lambda cand: cand.retrieval_score, reverse=True)
        return merged[: self.max_candidates]

    def _dedupe_candidates(self, candidates: List[GeoCandidate]) -> List[GeoCandidate]:
        if self.dedupe_radius_m <= 0.0:
            return candidates
        ranked = sorted(candidates, key=lambda cand: cand.retrieval_score, reverse=True)
        clusters: List[_CandidateCluster] = []
        for cand in ranked:
            cluster = self._find_cluster(clusters, cand)
            if cluster is None:
                cluster = _CandidateCluster()
                clusters.append(cluster)
            cluster.add(cand)
        return [cluster.to_candidate() for cluster in clusters]

    def _find_cluster(self, clusters: List[_CandidateCluster], cand: GeoCandidate) -> Optional[_CandidateCluster]:
        for cluster in clusters:
            distance_m = _haversine_m(
                cand.latitude,
                cand.longitude,
                cluster.centroid_lat,
                cluster.centroid_lon,
            )
            if distance_m <= self.dedupe_radius_m:
                return cluster
        return None
