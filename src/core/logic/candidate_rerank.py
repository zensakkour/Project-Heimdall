"""
Feature-based candidate reranking for geo fusion.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from .types import GeoCandidate


FEATURE_NAMES = (
    "bias",
    "rank_frac",
    "retrieval_unit",
    "retrieval_gap_top",
    "retrieval_gap_prev",
    "support_1km",
    "support_3km",
    "support_8km",
    "support_20km",
    "dist_top_km",
    "dist_centroid_km",
    "source_retrieval",
    "source_consensus",
    "source_geoclip",
)


@dataclass(frozen=True)
class CandidateRerankModel:
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    intercept: float = 0.0
    means: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()
    output_floor: float = 0.05
    output_ceiling: float = 1.0
    activation: str = "sigmoid"


def candidate_rerank_likelihoods(
    candidates: Sequence[GeoCandidate],
    model: CandidateRerankModel,
) -> List[float]:
    if not candidates:
        return []
    matrix = candidate_feature_matrix(candidates, model.feature_names)
    likes: List[float] = []
    for row in matrix:
        value = float(model.intercept)
        for idx, raw in enumerate(row):
            scale = model.scales[idx] if idx < len(model.scales) else 1.0
            mean = model.means[idx] if idx < len(model.means) else 0.0
            if abs(scale) < 1e-12:
                scaled = 0.0
            else:
                scaled = (raw - mean) / scale
            weight = model.weights[idx] if idx < len(model.weights) else 0.0
            value += weight * scaled
        likes.append(_activate(value, model.activation, model.output_floor, model.output_ceiling))
    peak = max(likes) if likes else 0.0
    if peak <= 0.0:
        return [1.0 for _ in candidates]
    return [max(model.output_floor, min(model.output_ceiling, value / peak)) for value in likes]


def candidate_feature_matrix(
    candidates: Sequence[GeoCandidate],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> List[List[float]]:
    scores = [_to_unit_interval(cand.retrieval_score) for cand in candidates]
    top_score = scores[0] if scores else 0.5
    top = candidates[0] if candidates else None
    centroid = _weighted_centroid(candidates, scores)
    supports = {
        1: _support_likelihoods(candidates, scores, 1.0),
        3: _support_likelihoods(candidates, scores, 3.0),
        8: _support_likelihoods(candidates, scores, 8.0),
        20: _support_likelihoods(candidates, scores, 20.0),
    }

    rows: List[List[float]] = []
    n = max(1, len(candidates) - 1)
    prev_score = top_score
    for idx, cand in enumerate(candidates):
        score = scores[idx]
        source = candidate_source_key(cand)
        values = {
            "bias": 1.0,
            "rank_frac": float(idx) / float(n),
            "retrieval_unit": score,
            "retrieval_gap_top": max(0.0, top_score - score),
            "retrieval_gap_prev": max(0.0, prev_score - score) if idx > 0 else 0.0,
            "support_1km": supports[1][idx],
            "support_3km": supports[3][idx],
            "support_8km": supports[8][idx],
            "support_20km": supports[20][idx],
            "dist_top_km": _haversine_km(cand, top) if top is not None else 0.0,
            "dist_centroid_km": _distance_to_point_km(cand, centroid),
            "source_retrieval": 1.0 if source == "retrieval" else 0.0,
            "source_consensus": 1.0 if source == "consensus" else 0.0,
            "source_geoclip": 1.0 if source == "geoclip" else 0.0,
        }
        rows.append([float(values.get(name, 0.0)) for name in feature_names])
        prev_score = score
    return rows


def candidate_source_key(cand: GeoCandidate) -> str:
    match_id = cand.match_id or ""
    if match_id == "geoclip" or match_id.startswith("geoclip:"):
        return "geoclip"
    if match_id.startswith("retrieval:consensus"):
        return "consensus"
    return "retrieval"


def load_candidate_rerank_model(path: str) -> CandidateRerankModel:
    resolved = str(Path(path).expanduser().resolve())
    return _load_candidate_rerank_model_cached(resolved)


@lru_cache(maxsize=16)
def _load_candidate_rerank_model_cached(path: str) -> CandidateRerankModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    names = tuple(str(item) for item in payload.get("feature_names", FEATURE_NAMES))
    weights = tuple(float(item) for item in payload.get("weights", []))
    if len(weights) < len(names):
        weights = weights + tuple(0.0 for _ in range(len(names) - len(weights)))
    return CandidateRerankModel(
        feature_names=names,
        weights=weights[: len(names)],
        intercept=float(payload.get("intercept", payload.get("bias", 0.0))),
        means=tuple(float(item) for item in payload.get("means", [])),
        scales=tuple(float(item) for item in payload.get("scales", [])),
        output_floor=float(payload.get("output_floor", 0.05)),
        output_ceiling=float(payload.get("output_ceiling", 1.0)),
        activation=str(payload.get("activation", "sigmoid")).lower(),
    )


def model_to_json(model: CandidateRerankModel, extra: Optional[Mapping[str, object]] = None) -> dict:
    payload = {
        "version": 1,
        "feature_names": list(model.feature_names),
        "weights": list(model.weights),
        "intercept": model.intercept,
        "means": list(model.means),
        "scales": list(model.scales),
        "activation": model.activation,
        "output_floor": model.output_floor,
        "output_ceiling": model.output_ceiling,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _activate(value: float, activation: str, floor: float, ceiling: float) -> float:
    lo = max(1e-6, min(1.0, float(floor)))
    hi = max(lo, min(1.0, float(ceiling)))
    if activation == "linear_clamp":
        return max(lo, min(hi, value))
    prob = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))
    return lo + (hi - lo) * prob


def _support_likelihoods(candidates: Sequence[GeoCandidate], scores: Sequence[float], sigma_km: float) -> List[float]:
    if not candidates:
        return []
    sigma = max(0.1, float(sigma_km))
    priors = [max(1e-6, float(score)) for score in scores]
    total = sum(priors)
    if total <= 0.0:
        priors = [1.0 / len(candidates) for _ in candidates]
    else:
        priors = [score / total for score in priors]
    raw = []
    for idx_i, cand_i in enumerate(candidates):
        support = 0.0
        for idx_j, cand_j in enumerate(candidates):
            dist = _haversine_km(cand_i, cand_j)
            self_boost = 0.35 if idx_i == idx_j else 1.0
            support += priors[idx_j] * self_boost * math.exp(-0.5 * (dist / sigma) ** 2)
        raw.append(support)
    peak = max(raw) if raw else 0.0
    if peak <= 0.0:
        return [1.0 for _ in candidates]
    return [max(1e-6, min(1.0, value / peak)) for value in raw]


def _weighted_centroid(candidates: Sequence[GeoCandidate], scores: Sequence[float]) -> tuple[float, float]:
    if not candidates:
        return 0.0, 0.0
    weights = [max(1e-6, float(score)) for score in scores]
    total = sum(weights)
    if total <= 0.0:
        weights = [1.0 for _ in candidates]
        total = float(len(candidates))
    lat = sum(cand.latitude * weight for cand, weight in zip(candidates, weights)) / total
    lon = _weighted_circular_mean_deg([cand.longitude for cand in candidates], weights)
    return lat, lon


def _distance_to_point_km(cand: GeoCandidate, point: tuple[float, float]) -> float:
    return _haversine_points_km(cand.latitude, cand.longitude, point[0], point[1])


def _haversine_km(a: GeoCandidate, b: Optional[GeoCandidate]) -> float:
    if b is None:
        return 0.0
    return _haversine_points_km(a.latitude, a.longitude, b.latitude, b.longitude)


def _haversine_points_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    term = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    arc = 2.0 * math.atan2(math.sqrt(term), math.sqrt(max(1.0 - term, 0.0)))
    return radius_km * arc


def _weighted_circular_mean_deg(angles_deg: Sequence[float], weights: Sequence[float]) -> float:
    sin_sum = sum(weight * math.sin(math.radians(angle)) for angle, weight in zip(angles_deg, weights))
    cos_sum = sum(weight * math.cos(math.radians(angle)) for angle, weight in zip(angles_deg, weights))
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        total = sum(weights) or 1.0
        return _normalize_longitude(sum(angle * weight for angle, weight in zip(angles_deg, weights)) / total)
    return _normalize_longitude(math.degrees(math.atan2(sin_sum, cos_sum)))


def _normalize_longitude(lon: float) -> float:
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and lon > 0.0:
        return 180.0
    return wrapped


def _to_unit_interval(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    if 0.0 <= value <= 1.0:
        return value
    return 1.0 / (1.0 + math.exp(-value))


def flatten_rows(rows: Iterable[Iterable[float]]) -> List[List[float]]:
    return [[float(value) for value in row] for row in rows]
