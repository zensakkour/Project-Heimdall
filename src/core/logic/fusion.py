"""
Probabilistic fusion of geo candidates and verification signals.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from .astro import sun_position
from .config import FusionConfig
from .image_meta import extract_capture_time
from .likelihoods import gaussian_likelihood
from .types import (
    Detection,
    Evidence,
    FusionCandidate,
    FusionResult,
    GeoCandidate,
    UncertaintyEllipse,
)


def fuse_candidates(
    image_path: str,
    candidates: Sequence[GeoCandidate],
    detections: Sequence[Detection],
    config: Optional[FusionConfig] = None,
) -> Optional[FusionResult]:
    if not candidates:
        return None
    cfg = config or FusionConfig()

    log_weights: List[float] = []
    evidences: List[Evidence] = []

    capture_time = extract_capture_time(image_path)
    observed_shadow = _mean_shadow_azimuth(detections)
    retrieval_scores = _normalized_retrieval_scores(candidates, cfg.retrieval_score_norm)

    for cand, norm_score in zip(candidates, retrieval_scores):
        retrieval_logp = _temperature_scaled_logprob(norm_score, cfg.retrieval_temperature)
        shadow_residual = None
        shadow_like = None
        if cfg.use_shadow and capture_time is not None and observed_shadow is not None:
            expected_shadow = _expected_shadow_azimuth(capture_time, cand.latitude, cand.longitude)
            shadow_residual = _angular_diff(expected_shadow, observed_shadow)
            shadow_like = gaussian_likelihood(shadow_residual, cfg.shadow_sigma_deg)

        terrain_residual = None
        terrain_like = None
        if cfg.use_terrain:
            terrain_residual = 0.0
            terrain_like = gaussian_likelihood(terrain_residual, cfg.terrain_sigma)

        logp = retrieval_logp
        likelihoods = {"retrieval": math.exp(retrieval_logp)}
        if shadow_like is not None:
            logp += math.log(max(shadow_like, 1e-12))
            likelihoods["shadow"] = shadow_like
        if terrain_like is not None:
            logp += math.log(max(terrain_like, 1e-12))
            likelihoods["terrain"] = terrain_like

        log_weights.append(logp)
        evidences.append(
            Evidence(
                retrieval_score=norm_score,
                shadow_residual_deg=shadow_residual,
                terrain_residual=terrain_residual,
                likelihoods=likelihoods,
                posterior_weight=0.0,
                explanation=_explain_candidate(cand, norm_score, shadow_residual, terrain_residual, likelihoods),
            )
        )

    weights = _softmax(log_weights)

    fused: List[FusionCandidate] = []
    for cand, weight, evidence in zip(candidates, weights, evidences):
        fused.append(
            FusionCandidate(
                candidate=cand,
                posterior_weight=weight,
                evidence=Evidence(
                    retrieval_score=evidence.retrieval_score,
                    shadow_residual_deg=evidence.shadow_residual_deg,
                    terrain_residual=evidence.terrain_residual,
                    likelihoods=evidence.likelihoods,
                    posterior_weight=weight,
                    explanation=evidence.explanation,
                ),
            )
        )

    fused.sort(key=lambda item: item.posterior_weight, reverse=True)
    fused = fused[: cfg.top_k]

    mean_lat, mean_lon, cov = _weighted_mean_cov(fused)
    ellipse = _covariance_to_ellipse(cov)
    uncertainty_radius = max(ellipse.major_axis_m, ellipse.minor_axis_m)

    return FusionResult(
        candidates=fused,
        mean_latitude=mean_lat,
        mean_longitude=mean_lon,
        covariance_m=cov,
        ellipse=ellipse,
        uncertainty_radius_m=uncertainty_radius,
    )


def _temperature_scaled_logprob(score: float, temperature: float) -> float:
    temp = max(1e-6, temperature)
    return score / temp


def _softmax(logits: Iterable[float]) -> List[float]:
    values = list(logits)
    if not values:
        return []
    max_log = max(values)
    exp_vals = [math.exp(v - max_log) for v in values]
    total = sum(exp_vals)
    if total <= 0.0:
        return [1.0 / len(values) for _ in values]
    return [v / total for v in exp_vals]


def _normalized_retrieval_scores(candidates: Sequence[GeoCandidate], mode: str) -> List[float]:
    if mode == "none":
        return [cand.retrieval_score for cand in candidates]
    if mode == "zscore_sigmoid":
        return _zscore_sigmoid_by_source(candidates)
    return [cand.retrieval_score for cand in candidates]


def _candidate_source(cand: GeoCandidate) -> str:
    match_id = cand.match_id or ""
    if match_id.startswith("exif:"):
        return "exif"
    if match_id == "geoclip":
        return "geoclip"
    return "retrieval"


def _zscore_sigmoid(values: List[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    if std < 1e-6:
        return values[:]
    out = []
    for v in values:
        z = (v - mean) / std
        out.append(1.0 / (1.0 + math.exp(-z)))
    return out


def _zscore_sigmoid_by_source(candidates: Sequence[GeoCandidate]) -> List[float]:
    if not candidates:
        return []
    grouped = {}
    order = []
    for idx, cand in enumerate(candidates):
        src = _candidate_source(cand)
        order.append(src)
        grouped.setdefault(src, []).append((idx, cand.retrieval_score))

    scores = [cand.retrieval_score for cand in candidates]
    for items in grouped.values():
        idxs = [idx for idx, _ in items]
        vals = [val for _, val in items]
        if len(vals) < 2:
            continue
        norm = _zscore_sigmoid(vals)
        for idx, val in zip(idxs, norm):
            scores[idx] = val
    return scores


def _expected_shadow_azimuth(captured, latitude: float, longitude: float) -> float:
    azimuth, _ = sun_position(captured, latitude, longitude)
    return (azimuth + 180.0) % 360.0


def _mean_shadow_azimuth(detections: Sequence[Detection]) -> Optional[float]:
    angles = [d.shadow_azimuth_deg for d in detections if d.shadow_azimuth_deg is not None]
    if not angles:
        return None
    # Circular mean in degrees.
    sin_sum = sum(math.sin(math.radians(a)) for a in angles)
    cos_sum = sum(math.cos(math.radians(a)) for a in angles)
    mean = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0
    return mean


def _angular_diff(a: float, b: float) -> float:
    delta = abs((a - b) % 360.0)
    return min(delta, 360.0 - delta)


def _weighted_mean_cov(fused: Sequence[FusionCandidate]) -> Tuple[float, float, Tuple[Tuple[float, float], Tuple[float, float]]]:
    weights = [item.posterior_weight for item in fused]
    if not weights or sum(weights) == 0.0:
        first = fused[0]
        return first.candidate.latitude, first.candidate.longitude, ((0.0, 0.0), (0.0, 0.0))

    mean_lat = sum(item.candidate.latitude * item.posterior_weight for item in fused)
    mean_lon = sum(item.candidate.longitude * item.posterior_weight for item in fused)

    lat0 = mean_lat
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    xs = []
    ys = []
    for item in fused:
        dx = (item.candidate.longitude - mean_lon) * meters_per_deg_lon
        dy = (item.candidate.latitude - mean_lat) * meters_per_deg_lat
        xs.append(dx)
        ys.append(dy)

    cov_xx = sum(w * (x ** 2) for w, x in zip(weights, xs))
    cov_yy = sum(w * (y ** 2) for w, y in zip(weights, ys))
    cov_xy = sum(w * x * y for w, x, y in zip(weights, xs, ys))

    return mean_lat, mean_lon, ((cov_xx, cov_xy), (cov_xy, cov_yy))


def _covariance_to_ellipse(cov: Tuple[Tuple[float, float], Tuple[float, float]]) -> UncertaintyEllipse:
    (a, b), (_, d) = cov
    trace = a + d
    det = a * d - b * b
    if trace <= 0.0:
        return UncertaintyEllipse(major_axis_m=0.0, minor_axis_m=0.0, orientation_deg=0.0)

    temp = max(trace * trace / 4.0 - det, 0.0)
    root = math.sqrt(temp)
    eig1 = trace / 2.0 + root
    eig2 = trace / 2.0 - root
    major = math.sqrt(max(eig1, 0.0))
    minor = math.sqrt(max(eig2, 0.0))

    if abs(b) < 1e-9:
        orientation = 0.0
    else:
        orientation = math.degrees(0.5 * math.atan2(2.0 * b, a - d))

    return UncertaintyEllipse(major_axis_m=major, minor_axis_m=minor, orientation_deg=orientation)


def _explain_candidate(
    cand: GeoCandidate,
    normalized_score: float,
    shadow_residual: Optional[float],
    terrain_residual: Optional[float],
    likelihoods: dict[str, float],
) -> str:
    parts = [f"retrieval={cand.retrieval_score:.3f} (norm={normalized_score:.3f})"]
    if shadow_residual is not None:
        parts.append(f"shadow_residual={shadow_residual:.1f}deg")
    if terrain_residual is not None:
        parts.append(f"terrain_residual={terrain_residual:.1f}")
    parts.append(f"likelihoods={','.join(f'{k}:{v:.3g}' for k, v in likelihoods.items())}")
    return "; ".join(parts)


