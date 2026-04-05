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
    spatial_likes = _spatial_consensus_likelihoods(
        candidates,
        retrieval_scores,
        sigma_km=cfg.spatial_sigma_km,
    )

    for idx, (cand, norm_score) in enumerate(zip(candidates, retrieval_scores)):
        retrieval_like = _to_unit_interval(norm_score)
        retrieval_logp = _temperature_scaled_logprob(retrieval_like, cfg.retrieval_temperature)
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
        likelihoods = {"retrieval": retrieval_like}
        if cfg.use_spatial_consensus:
            spatial_like = spatial_likes[idx]
            logp += max(0.0, cfg.spatial_consensus_weight) * math.log(max(spatial_like, 1e-12))
            likelihoods["spatial"] = spatial_like
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
    limit = cfg.top_k if cfg.top_k > 0 else len(fused)
    fused = fused[:limit]

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
    temp = max(1e-3, temperature)
    unit = _to_unit_interval(score)
    unit = max(1e-6, min(1.0 - 1e-6, unit))
    # Log-odds keeps ordering while being numerically stable and less scale-sensitive.
    return math.log(unit / (1.0 - unit)) / temp


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
    if mode == "minmax":
        return _minmax_scale([cand.retrieval_score for cand in candidates])
    if mode == "rank_exp":
        return _rank_exp_scale([cand.retrieval_score for cand in candidates])
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
        return [0.5 for _ in values]
    out = []
    for v in values:
        z = (v - mean) / std
        out.append(1.0 / (1.0 + math.exp(-z)))
    return out


def _zscore_sigmoid_by_source(candidates: Sequence[GeoCandidate]) -> List[float]:
    if not candidates:
        return []
    grouped = {}
    for idx, cand in enumerate(candidates):
        src = _candidate_source(cand)
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


def _minmax_scale(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / span for v in values]


def _rank_exp_scale(values: List[float]) -> List[float]:
    if not values:
        return []
    ranked_idx = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    tau = max(1.0, len(values) / 3.0)
    out = [0.0 for _ in values]
    for rank, idx in enumerate(ranked_idx):
        out[idx] = math.exp(-rank / tau)
    return out


def _to_unit_interval(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    if 0.0 <= value <= 1.0:
        return value
    return 1.0 / (1.0 + math.exp(-value))


def _haversine_km(a: GeoCandidate, b: GeoCandidate) -> float:
    radius_km = 6371.0
    lat1 = math.radians(a.latitude)
    lat2 = math.radians(b.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(b.longitude - a.longitude)
    term = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    arc = 2.0 * math.atan2(math.sqrt(term), math.sqrt(max(1.0 - term, 0.0)))
    return radius_km * arc


def _spatial_consensus_likelihoods(
    candidates: Sequence[GeoCandidate],
    retrieval_scores: Sequence[float],
    sigma_km: float,
) -> List[float]:
    if not candidates:
        return []
    if len(candidates) == 1:
        return [1.0]

    sigma = max(0.1, float(sigma_km))
    priors = [max(1e-3, _to_unit_interval(score)) for score in retrieval_scores]
    total = sum(priors)
    if total <= 0.0:
        priors = [1.0 / len(candidates) for _ in candidates]
    else:
        priors = [p / total for p in priors]

    raw = []
    for cand_i in candidates:
        density = 0.0
        for cand_j, prior_j in zip(candidates, priors):
            dist = _haversine_km(cand_i, cand_j)
            kernel = math.exp(-0.5 * (dist / sigma) ** 2)
            density += prior_j * kernel
        raw.append(density)

    peak = max(raw) if raw else 0.0
    if peak <= 0.0:
        return [1.0 for _ in candidates]
    return [max(1e-3, min(1.0, value / peak)) for value in raw]


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
    if not fused:
        return 0.0, 0.0, ((0.0, 0.0), (0.0, 0.0))
    weights = [item.posterior_weight for item in fused]
    total_weight = sum(weights)
    if not weights or total_weight <= 0.0:
        first = fused[0]
        return first.candidate.latitude, first.candidate.longitude, ((0.0, 0.0), (0.0, 0.0))

    norm_weights = [w / total_weight for w in weights]
    mean_lat = sum(item.candidate.latitude * w for item, w in zip(fused, norm_weights))
    mean_lon = _weighted_circular_mean_deg(
        [item.candidate.longitude for item in fused],
        norm_weights,
    )

    lat0 = mean_lat
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = max(1e-6, abs(111_320.0 * math.cos(math.radians(lat0))))

    xs = []
    ys = []
    for item in fused:
        dlon = _wrapped_longitude_diff(item.candidate.longitude, mean_lon)
        dx = dlon * meters_per_deg_lon
        dy = (item.candidate.latitude - mean_lat) * meters_per_deg_lat
        xs.append(dx)
        ys.append(dy)

    cov_xx = sum(w * (x ** 2) for w, x in zip(norm_weights, xs))
    cov_yy = sum(w * (y ** 2) for w, y in zip(norm_weights, ys))
    cov_xy = sum(w * x * y for w, x, y in zip(norm_weights, xs, ys))

    return mean_lat, mean_lon, ((cov_xx, cov_xy), (cov_xy, cov_yy))


def _weighted_circular_mean_deg(angles_deg: Sequence[float], weights: Sequence[float]) -> float:
    sin_sum = sum(w * math.sin(math.radians(a)) for a, w in zip(angles_deg, weights))
    cos_sum = sum(w * math.cos(math.radians(a)) for a, w in zip(angles_deg, weights))
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return _normalize_longitude(sum(a * w for a, w in zip(angles_deg, weights)))
    return _normalize_longitude(math.degrees(math.atan2(sin_sum, cos_sum)))


def _wrapped_longitude_diff(lon: float, reference_lon: float) -> float:
    return ((lon - reference_lon + 540.0) % 360.0) - 180.0


def _normalize_longitude(lon: float) -> float:
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and lon > 0.0:
        return 180.0
    return wrapped


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

