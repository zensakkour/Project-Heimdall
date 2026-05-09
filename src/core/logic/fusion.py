"""
Probabilistic fusion of geo candidates and verification signals.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

from .astro import sun_position
from .candidate_rerank import candidate_rerank_likelihoods, load_candidate_rerank_model
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
    capture_time: Optional[datetime] = None,
) -> Optional[FusionResult]:
    if not candidates:
        return None
    cfg = config or FusionConfig()

    log_weights: List[float] = []
    evidences: List[Evidence] = []

    capture_time = capture_time or extract_capture_time(image_path)
    observed_shadow = _mean_shadow_azimuth(detections)
    retrieval_scores = _normalized_retrieval_scores(candidates, cfg.retrieval_score_norm)
    spatial_likes = _spatial_consensus_likelihoods(
        candidates,
        retrieval_scores,
        sigma_km=cfg.spatial_sigma_km,
    )
    cross_source_likes = _cross_source_agreement_likelihoods(
        candidates,
        retrieval_scores,
        sigma_km=cfg.cross_source_sigma_km,
    )
    candidate_rerank_likes = _candidate_reranker_likelihoods(candidates, cfg)

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
        source_prior = _source_prior_for_candidate(cand, cfg)
        logp += math.log(max(source_prior, 1e-12))
        likelihoods["source_prior"] = source_prior
        if cfg.use_spatial_consensus:
            spatial_like = spatial_likes[idx]
            logp += max(0.0, max(1.2, cfg.spatial_consensus_weight)) * math.log(max(spatial_like, 1e-12))
            likelihoods["spatial"] = spatial_like
        if cfg.use_cross_source_agreement:
            cross_like = cross_source_likes[idx]
            logp += max(0.0, cfg.cross_source_weight) * math.log(max(cross_like, 1e-12))
            likelihoods["cross_source"] = cross_like
        if candidate_rerank_likes:
            rerank_like = candidate_rerank_likes[idx]
            logp += max(0.0, cfg.candidate_reranker_weight) * math.log(max(rerank_like, 1e-12))
            likelihoods["candidate_reranker"] = rerank_like
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
    plausibility_likes = [1.0 for _ in candidates]
    if cfg.use_plausibility_rerank:
        plausibility_likes = _plausibility_rerank_likelihoods(
            candidates,
            weights,
            radius_km=cfg.plausibility_radius_km,
        )
        weights = _apply_weight_likelihood(weights, plausibility_likes, cfg.plausibility_weight)
    outlier_likes = [1.0 for _ in candidates]
    if cfg.use_adaptive_outlier_guard:
        outlier_likes = _adaptive_outlier_likelihoods(
            candidates,
            weights,
            min_scale_km=cfg.outlier_guard_min_scale_km,
            mad_scale=cfg.outlier_guard_mad_scale,
        )
        weights = _apply_weight_likelihood(weights, outlier_likes, cfg.outlier_guard_strength)

    top_idx = max(range(len(weights)), key=lambda idx: weights[idx]) if weights else 0
    top1_cross_source_support = (
        max(0.0, min(1.0, float(cross_source_likes[top_idx])))
        if cross_source_likes and top_idx < len(cross_source_likes)
        else 1.0
    )
    has_multi_source = len({_candidate_source(cand) for cand in candidates}) >= 2

    norm_entropy, eff_count, top1, calibrated_top1, top2_margin, conf_tier, ambiguous = _fusion_confidence_metrics(
        weights,
        cfg,
    )

    fused: List[FusionCandidate] = []
    for idx, (cand, weight, evidence) in enumerate(zip(candidates, weights, evidences)):
        likelihoods = dict(evidence.likelihoods)
        if cfg.use_plausibility_rerank:
            likelihoods["plausibility"] = plausibility_likes[idx]
        if cfg.use_adaptive_outlier_guard:
            likelihoods["outlier_guard"] = outlier_likes[idx]
        fused.append(
            FusionCandidate(
                candidate=cand,
                posterior_weight=weight,
                evidence=Evidence(
                    retrieval_score=evidence.retrieval_score,
                    shadow_residual_deg=evidence.shadow_residual_deg,
                    terrain_residual=evidence.terrain_residual,
                    likelihoods=likelihoods,
                    posterior_weight=weight,
                    explanation=evidence.explanation,
                ),
            )
        )

    fused.sort(key=lambda item: item.posterior_weight, reverse=True)
    limit = cfg.top_k if cfg.top_k > 0 else len(fused)
    fused = fused[:limit]

    stats_candidates = _select_stats_candidates(fused, cfg, ambiguous=ambiguous)
    mean_lat, mean_lon, cov = _weighted_mean_cov(stats_candidates)
    ellipse = _covariance_to_ellipse(cov)
    uncertainty_radius = max(ellipse.major_axis_m, ellipse.minor_axis_m)
    conf_tier = _apply_cross_source_tier_cap(
        conf_tier,
        top1_cross_source_support=top1_cross_source_support,
        cfg=cfg,
        enabled=cfg.use_cross_source_agreement and has_multi_source,
    )
    conf_tier = _apply_uncertainty_tier_cap(conf_tier, uncertainty_radius, cfg)
    ambiguous = conf_tier == "low" or eff_count >= max(3.0, 0.65 * max(len(weights), 1))

    return FusionResult(
        candidates=fused,
        mean_latitude=mean_lat,
        mean_longitude=mean_lon,
        covariance_m=cov,
        ellipse=ellipse,
        uncertainty_radius_m=uncertainty_radius,
        normalized_entropy=norm_entropy,
        effective_candidate_count=eff_count,
        top1_posterior=top1,
        calibrated_top1_posterior=calibrated_top1,
        top1_cross_source_support=top1_cross_source_support,
        top2_margin=top2_margin,
        confidence_tier=conf_tier,
        ambiguous=ambiguous,
        credible_set_size=len(stats_candidates),
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
    if match_id == "geoclip" or match_id.startswith("geoclip:"):
        return "geoclip"
    retrieval_source = _parse_retrieval_source(match_id)
    if retrieval_source:
        return f"retrieval:{retrieval_source}"
    return "retrieval"


def _source_prior_for_candidate(cand: GeoCandidate, cfg: FusionConfig) -> float:
    src = _candidate_source(cand)
    if src == "exif":
        return max(1e-3, float(cfg.source_prior_exif))
    if src == "geoclip":
        return max(1e-3, float(cfg.source_prior_geoclip))
    overrides = cfg.source_prior_retrieval_by_source or {}
    if src.startswith("retrieval:"):
        if src in overrides:
            return max(1e-3, float(overrides[src]))
        source_name = src.split(":", 1)[1]
        if source_name in overrides:
            return max(1e-3, float(overrides[source_name]))
    return max(1e-3, float(cfg.source_prior_retrieval))


def _parse_retrieval_source(match_id: str) -> Optional[str]:
    if not match_id.startswith("retrieval:"):
        return None
    parts = match_id.split(":")
    # Keep compatibility with legacy IDs like "retrieval:tile-123".
    if len(parts) < 3:
        return None
    source = parts[1].strip()
    return source or None


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


def _plausibility_rerank_likelihoods(
    candidates: Sequence[GeoCandidate],
    weights: Sequence[float],
    radius_km: float,
) -> List[float]:
    if not candidates:
        return []
    if len(candidates) == 1:
        return [1.0]
    radius = max(1.0, float(radius_km))
    safe_weights = [max(0.0, float(w)) for w in weights]
    total = sum(safe_weights)
    if total <= 0.0:
        safe_weights = [1.0 / len(candidates) for _ in candidates]
    else:
        safe_weights = [w / total for w in safe_weights]

    raw = []
    for idx_i, cand_i in enumerate(candidates):
        support = 0.0
        for idx_j, (cand_j, wj) in enumerate(zip(candidates, safe_weights)):
            if idx_i == idx_j:
                continue
            if _haversine_km(cand_i, cand_j) <= radius:
                support += wj
        raw.append(max(1e-6, support))

    peak = max(raw) if raw else 0.0
    if peak <= 1e-6:
        return [1.0 for _ in candidates]
    return [max(1e-3, min(1.0, value / peak)) for value in raw]


def _apply_weight_likelihood(weights: Sequence[float], likes: Sequence[float], strength: float) -> List[float]:
    if not weights:
        return []
    alpha = max(0.0, float(strength))
    if alpha <= 0.0:
        total = sum(max(0.0, float(w)) for w in weights)
        if total <= 0.0:
            return [1.0 / len(weights) for _ in weights]
        return [max(0.0, float(w)) / total for w in weights]
    logits = []
    for w, like in zip(weights, likes):
        base = max(1e-12, float(w))
        lk = max(1e-12, float(like))
        logits.append(math.log(base) + alpha * math.log(lk))
    return _softmax(logits)


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    pairs = sorted(
        [(float(v), max(0.0, float(w))) for v, w in zip(values, weights)],
        key=lambda item: item[0],
    )
    if not pairs:
        return 0.0
    total = sum(item[1] for item in pairs)
    if total <= 0.0:
        return float(pairs[len(pairs) // 2][0])
    cutoff = 0.5 * total
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= cutoff:
            return float(value)
    return float(pairs[-1][0])


def _adaptive_outlier_likelihoods(
    candidates: Sequence[GeoCandidate],
    _weights: Sequence[float],
    *,
    min_scale_km: float,
    mad_scale: float,
) -> List[float]:
    if not candidates:
        return []
    if len(candidates) < 3:
        return [1.0 for _ in candidates]
    # Keep geometry robust to score spikes by using uniform support for center/scale.
    uniform = [1.0 for _ in candidates]

    centrality = []
    for idx_i, cand_i in enumerate(candidates):
        dist_sum = 0.0
        for idx_j, cand_j in enumerate(candidates):
            if idx_i == idx_j:
                continue
            dist_sum += _haversine_km(cand_i, cand_j)
        centrality.append(dist_sum)
    center_idx = min(range(len(candidates)), key=lambda idx: centrality[idx])
    center = candidates[center_idx]
    dists = [_haversine_km(center, cand) for cand in candidates]

    median_dist = _weighted_median(dists, uniform)
    abs_dev = [abs(dist - median_dist) for dist in dists]
    mad = _weighted_median(abs_dev, uniform)
    scale = max(
        max(1.0, float(min_scale_km)),
        float(median_dist) + max(0.0, float(mad_scale)) * max(float(mad), 1e-6),
    )
    raw = [math.exp(-0.5 * (dist / scale) ** 2) for dist in dists]
    peak = max(raw) if raw else 0.0
    if peak <= 0.0:
        return [1.0 for _ in candidates]
    return [max(1e-3, min(1.0, val / peak)) for val in raw]


def _apply_cross_source_tier_cap(
    tier: str,
    top1_cross_source_support: float,
    cfg: FusionConfig,
    enabled: bool,
) -> str:
    if not enabled:
        return tier
    out = tier
    support = max(0.0, min(1.0, float(top1_cross_source_support)))
    high_floor = cfg.confidence_high_min_cross_source_support
    medium_floor = cfg.confidence_medium_min_cross_source_support
    if out == "high" and high_floor is not None and support < float(high_floor):
        out = "medium"
    if out == "medium" and medium_floor is not None and support < float(medium_floor):
        out = "low"
    return out


def _apply_uncertainty_tier_cap(tier: str, uncertainty_radius_m: float, cfg: FusionConfig) -> str:
    out = tier
    high_cap = cfg.confidence_high_max_uncertainty_m
    med_cap = cfg.confidence_medium_max_uncertainty_m
    if out == "high" and high_cap is not None and uncertainty_radius_m > float(high_cap):
        out = "medium"
    if out == "medium" and med_cap is not None and uncertainty_radius_m > float(med_cap):
        out = "low"
    return out


def _cross_source_agreement_likelihoods(
    candidates: Sequence[GeoCandidate],
    retrieval_scores: Sequence[float],
    sigma_km: float,
) -> List[float]:
    if not candidates:
        return []
    if len(candidates) == 1:
        return [1.0]

    sources = [_candidate_source(cand) for cand in candidates]
    if len(set(sources)) < 2:
        return [1.0 for _ in candidates]

    sigma = max(0.1, float(sigma_km))
    priors = [max(1e-3, _to_unit_interval(score)) for score in retrieval_scores]
    total = sum(priors)
    if total <= 0.0:
        priors = [1.0 / len(candidates) for _ in candidates]
    else:
        priors = [p / total for p in priors]

    raw: List[float] = []
    for idx_i, cand_i in enumerate(candidates):
        src_i = sources[idx_i]
        weighted_sum = 0.0
        other_prior = 0.0
        for idx_j, (cand_j, prior_j) in enumerate(zip(candidates, priors)):
            if idx_i == idx_j or sources[idx_j] == src_i:
                continue
            other_prior += prior_j
            dist = _haversine_km(cand_i, cand_j)
            weighted_sum += prior_j * math.exp(-0.5 * (dist / sigma) ** 2)

        if other_prior <= 0.0:
            raw.append(1.0)
            continue
        raw.append(weighted_sum / other_prior)

    peak = max(raw) if raw else 0.0
    if peak <= 0.0:
        return [1.0 for _ in candidates]
    return [max(1e-3, min(1.0, value / peak)) for value in raw]


def _candidate_reranker_likelihoods(candidates: Sequence[GeoCandidate], cfg: FusionConfig) -> List[float]:
    path = cfg.candidate_reranker_path
    weight = max(0.0, float(cfg.candidate_reranker_weight))
    if not path or weight <= 0.0:
        return []
    try:
        model = load_candidate_rerank_model(path)
    except Exception:
        return []
    likes = candidate_rerank_likelihoods(candidates, model)
    if len(likes) != len(candidates):
        return []
    return likes


def _calibrate_probability(prob: float, logit_scale: float, logit_bias: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(prob)))
    logit = math.log(p / (1.0 - p))
    calibrated = 1.0 / (1.0 + math.exp(-(logit * float(logit_scale) + float(logit_bias))))
    return max(0.0, min(1.0, calibrated))


def _fusion_confidence_metrics(weights: Sequence[float], cfg: FusionConfig) -> Tuple[float, float, float, float, float, str, bool]:
    if not weights:
        return 1.0, 0.0, 0.0, 0.0, 0.0, "low", True
    n = len(weights)
    safe = [max(0.0, float(w)) for w in weights]
    total = sum(safe)
    if total <= 0.0:
        safe = [1.0 / n for _ in range(n)]
    else:
        safe = [w / total for w in safe]

    entropy = -sum(w * math.log(max(w, 1e-12)) for w in safe)
    max_entropy = math.log(max(n, 2))
    norm_entropy = max(0.0, min(1.0, entropy / max_entropy))
    eff_count = 1.0 / max(sum(w * w for w in safe), 1e-12)

    ranked = sorted(safe, reverse=True)
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else 0.0
    margin = top1 - top2

    calibrated_top1 = _calibrate_probability(
        top1,
        cfg.confidence_calibration_logit_scale,
        cfg.confidence_calibration_logit_bias,
    )

    if calibrated_top1 >= cfg.confidence_high_threshold and margin >= 0.20 and norm_entropy <= 0.55:
        tier = "high"
    elif calibrated_top1 >= cfg.confidence_medium_threshold and margin >= 0.08 and norm_entropy <= 0.80:
        tier = "medium"
    else:
        tier = "low"

    ambiguous = tier == "low" or eff_count >= max(3.0, 0.65 * n)
    return norm_entropy, eff_count, top1, calibrated_top1, margin, tier, ambiguous


def _credible_set_for_stats(
    fused: Sequence[FusionCandidate],
    credible_mass: float,
    min_candidates: int,
) -> List[FusionCandidate]:
    if not fused:
        return []
    target = max(0.50, min(0.999, float(credible_mass)))
    required = max(1, int(min_candidates))
    selected: List[FusionCandidate] = []
    cumulative = 0.0
    for item in fused:
        selected.append(item)
        cumulative += max(0.0, item.posterior_weight)
        if cumulative >= target and len(selected) >= required:
            break
    while len(selected) < min(required, len(fused)):
        selected.append(fused[len(selected)])
    return selected


def _select_stats_candidates(
    fused: Sequence[FusionCandidate],
    cfg: FusionConfig,
    ambiguous: bool,
) -> List[FusionCandidate]:
    base = _credible_set_for_stats(
        fused,
        credible_mass=cfg.credible_mass,
        min_candidates=cfg.min_credible_candidates,
    )
    if not base or not ambiguous or not cfg.use_top_cluster_for_stats:
        return base

    radius_km = max(1.0, float(cfg.credible_cluster_radius_km))
    min_cluster_weight = max(0.0, min(1.0, float(cfg.min_credible_cluster_weight)))
    required = min(max(1, int(cfg.min_credible_candidates)), len(base))
    best_cluster: Optional[List[FusionCandidate]] = None
    best_score: Optional[Tuple[float, float, float]] = None
    for anchor in base:
        cluster = [
            item
            for item in base
            if _haversine_km(anchor.candidate, item.candidate) <= radius_km
        ]
        if not cluster:
            continue
        cluster_weight = sum(max(0.0, item.posterior_weight) for item in cluster)
        if len(cluster) < required or cluster_weight < min_cluster_weight:
            continue
        # Prefer clusters with stronger total posterior support, then larger size,
        # then stronger anchor posterior.
        score = (cluster_weight, float(len(cluster)), max(0.0, anchor.posterior_weight))
        if best_score is None or score > best_score:
            best_score = score
            best_cluster = cluster

    if best_cluster is not None:
        return best_cluster
    return base


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
