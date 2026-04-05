"""
Bayesian filtering for temporal fusion.
"""
from __future__ import annotations

import math
from typing import List, Sequence

from .types import FusionResult, UncertaintyEllipse


def update_posterior(prior: FusionResult, current: FusionResult) -> FusionResult:
    if not current.candidates:
        return current

    sigma_m = _temporal_sigma_m(prior, current)
    curr_weights = _normalize_weights([item.posterior_weight for item in current.candidates])
    prior_weights = _normalize_weights([item.posterior_weight for item in prior.candidates])

    log_scores = []
    support_values = []
    for idx, curr in enumerate(current.candidates):
        support = 0.0
        for prior_item, prior_w in zip(prior.candidates, prior_weights):
            dist = _haversine_m(
                curr.candidate.latitude,
                curr.candidate.longitude,
                prior_item.candidate.latitude,
                prior_item.candidate.longitude,
            )
            support += prior_w * math.exp(-0.5 * (dist / sigma_m) ** 2)
        if not prior.candidates:
            support = 1.0
        support_values.append(support)
        base = max(1e-12, curr_weights[idx])
        log_scores.append(math.log(base) + math.log(max(1e-12, support)))

    fused_weights = _softmax(log_scores)
    fused_candidates = []
    for item, weight in zip(current.candidates, fused_weights):
        fused_candidates.append(
            item.__class__(
                candidate=item.candidate,
                posterior_weight=weight,
                evidence=item.evidence.__class__(
                    retrieval_score=item.evidence.retrieval_score,
                    shadow_residual_deg=item.evidence.shadow_residual_deg,
                    terrain_residual=item.evidence.terrain_residual,
                    likelihoods=dict(item.evidence.likelihoods),
                    posterior_weight=weight,
                    explanation=item.evidence.explanation,
                ),
            )
        )
    fused_candidates.sort(key=lambda item: item.posterior_weight, reverse=True)

    mean_lat, mean_lon = _weighted_mean_latlon(fused_candidates)
    cov = _weighted_covariance_m(fused_candidates, mean_lat, mean_lon)
    spread_radius = _covariance_radius_m(cov)
    support_strength = max(support_values) if support_values else 0.0
    shrink = max(0.55, 1.0 - 0.35 * max(0.0, min(1.0, support_strength)))
    uncertainty_radius = max(spread_radius, current.uncertainty_radius_m * shrink)

    entropy, eff_count, top1, margin = _posterior_diagnostics(fused_weights)
    confidence_tier = _tier_from_posterior(top1, margin, entropy)
    ambiguous = confidence_tier == "low" or eff_count >= max(3.0, 0.65 * max(len(fused_weights), 1))
    credible_size = _credible_size(fused_weights, mass=0.90)

    return FusionResult(
        candidates=fused_candidates,
        mean_latitude=mean_lat,
        mean_longitude=mean_lon,
        covariance_m=cov,
        ellipse=UncertaintyEllipse(
            major_axis_m=uncertainty_radius,
            minor_axis_m=uncertainty_radius,
            orientation_deg=0.0,
        ),
        uncertainty_radius_m=uncertainty_radius,
        normalized_entropy=entropy,
        effective_candidate_count=eff_count,
        top1_posterior=top1,
        calibrated_top1_posterior=top1,
        top1_cross_source_support=current.top1_cross_source_support,
        top2_margin=margin,
        confidence_tier=confidence_tier,
        ambiguous=ambiguous,
        credible_set_size=credible_size,
    )


def _temporal_sigma_m(prior: FusionResult, current: FusionResult) -> float:
    unc = max(0.0, float(prior.uncertainty_radius_m)) + max(0.0, float(current.uncertainty_radius_m))
    adaptive = 0.5 * unc
    return max(500.0, min(25_000.0, adaptive))


def _normalize_weights(values: Sequence[float]) -> List[float]:
    safe = [max(0.0, float(v)) for v in values]
    total = sum(safe)
    if total <= 0.0:
        if not safe:
            return []
        return [1.0 / len(safe) for _ in safe]
    return [v / total for v in safe]


def _softmax(logits: Sequence[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits) for _ in logits]
    return [v / total for v in exps]


def _weighted_mean_latlon(candidates: Sequence) -> tuple[float, float]:
    if not candidates:
        return 0.0, 0.0
    weights = _normalize_weights([item.posterior_weight for item in candidates])
    mean_lat = sum(item.candidate.latitude * w for item, w in zip(candidates, weights))
    sin_sum = sum(w * math.sin(math.radians(item.candidate.longitude)) for item, w in zip(candidates, weights))
    cos_sum = sum(w * math.cos(math.radians(item.candidate.longitude)) for item, w in zip(candidates, weights))
    mean_lon = math.degrees(math.atan2(sin_sum, cos_sum)) if abs(sin_sum) > 1e-12 or abs(cos_sum) > 1e-12 else 0.0
    return mean_lat, ((mean_lon + 180.0) % 360.0) - 180.0


def _weighted_covariance_m(candidates: Sequence, mean_lat: float, mean_lon: float) -> tuple[tuple[float, float], tuple[float, float]]:
    if not candidates:
        return ((0.0, 0.0), (0.0, 0.0))
    weights = _normalize_weights([item.posterior_weight for item in candidates])
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = max(1e-6, abs(111_320.0 * math.cos(math.radians(mean_lat))))
    xs = []
    ys = []
    for item in candidates:
        dlon = ((item.candidate.longitude - mean_lon + 540.0) % 360.0) - 180.0
        xs.append(dlon * meters_per_deg_lon)
        ys.append((item.candidate.latitude - mean_lat) * meters_per_deg_lat)
    cov_xx = sum(w * (x ** 2) for w, x in zip(weights, xs))
    cov_yy = sum(w * (y ** 2) for w, y in zip(weights, ys))
    cov_xy = sum(w * x * y for w, x, y in zip(weights, xs, ys))
    return ((cov_xx, cov_xy), (cov_xy, cov_yy))


def _covariance_radius_m(cov: tuple[tuple[float, float], tuple[float, float]]) -> float:
    (a, b), (_, d) = cov
    trace = max(0.0, a + d)
    det = a * d - b * b
    root = math.sqrt(max(0.0, trace * trace / 4.0 - det))
    eig1 = max(0.0, trace / 2.0 + root)
    eig2 = max(0.0, trace / 2.0 - root)
    return max(math.sqrt(eig1), math.sqrt(eig2))


def _posterior_diagnostics(weights: Sequence[float]) -> tuple[float, float, float, float]:
    safe = _normalize_weights(weights)
    if not safe:
        return 1.0, 0.0, 0.0, 0.0
    n = len(safe)
    entropy = -sum(w * math.log(max(w, 1e-12)) for w in safe)
    max_entropy = math.log(max(n, 2))
    norm_entropy = max(0.0, min(1.0, entropy / max_entropy))
    eff_count = 1.0 / max(sum(w * w for w in safe), 1e-12)
    ranked = sorted(safe, reverse=True)
    top1 = ranked[0]
    top2 = ranked[1] if len(ranked) > 1 else 0.0
    return norm_entropy, eff_count, top1, (top1 - top2)


def _tier_from_posterior(top1: float, margin: float, norm_entropy: float) -> str:
    if top1 >= 0.75 and margin >= 0.20 and norm_entropy <= 0.55:
        return "high"
    if top1 >= 0.50 and margin >= 0.08 and norm_entropy <= 0.80:
        return "medium"
    return "low"


def _credible_size(weights: Sequence[float], mass: float) -> int:
    if not weights:
        return 0
    target = max(0.5, min(0.999, float(mass)))
    cum = 0.0
    count = 0
    for w in sorted(_normalize_weights(weights), reverse=True):
        cum += w
        count += 1
        if cum >= target:
            break
    return count


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    term = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    arc = 2.0 * math.atan2(math.sqrt(term), math.sqrt(max(0.0, 1.0 - term)))
    return radius_m * arc
