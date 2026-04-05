"""
Serialization helpers for pipeline outputs.
"""
from __future__ import annotations

from typing import Any

from .types import Assessment


def assessment_to_dict(result: Assessment) -> dict[str, Any]:
    return {
        "detections": [
            {
                "label": d.label,
                "confidence": d.confidence,
                "obb": d.obb,
                "heading_deg": d.heading_deg,
                "shadow_azimuth_deg": d.shadow_azimuth_deg,
                "shadow_length_ratio": d.shadow_length_ratio,
            }
            for d in result.detections
        ],
        "geo": None
        if result.geo is None
        else {
            "latitude": result.geo.latitude,
            "longitude": result.geo.longitude,
            "confidence": result.geo.confidence,
            "landmarks": result.geo.landmarks,
            "uncertainty_m": result.geo.uncertainty_m,
            "uncertainty_radius_m": result.geo.uncertainty_m,
            "confidence_tier": _geo_confidence_tier(result.geo),
        },
        "verification": None
        if result.verification is None
        else {
            "shadow_ok": result.verification.shadow_ok,
            "topo_ok": result.verification.topo_ok,
            "notes": result.verification.notes,
        },
        "score": result.score,
        "candidates": [_candidate_to_dict(c) for c in result.candidates],
        "fusion": None if result.fusion is None else _fusion_to_dict(result.fusion),
    }


def _geo_confidence_tier(geo: "GeoEstimate") -> str:
    score = max(0.0, min(1.0, geo.confidence))
    if geo.uncertainty_m is not None:
        if geo.uncertainty_m > 1000:
            score *= 0.4
        elif geo.uncertainty_m > 200:
            score *= 0.6
        elif geo.uncertainty_m > 50:
            score *= 0.85
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _candidate_to_dict(candidate: "GeoCandidate") -> dict[str, Any]:
    return {
        "latitude": candidate.latitude,
        "longitude": candidate.longitude,
        "retrieval_score": candidate.retrieval_score,
        "match_id": candidate.match_id,
    }


def _fusion_to_dict(fusion: "FusionResult") -> dict[str, Any]:
    return {
        "mean_latitude": fusion.mean_latitude,
        "mean_longitude": fusion.mean_longitude,
        "covariance_m": fusion.covariance_m,
        "ellipse": {
            "major_axis_m": fusion.ellipse.major_axis_m,
            "minor_axis_m": fusion.ellipse.minor_axis_m,
            "orientation_deg": fusion.ellipse.orientation_deg,
        },
        "uncertainty_radius_m": fusion.uncertainty_radius_m,
        "normalized_entropy": fusion.normalized_entropy,
        "effective_candidate_count": fusion.effective_candidate_count,
        "top1_posterior": fusion.top1_posterior,
        "calibrated_top1_posterior": fusion.calibrated_top1_posterior,
        "top1_cross_source_support": fusion.top1_cross_source_support,
        "top2_margin": fusion.top2_margin,
        "confidence_tier": fusion.confidence_tier,
        "ambiguous": fusion.ambiguous,
        "credible_set_size": fusion.credible_set_size,
        "candidates": [
            {
                "candidate": _candidate_to_dict(item.candidate),
                "posterior_weight": item.posterior_weight,
                "evidence": {
                    "retrieval_score": item.evidence.retrieval_score,
                    "shadow_residual_deg": item.evidence.shadow_residual_deg,
                    "terrain_residual": item.evidence.terrain_residual,
                    "likelihoods": item.evidence.likelihoods,
                    "posterior_weight": item.evidence.posterior_weight,
                    "explanation": item.evidence.explanation,
                },
            }
            for item in fusion.candidates
        ],
    }
