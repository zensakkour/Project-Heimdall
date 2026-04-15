from __future__ import annotations

from src.core.geo.retrieval_provider import (
    _apply_consensus_refinement,
    _apply_locality_rerank,
    _select_diverse_geo_candidates,
)
from src.core.logic.types import GeoCandidate


def _candidates() -> list[GeoCandidate]:
    return [
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.92, match_id="near_a"),
        GeoCandidate(latitude=48.8567, longitude=2.3523, retrieval_score=0.91, match_id="near_b"),
        GeoCandidate(latitude=35.6764, longitude=139.6500, retrieval_score=0.84, match_id="far"),
    ]


def test_diverse_selection_prefers_geographic_spread() -> None:
    selected = _select_diverse_geo_candidates(
        _candidates(),
        top_k=2,
        radius_km=5.0,
        diversity_lambda=0.5,
        min_keep=1,
    )
    ids = [item.match_id for item in selected]
    assert "near_a" in ids
    assert "far" in ids


def test_diverse_selection_noop_when_disabled() -> None:
    selected = _select_diverse_geo_candidates(
        _candidates(),
        top_k=2,
        radius_km=0.0,
        diversity_lambda=1.0,
        min_keep=1,
    )
    assert [item.match_id for item in selected] == ["near_a", "near_b"]


def test_locality_rerank_can_downrank_isolated_outlier() -> None:
    ranked = [
        GeoCandidate(latitude=34.0522, longitude=-118.2437, retrieval_score=0.95, match_id="outlier"),
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.90, match_id="cluster_a"),
        GeoCandidate(latitude=48.8572, longitude=2.3530, retrieval_score=0.89, match_id="cluster_b"),
    ]
    rescored = _apply_locality_rerank(ranked, radius_km=40.0, weight=1.6)
    assert rescored[0].match_id in {"cluster_a", "cluster_b"}


def test_locality_rerank_noop_when_disabled() -> None:
    ranked = [
        GeoCandidate(latitude=10.0, longitude=10.0, retrieval_score=0.75, match_id="a"),
        GeoCandidate(latitude=10.1, longitude=10.2, retrieval_score=0.74, match_id="b"),
        GeoCandidate(latitude=11.0, longitude=11.0, retrieval_score=0.73, match_id="c"),
    ]
    rescored = _apply_locality_rerank(ranked, radius_km=0.0, weight=1.0)
    assert [item.match_id for item in rescored] == ["a", "b", "c"]


def test_consensus_refinement_can_promote_local_cluster_center() -> None:
    ranked = [
        GeoCandidate(latitude=34.0522, longitude=-118.2437, retrieval_score=0.95, match_id="outlier"),
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.90, match_id="cluster_a"),
        GeoCandidate(latitude=48.8571, longitude=2.3528, retrieval_score=0.89, match_id="cluster_b"),
        GeoCandidate(latitude=48.8574, longitude=2.3531, retrieval_score=0.88, match_id="cluster_c"),
    ]
    refined = _apply_consensus_refinement(
        ranked,
        top_n=4,
        radius_km=5.0,
        score_power=1.0,
    )
    assert refined[0].match_id == "retrieval:consensus"
    assert abs(refined[0].latitude - 48.8570) < 0.01
    assert abs(refined[0].longitude - 2.3527) < 0.01


def test_consensus_refinement_noop_when_disabled() -> None:
    ranked = [
        GeoCandidate(latitude=10.0, longitude=10.0, retrieval_score=0.90, match_id="a"),
        GeoCandidate(latitude=10.1, longitude=10.1, retrieval_score=0.89, match_id="b"),
        GeoCandidate(latitude=10.2, longitude=10.2, retrieval_score=0.88, match_id="c"),
    ]
    refined = _apply_consensus_refinement(ranked, top_n=0, radius_km=3.0, score_power=1.0)
    assert [item.match_id for item in refined] == ["a", "b", "c"]
