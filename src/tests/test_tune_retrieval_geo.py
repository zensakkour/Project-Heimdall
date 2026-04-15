from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.core.logic.types import GeoCandidate
from src.tools.tune_retrieval_geo import (
    RetrievalSample,
    _evaluate_samples,
    _parse_rank_objective,
    _parse_tta_reduce_list,
    _postprocess_candidates,
    _result_sort_key,
    _write_best_to_config,
)


def test_postprocess_candidates_respects_min_score() -> None:
    raw = [
        GeoCandidate(latitude=0.0, longitude=0.0, retrieval_score=0.4, match_id="a"),
        GeoCandidate(latitude=1.0, longitude=1.0, retrieval_score=0.2, match_id="b"),
    ]
    out = _postprocess_candidates(
        raw,
        top_k=5,
        min_score=0.3,
        min_keep_topk=0,
        diversity_radius_km=0.0,
        diversity_lambda=1.0,
        diversity_min_keep=1,
        locality_radius_km=0.0,
        locality_weight=0.0,
        source_balance_beta=0.0,
    )
    assert [cand.match_id for cand in out] == ["a"]


def test_evaluate_samples_uses_top_candidate_distance() -> None:
    samples = [
        RetrievalSample(
            image="x.jpg",
            gt_latitude=48.8566,
            gt_longitude=2.3522,
            candidates=[
                GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.9, match_id="hit"),
                GeoCandidate(latitude=35.6764, longitude=139.6500, retrieval_score=0.8, match_id="miss"),
            ],
        ),
        RetrievalSample(
            image="y.jpg",
            gt_latitude=48.8566,
            gt_longitude=2.3522,
            candidates=[],
        ),
    ]
    metrics = _evaluate_samples(
        samples,
        top_k=5,
        min_score=0.0,
        min_keep_topk=0,
        diversity_radius_km=0.0,
        diversity_lambda=1.0,
        diversity_min_keep=1,
        locality_radius_km=0.0,
        locality_weight=0.0,
        source_balance_beta=0.0,
    )
    assert metrics["evaluated"] == 1
    assert metrics["null_predictions"] == 1
    assert metrics["within_1km_pct"] == 100.0


def test_postprocess_candidates_min_keep_topk_can_override_threshold() -> None:
    raw = [
        GeoCandidate(latitude=0.0, longitude=0.0, retrieval_score=0.2, match_id="a"),
        GeoCandidate(latitude=1.0, longitude=1.0, retrieval_score=0.1, match_id="b"),
    ]
    out = _postprocess_candidates(
        raw,
        top_k=5,
        min_score=0.95,
        min_keep_topk=1,
        diversity_radius_km=0.0,
        diversity_lambda=1.0,
        diversity_min_keep=1,
        locality_radius_km=0.0,
        locality_weight=0.0,
        source_balance_beta=0.0,
    )
    assert len(out) == 1


def test_postprocess_candidates_source_balance_can_promote_other_source() -> None:
    raw = [
        GeoCandidate(latitude=0.0, longitude=0.0, retrieval_score=1.00, match_id="retrieval:a_idx:a1"),
        GeoCandidate(latitude=0.1, longitude=0.1, retrieval_score=0.99, match_id="retrieval:a_idx:a2"),
        GeoCandidate(latitude=1.0, longitude=1.0, retrieval_score=0.98, match_id="retrieval:b_idx:b1"),
    ]
    plain = _postprocess_candidates(
        raw,
        top_k=2,
        min_score=0.0,
        min_keep_topk=0,
        diversity_radius_km=0.0,
        diversity_lambda=1.0,
        diversity_min_keep=1,
        locality_radius_km=0.0,
        locality_weight=0.0,
        source_balance_beta=0.0,
    )
    assert [cand.match_id for cand in plain[:2]] == ["retrieval:a_idx:a1", "retrieval:a_idx:a2"]

    balanced = _postprocess_candidates(
        raw,
        top_k=2,
        min_score=0.0,
        min_keep_topk=0,
        diversity_radius_km=0.0,
        diversity_lambda=1.0,
        diversity_min_keep=1,
        locality_radius_km=0.0,
        locality_weight=0.0,
        source_balance_beta=0.8,
    )
    ids = {cand.match_id for cand in balanced[:2]}
    assert "retrieval:a_idx:a1" in ids
    assert "retrieval:b_idx:b1" in ids


def test_parse_tta_reduce_list_filters_unknown_and_dedupes() -> None:
    out = _parse_tta_reduce_list("max,median,MAX,unknown,rrf,median,mean")
    assert out == ["max", "median", "rrf", "mean"]


def test_write_best_to_config_updates_tta_reduce() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cfg.json"
        path.write_text(json.dumps({"geolocator": {}}), encoding="utf-8")
        _write_best_to_config(
            path,
            {
                "retrieval_top_k": 25,
                "retrieval_min_score": 0.1,
                "retrieval_min_keep_topk": 2,
                "retrieval_diversity_radius_km": 1.0,
                "retrieval_diversity_lambda": 0.9,
                "retrieval_diversity_min_keep": 1,
                "retrieval_locality_radius_km": 25.0,
                "retrieval_locality_weight": 0.8,
                "retrieval_source_balance_beta": 0.35,
                "retrieval_query_tta_reduce": "median",
            },
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["geolocator"]["retrieval_query_tta_reduce"] == "median"


def test_parse_rank_objective_unknown_falls_back_to_balanced() -> None:
    assert _parse_rank_objective("within_1km_pct") == "within_1km_pct"
    assert _parse_rank_objective("not_a_mode") == "balanced"


def test_result_sort_key_within_1km_prefers_higher_recall() -> None:
    rows = [
        {
            "null_predictions": 0,
            "within_1km_pct": 65.0,
            "within_5km_pct": 95.0,
            "median_km": 1.0,
            "mean_km": 2.0,
            "p90_km": 8.0,
        },
        {
            "null_predictions": 0,
            "within_1km_pct": 70.0,
            "within_5km_pct": 90.0,
            "median_km": 1.5,
            "mean_km": 3.0,
            "p90_km": 9.0,
        },
    ]
    ranked = sorted(rows, key=lambda item: _result_sort_key(item, "within_1km_pct"))
    assert ranked[0]["within_1km_pct"] == 70.0
