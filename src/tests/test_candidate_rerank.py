from __future__ import annotations

import json
from pathlib import Path

from src.core.logic.candidate_rerank import (
    CandidateRerankModel,
    candidate_feature_matrix,
    candidate_rerank_likelihoods,
    model_to_json,
)
from src.core.logic.config import FusionConfig
from src.core.logic.fusion import fuse_candidates
from src.core.logic.types import GeoCandidate
from src.tools.train_geo_candidate_reranker import _fit_listwise_softmax


def test_candidate_features_include_spatial_support() -> None:
    candidates = [
        GeoCandidate(48.8566, 2.3522, 0.9, match_id="retrieval:a"),
        GeoCandidate(48.8570, 2.3526, 0.7, match_id="retrieval:b"),
        GeoCandidate(35.0, 139.0, 0.8, match_id="geoclip"),
    ]

    rows = candidate_feature_matrix(candidates, ["rank_frac", "support_1km", "source_geoclip"])

    assert rows[0][0] == 0.0
    assert rows[0][1] > rows[2][1]
    assert rows[2][2] == 1.0


def test_candidate_reranker_can_promote_supported_lower_rank_candidate() -> None:
    candidates = [
        GeoCandidate(35.0, 139.0, 0.93, match_id="retrieval:isolated"),
        GeoCandidate(48.8566, 2.3522, 0.72, match_id="retrieval:near_a"),
        GeoCandidate(48.8570, 2.3528, 0.71, match_id="retrieval:near_b"),
    ]
    model = CandidateRerankModel(
        feature_names=("support_1km",),
        weights=(4.0,),
        intercept=0.0,
    )

    likes = candidate_rerank_likelihoods(candidates, model)

    assert likes[1] > likes[0]
    assert likes[2] > likes[0]


def test_candidate_reranker_exp_activation_normalizes_peak() -> None:
    candidates = [
        GeoCandidate(48.8566, 2.3522, 0.9, match_id="retrieval:a"),
        GeoCandidate(48.8570, 2.3528, 0.7, match_id="retrieval:b"),
    ]
    model = CandidateRerankModel(
        feature_names=("rank_frac",),
        weights=(-2.0,),
        intercept=0.0,
        activation="exp",
    )

    likes = candidate_rerank_likelihoods(candidates, model)

    assert likes[0] == 1.0
    assert 0.0 < likes[1] < likes[0]


def test_listwise_reranker_learns_group_winner() -> None:
    model = _fit_listwise_softmax(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        [[0.1, 5.0], [0.2, 4.0]],
        target_sigma_km=1.0,
        learning_rate=0.1,
        epochs=80,
        l2=0.0,
        seed=42,
    )
    candidates = [
        GeoCandidate(48.8566, 2.3522, 0.5, match_id="retrieval:good"),
        GeoCandidate(48.8600, 2.3600, 0.5, match_id="retrieval:bad"),
    ]

    likes = candidate_rerank_likelihoods(candidates, model)

    assert model.activation == "exp"
    assert likes[0] > likes[1]


def test_fusion_applies_candidate_reranker_model() -> None:
    candidates = [
        GeoCandidate(35.0, 139.0, 0.93, match_id="retrieval:isolated"),
        GeoCandidate(48.8566, 2.3522, 0.72, match_id="retrieval:near_a"),
        GeoCandidate(48.8570, 2.3528, 0.71, match_id="retrieval:near_b"),
    ]
    model = CandidateRerankModel(
        feature_names=("support_1km",),
        weights=(5.0,),
        intercept=0.0,
    )
    model_path = Path("runs") / "test_candidate_reranker_model.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model_path.write_text(json.dumps(model_to_json(model)), encoding="utf-8")

        result = fuse_candidates(
            "image.jpg",
            candidates,
            detections=[],
            config=FusionConfig(
                retrieval_temperature=0.5,
                use_cross_source_agreement=False,
                use_spatial_consensus=False,
                use_plausibility_rerank=False,
                candidate_reranker_path=str(model_path),
                candidate_reranker_weight=40.0,
                top_k=3,
            ),
        )

        assert result is not None
        assert result.candidates[0].candidate.match_id in {"retrieval:near_a", "retrieval:near_b"}
        assert "candidate_reranker" in result.candidates[0].evidence.likelihoods
    finally:
        model_path.unlink(missing_ok=True)
