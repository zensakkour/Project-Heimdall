from __future__ import annotations

from pathlib import Path

import pytest

from src.core.logic.types import GeoCandidate
from src.tools.mine_retrieval_hard_triplets import (
    GeoRecord,
    _reference_style_path,
    mine_retrieval_triplets,
    mine_triplets_for_query,
)


class FakeProvider:
    def __init__(self, candidates):
        self._candidates = candidates

    def candidates(self, image_path: str):
        return self._candidates


def test_mine_triplets_uses_retrieved_wrong_candidates_as_negatives() -> None:
    references = [
        GeoRecord(path="chips/positive.jpg", latitude=48.0005, longitude=2.0000),
        GeoRecord(path="chips/far_retrieved.jpg", latitude=48.0180, longitude=2.0000),
        GeoRecord(path="chips/too_close.jpg", latitude=48.0010, longitude=2.0000),
    ]
    candidates = [
        GeoCandidate(
            latitude=48.0010,
            longitude=2.0000,
            retrieval_score=0.99,
            match_id="near",
            image_path="data/spacenet_paris/chips/too_close.jpg",
        ),
        GeoCandidate(
            latitude=48.0180,
            longitude=2.0000,
            retrieval_score=0.98,
            match_id="wrong",
            image_path="data/spacenet_paris/chips/far_retrieved.jpg",
        ),
    ]

    row = mine_triplets_for_query(
        query_path="street/query.jpg",
        gt_latitude=48.0000,
        gt_longitude=2.0000,
        candidates=candidates,
        reference_records=references,
        positive_radius_km=0.2,
        positive_fallback_top_k=2,
        negative_min_gt_distance_km=1.0,
        negative_max_gt_distance_km=5.0,
        max_positives=2,
        max_negatives=3,
    )

    assert row is not None
    assert row["query_path"] == "street/query.jpg"
    assert row["positives"][0]["path"] == "chips/positive.jpg"
    assert [item["path"] for item in row["hard_negatives"]] == ["chips/far_retrieved.jpg"]
    assert row["hard_negatives"][0]["retrieval_rank"] == 2
    assert row["triplet_weight"] > 1.0


def test_mine_retrieval_triplets_resolves_existing_images(tmp_path: Path) -> None:
    images_dir = tmp_path / "street"
    images_dir.mkdir()
    (images_dir / "query.jpg").write_bytes(b"not-a-real-image")
    provider = FakeProvider(
        [
            GeoCandidate(
                latitude=48.0200,
                longitude=2.0000,
                retrieval_score=0.91,
                match_id="hard",
                image_path="chips/hard.jpg",
            )
        ]
    )

    triplets, summary = mine_retrieval_triplets(
        records=[{"path": "query.jpg", "latitude": 48.0, "longitude": 2.0}],
        images_dir=images_dir,
        reference_records=[
            GeoRecord(path="chips/positive.jpg", latitude=48.0005, longitude=2.0000),
            GeoRecord(path="chips/hard.jpg", latitude=48.0200, longitude=2.0000),
        ],
        provider=provider,
        limit=10,
        seed=7,
        positive_radius_km=0.2,
        positive_fallback_top_k=2,
        negative_min_gt_distance_km=1.0,
        negative_max_gt_distance_km=5.0,
        max_positives=2,
        max_negatives=2,
    )

    assert len(triplets) == 1
    assert summary["triplets_written"] == 1
    assert summary["missing_files"] == 0
    assert summary["candidate_count_mean"] == pytest.approx(1.0)


def test_reference_style_path_matches_spacenet_chip_paths() -> None:
    assert _reference_style_path("data/spacenet_paris/chips/AOI_3_Paris.jpg") == "chips/AOI_3_Paris.jpg"
    assert _reference_style_path("chips/AOI_3_Paris.jpg") == "chips/AOI_3_Paris.jpg"
