from __future__ import annotations

import pytest

from src.tools.mine_hard_negative_triplets import EvalFailure, GeoRecord, merge_failures, mine_triplets, scene_key


def test_mine_triplets_from_eval_failures() -> None:
    records = [
        GeoRecord(path="q.jpg", latitude=48.0000, longitude=2.0000),
        GeoRecord(path="p1.jpg", latitude=48.0018, longitude=2.0000),   # ~0.2 km
        GeoRecord(path="n1.jpg", latitude=48.0180, longitude=2.0000),   # ~2.0 km
        GeoRecord(path="n2.jpg", latitude=48.0180, longitude=2.0020),   # ~2.1 km
        GeoRecord(path="far.jpg", latitude=48.0900, longitude=2.0900),  # far
    ]
    failures = [
        EvalFailure(
            query_path="q.jpg",
            gt_latitude=48.0000,
            gt_longitude=2.0000,
            pred_latitude=48.0180,
            pred_longitude=2.0000,
            distance_km=2.0,
        )
    ]

    out = mine_triplets(
        records,
        failures,
        min_error_km=1.0,
        positive_radius_km=0.4,
        negative_pred_radius_km=0.6,
        negative_min_gt_distance_km=1.5,
        negative_max_gt_distance_km=30.0,
        max_positives=2,
        max_negatives=2,
    )
    assert out
    row = out[0]
    assert row["query_path"] == "q.jpg"
    assert row["positives"][0]["path"] == "p1.jpg"
    neg_paths = {item["path"] for item in row["hard_negatives"]}
    assert "n1.jpg" in neg_paths


def test_mine_triplets_emits_difficulty_metadata() -> None:
    records = [
        GeoRecord(path="q.jpg", latitude=48.0000, longitude=2.0000),
        GeoRecord(path="p1.jpg", latitude=48.0018, longitude=2.0000),
        GeoRecord(path="n1.jpg", latitude=48.0180, longitude=2.0000),
        GeoRecord(path="n2.jpg", latitude=48.0180, longitude=2.0020),
    ]
    failures = [
        EvalFailure(
            query_path="q.jpg",
            gt_latitude=48.0000,
            gt_longitude=2.0000,
            pred_latitude=48.0180,
            pred_longitude=2.0000,
            distance_km=4.0,
        )
    ]

    out = mine_triplets(
        records,
        failures,
        min_error_km=1.0,
        positive_radius_km=0.4,
        negative_pred_radius_km=0.6,
        negative_min_gt_distance_km=1.5,
        negative_max_gt_distance_km=30.0,
        max_positives=2,
        max_negatives=2,
        difficulty_mode="error_km_predmix",
        difficulty_reference_km=2.0,
        difficulty_max_weight=3.0,
    )

    assert out
    row = out[0]
    assert row["difficulty_mode"] == "error_km_predmix"
    assert row["triplet_weight"] > 1.0
    assert row["hard_negative_source_counts"]["pred_neighborhood"] >= 1
    assert row["closest_positive_gt_km"] == pytest.approx(row["positives"][0]["distance_to_gt_km"])
    assert row["closest_negative_gt_km"] == pytest.approx(2.0015086796020573, rel=1e-6)
    assert row["closest_negative_pred_km"] == pytest.approx(0.0, abs=1e-9)


def test_mine_triplets_skips_low_error_failures() -> None:
    records = [
        GeoRecord(path="q.jpg", latitude=48.0000, longitude=2.0000),
        GeoRecord(path="p1.jpg", latitude=48.0018, longitude=2.0000),
        GeoRecord(path="n1.jpg", latitude=48.0180, longitude=2.0000),
    ]
    failures = [
        EvalFailure(
            query_path="q.jpg",
            gt_latitude=48.0000,
            gt_longitude=2.0000,
            pred_latitude=48.0180,
            pred_longitude=2.0000,
            distance_km=0.3,
        )
    ]

    out = mine_triplets(
        records,
        failures,
        min_error_km=1.0,
        positive_radius_km=0.4,
        negative_pred_radius_km=0.6,
        negative_min_gt_distance_km=1.5,
        negative_max_gt_distance_km=30.0,
        max_positives=2,
        max_negatives=2,
    )
    assert out == []


def test_scene_key_normalizes_modal_prefixes() -> None:
    assert scene_key("RGB-PanSharpen_AOI_3_Paris_img1_r0_c0.jpg") == "AOI_3_Paris_img1_r0_c0.jpg"
    assert scene_key("PAN_AOI_3_Paris_img1_r0_c0.jpg") == "AOI_3_Paris_img1_r0_c0.jpg"
    assert scene_key("MUL-PanSharpen_AOI_3_Paris_img1_r0_c0.jpg") == "AOI_3_Paris_img1_r0_c0.jpg"


def test_mine_triplets_with_separate_reference_pool() -> None:
    query_records = [
        GeoRecord(path="q.jpg", latitude=48.0000, longitude=2.0000),
    ]
    reference_records = [
        GeoRecord(path="ref_pos.jpg", latitude=48.0018, longitude=2.0000),  # ~0.2km
        GeoRecord(path="ref_neg.jpg", latitude=48.0180, longitude=2.0000),  # ~2.0km
    ]
    failures = [
        EvalFailure(
            query_path="q.jpg",
            gt_latitude=48.0000,
            gt_longitude=2.0000,
            pred_latitude=48.0180,
            pred_longitude=2.0000,
            distance_km=2.0,
        )
    ]

    out = mine_triplets(
        query_records,
        failures,
        reference_records=reference_records,
        min_error_km=1.0,
        positive_radius_km=0.4,
        negative_pred_radius_km=0.6,
        negative_min_gt_distance_km=1.5,
        negative_max_gt_distance_km=30.0,
        max_positives=2,
        max_negatives=2,
    )
    assert out
    row = out[0]
    assert row["query_path"] == "q.jpg"
    assert row["positives"][0]["path"] == "ref_pos.jpg"
    neg_paths = {item["path"] for item in row["hard_negatives"]}
    assert "ref_neg.jpg" in neg_paths


def test_merge_failures_keeps_hardest_per_query_and_dedupes() -> None:
    failures = [
        EvalFailure(
            query_path="q.jpg",
            gt_latitude=48.0,
            gt_longitude=2.0,
            pred_latitude=48.01,
            pred_longitude=2.01,
            distance_km=5.0,
        ),
        EvalFailure(
            query_path="q.jpg",
            gt_latitude=48.0,
            gt_longitude=2.0,
            pred_latitude=48.01,
            pred_longitude=2.01,
            distance_km=5.0,
        ),
        EvalFailure(
            query_path="q.jpg",
            gt_latitude=48.0,
            gt_longitude=2.0,
            pred_latitude=48.02,
            pred_longitude=2.02,
            distance_km=3.0,
        ),
        EvalFailure(
            query_path="other.jpg",
            gt_latitude=48.1,
            gt_longitude=2.1,
            pred_latitude=48.11,
            pred_longitude=2.11,
            distance_km=4.0,
        ),
    ]

    merged = merge_failures(failures, max_failures_per_query=1)
    assert len(merged) == 2
    assert merged[0].query_path == "other.jpg"
    assert merged[1].query_path == "q.jpg"
    assert merged[1].distance_km == pytest.approx(5.0)


def test_mine_triplets_accepts_exact_coordinate_positive_from_reference_pool() -> None:
    query_records = [
        GeoRecord(path="street.jpg", latitude=48.0000, longitude=2.0000),
    ]
    reference_records = [
        GeoRecord(path="aerial_exact.png", latitude=48.0000, longitude=2.0000),
        GeoRecord(path="aerial_neg.png", latitude=48.0180, longitude=2.0000),
    ]
    failures = [
        EvalFailure(
            query_path="street.jpg",
            gt_latitude=48.0000,
            gt_longitude=2.0000,
            pred_latitude=48.0180,
            pred_longitude=2.0000,
            distance_km=2.0,
        )
    ]

    out = mine_triplets(
        query_records,
        failures,
        reference_records=reference_records,
        min_error_km=1.0,
        positive_radius_km=0.1,
        negative_pred_radius_km=0.6,
        negative_min_gt_distance_km=1.5,
        negative_max_gt_distance_km=30.0,
        max_positives=2,
        max_negatives=2,
    )

    assert out
    row = out[0]
    assert row["positives"][0]["path"] == "aerial_exact.png"
    assert row["positives"][0]["distance_to_gt_km"] == pytest.approx(0.0, abs=1e-9)
