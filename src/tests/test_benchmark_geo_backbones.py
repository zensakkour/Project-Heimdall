from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.tools import benchmark_geo_backbones as bench
from src.tools import upgrade_retrieval_backbone as upgrade
from src.tools.train_retrieval_projection import TripletRecord


def test_collect_reference_paths_and_filter_rows() -> None:
    triplets = [
        {
            "query_path": "query/a.jpg",
            "positives": [{"path": "chips/p1.jpg"}, {"path": "chips/p2.jpg"}],
            "hard_negatives": [{"path": "chips/n1.jpg"}, {"path": "chips/p2.jpg"}],
        }
    ]
    rows = [
        {"path": "chips/p2.jpg", "latitude": 1.0, "longitude": 2.0},
        {"path": "chips/n1.jpg", "latitude": 3.0, "longitude": 4.0},
        {"path": "query/a.jpg", "latitude": 5.0, "longitude": 6.0},
    ]

    assert bench._collect_reference_paths(triplets) == ["chips/p1.jpg", "chips/p2.jpg", "chips/n1.jpg"]
    assert bench._filter_rows_by_paths(rows, bench._collect_reference_paths(triplets)) == [
        {"path": "chips/p2.jpg", "latitude": 1.0, "longitude": 2.0},
        {"path": "chips/n1.jpg", "latitude": 3.0, "longitude": 4.0},
    ]


def test_resolve_projection_device_auto_without_torch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TorchStub:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    monkeypatch.setattr(bench.projection_tools, "torch", _TorchStub, raising=False)
    assert bench._resolve_projection_device("auto") == "cpu"
    assert bench._resolve_projection_device("cuda") == "cpu"


def test_fit_projection_for_index_writes_projection_and_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    triplets = [{"query_path": "q.jpg", "positives": [{"path": "p.jpg"}], "hard_negatives": [{"path": "n.jpg"}]}]
    train_rows = [TripletRecord(query_idx=0, positive_indices=(1,), negative_indices=(2,), sample_weight=1.0)]

    monkeypatch.setattr(bench.projection_tools, "_load_triplets", lambda path, max_triplets: triplets)
    monkeypatch.setattr(bench.projection_tools, "_collect_requested_paths", lambda rows: ["q.jpg", "p.jpg", "n.jpg"])
    monkeypatch.setattr(bench.projection_tools, "_load_index_embeddings", lambda path: ({}, {}, 4))
    monkeypatch.setattr(bench.projection_tools, "_embed_missing", lambda **kwargs: (2, 0))
    monkeypatch.setattr(
        bench.projection_tools,
        "_summarize_missing_paths_by_role",
        lambda rows, by_exact, by_name: {
            "query_missing": 0,
            "positive_missing": 0,
            "negative_missing": 0,
            "query_examples": [],
            "positive_examples": [],
            "negative_examples": [],
        },
    )
    monkeypatch.setattr(
        bench.projection_tools,
        "_build_training_records",
        lambda rows, by_exact, by_name, sample_weight_mode, sample_weight_power, sample_weight_max: (
            np.eye(4, dtype=np.float32),
            train_rows,
            {"dropped_missing": 0, "dropped_structure": 0, "embedding_dim": 4, "unique_embeddings": 3},
        ),
    )
    monkeypatch.setattr(
        bench.projection_tools,
        "train_projection",
        lambda **kwargs: (
            np.eye(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            {"history": [], "rows": 1},
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        projection_path = tmp_path / "proj.npz"
        report_path = tmp_path / "proj.report.json"
        summary = bench._fit_projection_for_index(
            model_id="openai/clip-vit-large-patch14",
            raw_index_path=tmp_path / "raw_index.npz",
            triplet_path=tmp_path / "triplets.jsonl",
            projection_images_dir=tmp_path,
            projection_path=projection_path,
            projection_report_path=report_path,
            max_triplets=0,
            output_dim=0,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=1e-4,
            margin=0.1,
            temperature=0.07,
            ce_weight=0.3,
            orth_weight=0.0,
            sample_weight_mode="triplet_weight",
            sample_weight_power=1.0,
            sample_weight_max=3.0,
            seed=42,
            device="cpu",
        )

        assert summary["triplets_loaded"] == 1
        assert summary["triplets_used"] == 1
        assert projection_path.exists()
        assert report_path.exists()

        with np.load(projection_path) as payload:
            assert payload["matrix"].shape == (4, 4)
            assert payload["bias"].shape == (4,)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["model_id"] == "openai/clip-vit-large-patch14"
        assert report["triplets_used"] == 1


def test_fit_projection_for_index_raises_when_training_rows_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    triplets = [{"query_path": "q.jpg", "positives": [{"path": "p.jpg"}], "hard_negatives": [{"path": "n.jpg"}]}]

    monkeypatch.setattr(bench.projection_tools, "_load_triplets", lambda path, max_triplets: triplets)
    monkeypatch.setattr(bench.projection_tools, "_collect_requested_paths", lambda rows: ["q.jpg"])
    monkeypatch.setattr(bench.projection_tools, "_load_index_embeddings", lambda path: ({}, {}, 4))
    monkeypatch.setattr(bench.projection_tools, "_embed_missing", lambda **kwargs: (0, 1))
    monkeypatch.setattr(
        bench.projection_tools,
        "_summarize_missing_paths_by_role",
        lambda rows, by_exact, by_name: {
            "query_missing": 1,
            "positive_missing": 0,
            "negative_missing": 0,
            "query_examples": ["q.jpg"],
            "positive_examples": [],
            "negative_examples": [],
        },
    )
    monkeypatch.setattr(
        bench.projection_tools,
        "_build_training_records",
        lambda rows, by_exact, by_name, sample_weight_mode, sample_weight_power, sample_weight_max: (
            np.zeros((0, 0), dtype=np.float32),
            [],
            {"dropped_missing": 1, "dropped_structure": 0},
        ),
    )
    monkeypatch.setattr(
        bench.projection_tools,
        "_format_no_valid_training_records_error",
        lambda **kwargs: "no_valid_training_records:test",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with pytest.raises(ValueError, match="no_valid_training_records:test"):
            bench._fit_projection_for_index(
                model_id="openai/clip-vit-large-patch14",
                raw_index_path=tmp_path / "raw_index.npz",
                triplet_path=tmp_path / "triplets.jsonl",
                projection_images_dir=tmp_path,
                projection_path=tmp_path / "proj.npz",
                projection_report_path=tmp_path / "proj.report.json",
                max_triplets=0,
                output_dim=0,
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=1e-4,
                margin=0.1,
                temperature=0.07,
                ce_weight=0.3,
                orth_weight=0.0,
                sample_weight_mode="triplet_weight",
                sample_weight_power=1.0,
                sample_weight_max=3.0,
                seed=42,
                device="cpu",
            )


def test_upgrade_patch_config_persists_projection_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config_path = root / "cfg.json"
        config_path.write_text(json.dumps({"geolocator": {"retrieval_index_paths": ["x"], "retrieval_index_weights": [1.0]}}), encoding="utf-8")
        upgrade._patch_config(
            config_path=config_path,
            best_model="openai/clip-vit-large-patch14",
            final_index_path=root / "final_index.npz",
            final_projection_path=root / "final_projection.npz",
            query_expansion_top_n=0,
            query_expansion_beta=0.0,
            query_expansion_alpha=0.5,
            local_match_top_n=0,
            local_match_weight=0.0,
            local_match_ratio=0.8,
            local_match_max_features=1200,
            graph_rerank_top_n=0,
            graph_rerank_sigma_km=3.0,
            graph_rerank_score_alpha=0.4,
            graph_rerank_support_beta=1.0,
            graph_rerank_center_radius_km=0.0,
            kde_refine_top_n=0,
            kde_refine_sigma_km=2.0,
            kde_refine_score_power=1.0,
            kde_refine_margin_threshold=0.0,
            kde_refine_switch_radius_km=0.0,
            kde_refine_max_iters=8,
            preserve_multi_index=False,
        )
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        assert payload["geolocator"]["retrieval_projection_path"].endswith("final_projection.npz")
        assert payload["geolocator"]["retrieval_index_path"].endswith("final_index.npz")
        assert payload["geolocator"]["retrieval_index_paths"] == []
