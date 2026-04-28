from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.tools import train_retrieval_projection as trp


def test_build_training_records_applies_triplet_weight_cap_and_power() -> None:
    embeddings = {
        "a.jpg": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "b.jpg": np.asarray([0.9, 0.1, 0.0], dtype=np.float32),
        "c.jpg": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    }
    mat, rows, stats = trp._build_training_records(
        [
            {
                "query_path": "a.jpg",
                "positives": [{"path": "b.jpg"}],
                "hard_negatives": [{"path": "c.jpg"}],
                "triplet_weight": 4.0,
            }
        ],
        by_exact=dict(embeddings),
        by_name={},
        sample_weight_mode="triplet_weight",
        sample_weight_power=0.5,
        sample_weight_max=2.25,
    )

    assert mat.shape == (3, 3)
    assert len(rows) == 1
    assert rows[0].sample_weight == pytest.approx(1.5)
    assert stats["triplets_with_explicit_weight"] == 1
    assert stats["sample_weight_mean"] == pytest.approx(1.5)


def test_format_no_valid_training_records_error_hints_reference_index_mismatch() -> None:
    message = trp._format_no_valid_training_records_error(
        triplets_loaded=1,
        requested_unique_paths=3,
        dataset_stats={"dropped_missing": 1, "dropped_structure": 0},
        missing_summary={
            "query_missing": 0,
            "positive_missing": 1,
            "negative_missing": 1,
            "query_examples": [],
            "positive_examples": ["ref_pos.jpg"],
            "negative_examples": ["ref_neg.jpg"],
        },
        embedding_index="query_only_index.npz",
        images_dir="data/query/chips",
    )

    assert message.startswith("no_valid_training_records:")
    assert "missing_positive=1" in message
    assert "missing_negative=1" in message
    assert "--reference-metadata" in message
    assert "reference index instead of the query index" in message


@pytest.mark.skipif(trp.torch is None, reason="torch_not_available")
def test_train_retrieval_projection_from_index_only() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        triplets_path = root / "triplets.jsonl"
        index_path = root / "index.npz"
        output_path = root / "projection.npz"

        triplets_path.write_text(
            json.dumps(
                {
                    "query_path": "a.jpg",
                    "positives": [{"path": "b.jpg"}],
                    "hard_negatives": [{"path": "c.jpg"}, {"path": "d.jpg"}],
                    "triplet_weight": 2.5,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        embeddings = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
            ],
            dtype=np.float32,
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        np.savez(
            index_path,
            embeddings=embeddings,
            latitudes=np.asarray([48.0, 48.1, 48.2, 48.3], dtype=np.float32),
            longitudes=np.asarray([2.0, 2.1, 2.2, 2.3], dtype=np.float32),
            ids=np.asarray(["a", "b", "c", "d"], dtype=object),
            paths=np.asarray(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], dtype=object),
            model_id=np.asarray("openai/clip-vit-large-patch14", dtype=np.str_),
        )

        code = trp.main(
            [
                "--triplets",
                str(triplets_path),
                "--embedding-index",
                str(index_path),
                "--output",
                str(output_path),
                "--epochs",
                "2",
                "--batch-size",
                "1",
                "--sample-weight-mode",
                "triplet_weight",
                "--sample-weight-max",
                "2.0",
                "--device",
                "cpu",
            ]
        )
        assert code == 0
        assert output_path.exists()
        report_path = output_path.with_suffix(".report.json")
        assert report_path.exists()

        with np.load(output_path, allow_pickle=False) as data:
            matrix = np.asarray(data["matrix"], dtype=np.float32)
            bias = np.asarray(data["bias"], dtype=np.float32)
            assert matrix.shape == (4, 4)
            assert bias.shape == (4,)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["triplets_used"] == 1
        assert report["missing_by_role_before_embed"]["query_missing"] == 0
        assert report["missing_by_role_after_embed"]["negative_missing"] == 0
        assert report["dataset_stats"]["embedding_dim"] == 4
        assert report["dataset_stats"]["triplets_with_explicit_weight"] == 1
        assert report["dataset_stats"]["sample_weight_max"] == pytest.approx(2.0)
        assert report["training"]["sample_weight_mode"] == "triplet_weight"
