from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.tools import train_retrieval_projection as trp


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
        assert report["dataset_stats"]["embedding_dim"] == 4
