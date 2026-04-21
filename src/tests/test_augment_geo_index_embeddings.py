from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.tools import augment_geo_index_embeddings as tool


def test_augment_geo_index_embeddings_main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        index_path = root / "index.npz"
        out_path = root / "index_dba.npz"

        embeddings = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.95, 0.05, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
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

        code = tool.main(
            [
                "--index",
                str(index_path),
                "--output",
                str(out_path),
                "--neighbors",
                "2",
                "--self-weight",
                "1.0",
                "--temperature",
                "0.07",
                "--min-similarity",
                "0.0",
            ]
        )
        assert code == 0
        assert out_path.exists()

        with np.load(out_path, allow_pickle=True) as data:
            out = np.asarray(data["embeddings"], dtype=np.float32)
            assert out.shape == embeddings.shape
            norms = np.linalg.norm(out, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5)
            assert "dba_neighbors" in data
            assert int(data["dba_neighbors"]) == 2
            assert "dba_source_index" in data

            # Row 0 should shift slightly toward its close neighbors.
            assert float(out[0, 1]) > float(embeddings[0, 1])


def test_augment_embeddings_neighbors_zero_is_noop() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    out = tool.augment_embeddings(
        embeddings,
        neighbors=0,
        self_weight=1.0,
        min_similarity=-1.0,
        temperature=0.07,
    )
    assert np.allclose(out, embeddings, atol=1e-6)


def test_augment_embeddings_geo_radius_filters_far_neighbors() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.02, 0.0],
            [0.97, 0.03, 0.0],
        ],
        dtype=np.float32,
    )
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    lat = np.asarray([48.8566, 48.8570, 40.7128], dtype=np.float64)  # row 2 is far away
    lon = np.asarray([2.3522, 2.3529, -74.0060], dtype=np.float64)

    out = tool.augment_embeddings(
        embeddings,
        neighbors=2,
        self_weight=1.0,
        min_similarity=-1.0,
        temperature=0.07,
        latitudes=lat,
        longitudes=lon,
        max_geo_distance_km=0.5,
    )
    # Row 0 should move toward row 1 (nearby Paris point), not toward row 2 (NYC).
    assert float(out[0, 1]) > float(embeddings[0, 1])
