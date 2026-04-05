from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.core.geo.retrieval_provider import load_index


def _write_index(path: Path, *, object_arrays: bool) -> None:
    str_dtype = object if object_arrays else np.str_
    np.savez_compressed(
        path,
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        latitudes=np.asarray([48.8566, 40.7128], dtype=np.float64),
        longitudes=np.asarray([2.3522, -74.0060], dtype=np.float64),
        ids=np.asarray(["paris", "nyc"], dtype=str_dtype),
        paths=np.asarray(["a.jpg", "b.jpg"], dtype=str_dtype),
    )


def test_load_index_supports_object_id_arrays() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "legacy.npz"
        _write_index(path, object_arrays=True)
        idx = load_index(path)
        assert idx.embeddings.shape == (2, 2)
        assert idx.ids.shape[0] == 2
        assert idx.paths.shape[0] == 2


def test_load_index_reads_model_id_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "with_model.npz"
        np.savez_compressed(
            path,
            embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            latitudes=np.asarray([48.8566], dtype=np.float64),
            longitudes=np.asarray([2.3522], dtype=np.float64),
            ids=np.asarray(["paris"], dtype=np.str_),
            paths=np.asarray(["a.jpg"], dtype=np.str_),
            model_id=np.asarray("google/siglip-base-patch16-224", dtype=np.str_),
        )
        idx = load_index(path)
        assert idx.model_id == "google/siglip-base-patch16-224"
