from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.tools.merge_geo_indices import merge_indices


def _write_index(
    path: Path,
    embeddings: list[list[float]],
    latitudes: list[float],
    longitudes: list[float],
    ids: list[str],
    paths: list[str],
    *,
    use_object_arrays: bool = False,
) -> None:
    str_dtype = object if use_object_arrays else np.str_
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        latitudes=np.asarray(latitudes, dtype=np.float64),
        longitudes=np.asarray(longitudes, dtype=np.float64),
        ids=np.asarray(ids, dtype=str_dtype),
        paths=np.asarray(paths, dtype=str_dtype),
    )


def test_merge_indices_dedupes_exact_keys() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        first = root / "a.npz"
        second = root / "b.npz"
        _write_index(
            first,
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            latitudes=[48.85, 40.71],
            longitudes=[2.35, -74.0],
            ids=["id_a", "id_b"],
            paths=["a.jpg", "b.jpg"],
        )
        _write_index(
            second,
            embeddings=[[0.0, 1.0], [0.7, 0.7]],
            latitudes=[40.71, 35.68],
            longitudes=[-74.0, 139.65],
            ids=["id_b", "id_c"],
            paths=["b.jpg", "c.jpg"],
        )

        merged = merge_indices([first, second])
        assert merged["embeddings"].shape[0] == 3
        assert set(merged["ids"].tolist()) == {"id_a", "id_b", "id_c"}


def test_merge_indices_spatial_dedupe_with_similarity() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        first = root / "a.npz"
        second = root / "b.npz"
        _write_index(
            first,
            embeddings=[[1.0, 0.0]],
            latitudes=[48.856600],
            longitudes=[2.352200],
            ids=["id_a"],
            paths=["a.jpg"],
        )
        _write_index(
            second,
            embeddings=[[0.9999, 0.01]],
            latitudes=[48.856620],
            longitudes=[2.352210],
            ids=["id_b"],
            paths=["b.jpg"],
        )

        merged = merge_indices(
            [first, second],
            dedupe_radius_m=100.0,
            cosine_sim_threshold=0.99,
        )
        assert merged["embeddings"].shape[0] == 1


def test_merge_indices_supports_legacy_object_id_arrays() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        legacy = root / "legacy.npz"
        _write_index(
            legacy,
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            latitudes=[48.85, 40.71],
            longitudes=[2.35, -74.0],
            ids=["id_a", "id_b"],
            paths=["a.jpg", "b.jpg"],
            use_object_arrays=True,
        )
        merged = merge_indices([legacy])
        assert merged["embeddings"].shape[0] == 2
