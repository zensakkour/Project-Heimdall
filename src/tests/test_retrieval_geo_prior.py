from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.geo.retrieval_provider import (
    GeoRetrievalProvider,
    LoadedRetrievalIndex,
    RetrievalIndex,
    _apply_geo_prior_bbox,
)
from src.core.logic.types import GeoCandidate


class _StubEmbedder:
    def embed(self, _image: Image.Image) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def _index(
    rows: list[list[float]],
    *,
    ids: list[str],
    lats: list[float],
    lons: list[float],
) -> RetrievalIndex:
    emb = np.asarray(rows, dtype=np.float32)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    return RetrievalIndex(
        embeddings=emb,
        latitudes=np.asarray(lats, dtype=np.float64),
        longitudes=np.asarray(lons, dtype=np.float64),
        ids=np.asarray(ids, dtype=np.str_),
        paths=np.asarray([f"{item}.jpg" for item in ids], dtype=np.str_),
    )


def _loaded(path: Path, source: str, idx: RetrievalIndex) -> LoadedRetrievalIndex:
    return LoadedRetrievalIndex(
        source=source,
        path=path,
        model_id="openai/clip-vit-large-patch14",
        index=idx,
    )


def test_geo_prior_soft_downweights_far_candidate() -> None:
    ranked = [
        GeoCandidate(latitude=40.7128, longitude=-74.0060, retrieval_score=0.9, match_id="far"),
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.85, match_id="near"),
    ]
    out = _apply_geo_prior_bbox(
        ranked,
        mode="soft",
        bbox=(48.4, 49.1, 2.05, 2.36),
        sigma_km=35.0,
        min_keep=0,
    )
    assert out
    assert out[0].match_id == "near"
    assert out[-1].match_id == "far"


def test_geo_prior_hard_filters_outside_bbox() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "q.jpg"
        Image.new("RGB", (16, 16), color=(80, 80, 80)).save(image_path)
        marker = root / "idx.npz"
        marker.write_bytes(b"ok")

        idx = _index(
            [[1.0, 0.0], [0.92, 0.39]],
            ids=["far", "near"],
            lats=[40.7128, 48.8566],
            lons=[-74.0060, 2.3522],
        )
        provider = GeoRetrievalProvider(
            index_path=str(marker),
            top_k=2,
            min_score=0.0,
            geo_prior_mode="hard",
            geo_prior_bbox=[48.4, 49.1, 2.05, 2.36],
            geo_prior_sigma_km=35.0,
            geo_prior_min_keep=0,
        )
        provider._ensure_indices = lambda: [  # type: ignore[method-assign]
            _loaded(marker, "stub", idx)
        ]
        provider._ensure_embedder = lambda: _StubEmbedder()  # type: ignore[method-assign]
        out = provider.candidates(str(image_path))
        assert out
        assert len(out) == 1
        assert out[0].match_id == "retrieval:stub:near"


def test_geo_prior_hard_min_keep_preserves_fallback_when_bbox_excludes_all() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "q.jpg"
        Image.new("RGB", (16, 16), color=(40, 40, 40)).save(image_path)
        marker = root / "idx.npz"
        marker.write_bytes(b"ok")

        idx = _index(
            [[1.0, 0.0], [0.92, 0.39]],
            ids=["a", "b"],
            lats=[40.7128, 48.8566],
            lons=[-74.0060, 2.3522],
        )
        provider = GeoRetrievalProvider(
            index_path=str(marker),
            top_k=2,
            min_score=0.0,
            geo_prior_mode="hard",
            geo_prior_bbox=[10.0, 11.0, 10.0, 11.0],
            geo_prior_sigma_km=35.0,
            geo_prior_min_keep=1,
        )
        provider._ensure_indices = lambda: [  # type: ignore[method-assign]
            _loaded(marker, "stub", idx)
        ]
        provider._ensure_embedder = lambda: _StubEmbedder()  # type: ignore[method-assign]
        out = provider.candidates(str(image_path))
        assert len(out) == 1
        assert out[0].match_id in {"retrieval:stub:a", "retrieval:stub:b"}
