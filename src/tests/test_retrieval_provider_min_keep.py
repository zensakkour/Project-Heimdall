from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.geo.retrieval_provider import GeoRetrievalProvider, RetrievalIndex


class _StubEmbedder:
    def embed(self, _image: Image.Image) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def _make_index() -> RetrievalIndex:
    emb = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    return RetrievalIndex(
        embeddings=emb,
        latitudes=np.asarray([48.85, 48.86, 48.87], dtype=np.float64),
        longitudes=np.asarray([2.35, 2.36, 2.37], dtype=np.float64),
        ids=np.asarray(["a", "b", "c"], dtype=np.str_),
        paths=np.asarray(["a.jpg", "b.jpg", "c.jpg"], dtype=np.str_),
    )


def test_min_keep_topk_preserves_candidates_when_threshold_filters_all() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "q.jpg"
        Image.new("RGB", (16, 16), color=(100, 20, 20)).save(image_path)
        marker = Path(tmpdir) / "idx.npz"
        marker.write_bytes(b"ok")

        provider = GeoRetrievalProvider(
            index_path=str(marker),
            top_k=3,
            min_score=1.1,
            min_keep_topk=2,
        )
        provider._ensure_indices = lambda: [  # type: ignore[method-assign]
            type("Loaded", (), {"source": "stub", "path": marker, "index": _make_index()})()
        ]
        provider._ensure_embedder = lambda: _StubEmbedder()  # type: ignore[method-assign]

        out = provider.candidates(str(image_path))
        assert len(out) == 2
        assert out[0].match_id in {
            "retrieval:stub:a",
            "retrieval:stub:b",
            "retrieval:stub:c",
        }


def test_min_keep_topk_zero_keeps_strict_threshold_behavior() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "q.jpg"
        Image.new("RGB", (16, 16), color=(100, 20, 20)).save(image_path)
        marker = Path(tmpdir) / "idx.npz"
        marker.write_bytes(b"ok")

        provider = GeoRetrievalProvider(
            index_path=str(marker),
            top_k=3,
            min_score=1.1,
            min_keep_topk=0,
        )
        provider._ensure_indices = lambda: [  # type: ignore[method-assign]
            type("Loaded", (), {"source": "stub", "path": marker, "index": _make_index()})()
        ]
        provider._ensure_embedder = lambda: _StubEmbedder()  # type: ignore[method-assign]

        out = provider.candidates(str(image_path))
        assert out == []
