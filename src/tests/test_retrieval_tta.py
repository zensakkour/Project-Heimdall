from __future__ import annotations

import numpy as np
from PIL import Image

from src.core.geo.retrieval_provider import (
    GeoRetrievalProvider,
    _aggregate_tta_scores,
    _normalize_tta_degrees,
    _query_embeddings,
)


def test_normalize_tta_degrees_dedupes_and_wraps() -> None:
    vals = _normalize_tta_degrees([0, 360, -360, 90, -270, 180, -180, float("nan"), "x"])  # type: ignore[list-item]
    assert vals == [0.0, 90.0, -180.0]


def test_aggregate_tta_scores_mean_median_and_max() -> None:
    scores = np.asarray(
        [
            [0.10, 0.50, 0.20],
            [0.90, 0.40, 0.60],
        ],
        dtype=np.float32,
    )
    mean_out = _aggregate_tta_scores(scores, mode="mean")
    median_out = _aggregate_tta_scores(scores, mode="median")
    max_out = _aggregate_tta_scores(scores, mode="max")
    assert np.allclose(mean_out, np.asarray([0.26666668, 0.6333333], dtype=np.float32))
    assert np.allclose(median_out, np.asarray([0.2, 0.6], dtype=np.float32))
    assert np.allclose(max_out, np.asarray([0.5, 0.9], dtype=np.float32))


def test_aggregate_tta_scores_rrf_is_rank_based() -> None:
    scores = np.asarray(
        [
            [0.10, 0.90, 0.20],  # best in col2 only
            [0.80, 0.20, 0.85],  # best in col1+col3
            [0.70, 0.40, 0.10],  # mid overall
        ],
        dtype=np.float32,
    )
    rrf = _aggregate_tta_scores(scores, mode="rrf")
    # Candidate 1 (index=1) should rank highest due to top ranks in 2/3 views.
    assert int(np.argmax(rrf)) == 1
    assert float(np.min(rrf)) >= 0.0
    assert float(np.max(rrf)) <= 1.0


def test_query_embeddings_runs_all_tta_variants_and_normalizes() -> None:
    class StubEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, _image: Image.Image) -> np.ndarray:
            self.calls += 1
            # Vary magnitude by call count; output should be normalized per row.
            return np.asarray([float(self.calls), 0.0, 0.0], dtype=np.float32)

    image = Image.new("RGB", (16, 16), color=(120, 10, 10))
    embedder = StubEmbedder()
    mat = _query_embeddings(embedder, image, [0.0, 90.0, 180.0, 270.0])
    assert mat.shape == (4, 3)
    assert embedder.calls == 4
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, np.ones_like(norms))


def test_provider_accepts_rrf_tta_reduce_mode() -> None:
    provider = GeoRetrievalProvider(index_path=None, query_tta_reduce="rrf")
    assert provider.query_tta_reduce == "rrf"


def test_provider_accepts_median_tta_reduce_mode() -> None:
    provider = GeoRetrievalProvider(index_path=None, query_tta_reduce="median")
    assert provider.query_tta_reduce == "median"
