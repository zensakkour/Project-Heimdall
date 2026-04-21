from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.core.geo import retrieval_provider as retrieval_mod
from src.core.geo.retrieval_provider import GeoRetrievalProvider, LoadedRetrievalIndex, RetrievalIndex
from src.core.logic.types import GeoCandidate


class _StubEmbedder:
    def embed(self, _image: Image.Image) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def _index(rows: list[list[float]], ids: list[str]) -> RetrievalIndex:
    emb = np.asarray(rows, dtype=np.float32)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    n = emb.shape[0]
    return RetrievalIndex(
        embeddings=emb,
        latitudes=np.asarray([48.85 + (0.01 * idx) for idx in range(n)], dtype=np.float64),
        longitudes=np.asarray([2.35 + (0.01 * idx) for idx in range(n)], dtype=np.float64),
        ids=np.asarray(ids, dtype=np.str_),
        paths=np.asarray([f"{item}.jpg" for item in ids], dtype=np.str_),
    )


def _loaded(path: Path, source: str, idx: RetrievalIndex, model_id: str = "openai/clip-vit-large-patch14") -> LoadedRetrievalIndex:
    return LoadedRetrievalIndex(source=source, path=path, model_id=model_id, index=idx)


def _index_with_coords(
    rows: list[list[float]],
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


def test_multi_index_weights_influence_top_candidate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "q.jpg"
        Image.new("RGB", (16, 16), color=(80, 50, 20)).save(image_path)
        first = root / "a.npz"
        second = root / "b.npz"
        first.write_bytes(b"ok")
        second.write_bytes(b"ok")

        idx_a = _index([[0.95, 0.05]], ["a"])
        idx_b = _index([[1.0, 0.0]], ["b"])

        provider = GeoRetrievalProvider(
            index_path=str(first),
            index_paths=[str(second)],
            index_weights=[1.0, 0.2],
            top_k=2,
            min_score=-1.0,
        )
        provider._ensure_indices = lambda: [  # type: ignore[method-assign]
            _loaded(first, "a_idx", idx_a),
            _loaded(second, "b_idx", idx_b),
        ]
        provider._ensure_embedder = lambda: _StubEmbedder()  # type: ignore[method-assign]
        out = provider.candidates(str(image_path))
        assert out
        assert out[0].match_id == "retrieval:a_idx:a"

        provider2 = GeoRetrievalProvider(
            index_path=str(first),
            index_paths=[str(second)],
            index_weights=[0.2, 1.0],
            top_k=2,
            min_score=-1.0,
        )
        provider2._ensure_indices = provider._ensure_indices  # type: ignore[method-assign]
        provider2._ensure_embedder = provider._ensure_embedder  # type: ignore[method-assign]
        out2 = provider2.candidates(str(image_path))
        assert out2
        assert out2[0].match_id == "retrieval:b_idx:b"


def test_per_index_top_k_prevents_single_source_domination() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "q.jpg"
        Image.new("RGB", (16, 16), color=(70, 70, 70)).save(image_path)
        first = root / "a.npz"
        second = root / "b.npz"
        first.write_bytes(b"ok")
        second.write_bytes(b"ok")

        idx_a = _index([[1.0, 0.0], [0.99, 0.01]], ["a1", "a2"])
        idx_b = _index([[0.98, 0.02]], ["b1"])

        provider = GeoRetrievalProvider(
            index_path=str(first),
            index_paths=[str(second)],
            top_k=2,
            per_index_top_k=1,
            min_score=-1.0,
        )
        provider._ensure_indices = lambda: [  # type: ignore[method-assign]
            _loaded(first, "a_idx", idx_a),
            _loaded(second, "b_idx", idx_b),
        ]
        provider._ensure_embedder = lambda: _StubEmbedder()  # type: ignore[method-assign]
        out = provider.candidates(str(image_path))
        ids = {cand.match_id for cand in out}
        assert "retrieval:a_idx:a1" in ids
        assert "retrieval:b_idx:b1" in ids


def test_index_score_norm_zscore_sigmoid_rebalances_heterogeneous_indices() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "q.jpg"
        Image.new("RGB", (16, 16), color=(50, 50, 80)).save(image_path)
        first = root / "a.npz"
        second = root / "b.npz"
        first.write_bytes(b"ok")
        second.write_bytes(b"ok")

        # Source A: higher absolute scores but very tight spread.
        idx_a = _index(
            [
                [0.90, 0.4358899],
                [0.89, 0.4559605],
                [0.88, 0.4749737],
                [0.87, 0.4930517],
            ],
            ["a1", "a2", "a3", "a4"],
        )
        # Source B: lower absolute max but much stronger relative separation.
        idx_b = _index(
            [
                [0.70, 0.7141428],
                [0.10, 0.9949874],
                [0.09, 0.9959418],
                [0.08, 0.9967949],
            ],
            ["b1", "b2", "b3", "b4"],
        )

        base = GeoRetrievalProvider(
            index_path=str(first),
            index_paths=[str(second)],
            top_k=1,
            per_index_top_k=1,
            index_score_norm="none",
            min_score=-1.0,
        )
        base._ensure_indices = lambda: [  # type: ignore[method-assign]
            _loaded(first, "a_idx", idx_a),
            _loaded(second, "b_idx", idx_b),
        ]
        base._ensure_embedder = lambda: _StubEmbedder()  # type: ignore[method-assign]
        out_base = base.candidates(str(image_path))
        assert out_base
        assert out_base[0].match_id == "retrieval:a_idx:a1"

        normed = GeoRetrievalProvider(
            index_path=str(first),
            index_paths=[str(second)],
            top_k=1,
            per_index_top_k=1,
            index_score_norm="zscore_sigmoid",
            min_score=-1.0,
        )
        normed._ensure_indices = base._ensure_indices  # type: ignore[method-assign]
        normed._ensure_embedder = base._ensure_embedder  # type: ignore[method-assign]
        out_normed = normed.candidates(str(image_path))
        assert out_normed
        assert out_normed[0].match_id == "retrieval:b_idx:b1"


def test_index_score_norm_auto_uses_multi_index_normalization() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "q.jpg"
        Image.new("RGB", (16, 16), color=(40, 40, 90)).save(image_path)
        first = root / "a.npz"
        second = root / "b.npz"
        first.write_bytes(b"ok")
        second.write_bytes(b"ok")
        idx_a = _index(
            [
                [0.90, 0.4358899],
                [0.89, 0.4559605],
                [0.88, 0.4749737],
                [0.87, 0.4930517],
            ],
            ["a1", "a2", "a3", "a4"],
        )
        idx_b = _index(
            [
                [0.70, 0.7141428],
                [0.10, 0.9949874],
                [0.09, 0.9959418],
                [0.08, 0.9967949],
            ],
            ["b1", "b2", "b3", "b4"],
        )

        auto = GeoRetrievalProvider(
            index_path=str(first),
            index_paths=[str(second)],
            top_k=1,
            per_index_top_k=1,
            index_score_norm="auto",
            min_score=-1.0,
        )
        auto._ensure_indices = lambda: [  # type: ignore[method-assign]
            _loaded(first, "a_idx", idx_a),
            _loaded(second, "b_idx", idx_b),
        ]
        auto._ensure_embedder = lambda: _StubEmbedder()  # type: ignore[method-assign]
        out = auto.candidates(str(image_path))
        assert out
        assert out[0].match_id == "retrieval:b_idx:b1"


def test_source_balance_beta_promotes_multi_source_topk() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "q.jpg"
        Image.new("RGB", (16, 16), color=(90, 30, 30)).save(image_path)
        first = root / "a.npz"
        second = root / "b.npz"
        first.write_bytes(b"ok")
        second.write_bytes(b"ok")

        idx_a = _index([[1.0, 0.0], [0.99, 0.01]], ["a1", "a2"])
        idx_b = _index([[0.98, 0.02]], ["b1"])

        plain = GeoRetrievalProvider(
            index_path=str(first),
            index_paths=[str(second)],
            top_k=2,
            per_index_top_k=2,
            index_score_norm="none",
            source_balance_beta=0.0,
            min_score=-1.0,
        )
        plain._ensure_indices = lambda: [  # type: ignore[method-assign]
            _loaded(first, "a_idx", idx_a),
            _loaded(second, "b_idx", idx_b),
        ]
        plain._ensure_embedder = lambda: _StubEmbedder()  # type: ignore[method-assign]
        out_plain = plain.candidates(str(image_path))
        assert [cand.match_id for cand in out_plain[:2]] == ["retrieval:a_idx:a1", "retrieval:a_idx:a2"]

        balanced = GeoRetrievalProvider(
            index_path=str(first),
            index_paths=[str(second)],
            top_k=2,
            per_index_top_k=2,
            index_score_norm="none",
            source_balance_beta=0.8,
            min_score=-1.0,
        )
        balanced._ensure_indices = plain._ensure_indices  # type: ignore[method-assign]
        balanced._ensure_embedder = plain._ensure_embedder  # type: ignore[method-assign]
        out_balanced = balanced.candidates(str(image_path))
        ids = {cand.match_id for cand in out_balanced[:2]}
        assert "retrieval:a_idx:a1" in ids
        assert "retrieval:b_idx:b1" in ids


def test_multi_index_supports_per_index_model_embeddings() -> None:
    class _ClipStub:
        def embed(self, _image: Image.Image) -> np.ndarray:
            return np.asarray([1.0, 0.0], dtype=np.float32)

    class _SiglipStub:
        def embed(self, _image: Image.Image) -> np.ndarray:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "q.jpg"
        Image.new("RGB", (16, 16), color=(20, 80, 90)).save(image_path)
        first = root / "a.npz"
        second = root / "b.npz"
        first.write_bytes(b"ok")
        second.write_bytes(b"ok")

        idx_a = _index([[1.0, 0.0]], ["a1"])
        idx_b = RetrievalIndex(
            embeddings=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            latitudes=np.asarray([48.86], dtype=np.float64),
            longitudes=np.asarray([2.36], dtype=np.float64),
            ids=np.asarray(["b1"], dtype=np.str_),
            paths=np.asarray(["b1.jpg"], dtype=np.str_),
        )
        idx_b.embeddings = idx_b.embeddings / np.clip(np.linalg.norm(idx_b.embeddings, axis=1, keepdims=True), 1e-12, None)

        provider = GeoRetrievalProvider(
            index_path=str(first),
            index_paths=[str(second)],
            index_model_ids=["openai/clip-vit-large-patch14", "google/siglip-base-patch16-224"],
            top_k=2,
            per_index_top_k=1,
            min_score=-1.0,
        )
        provider._ensure_indices = lambda: [  # type: ignore[method-assign]
            _loaded(first, "clip_idx", idx_a, "openai/clip-vit-large-patch14"),
            _loaded(second, "siglip_idx", idx_b, "google/siglip-base-patch16-224"),
        ]
        provider._ensure_embedder = lambda: _ClipStub()  # type: ignore[method-assign]
        provider._ensure_embedder_for_model = (  # type: ignore[method-assign]
            lambda model_id: _ClipStub()
            if model_id == "openai/clip-vit-large-patch14"
            else _SiglipStub()
        )
        out = provider.candidates(str(image_path))
        ids = {cand.match_id for cand in out}
        assert "retrieval:clip_idx:a1" in ids
        assert "retrieval:siglip_idx:b1" in ids


def test_source_fusion_mode_rrf_aggregates_cross_source_support() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "q.jpg"
        Image.new("RGB", (16, 16), color=(32, 32, 32)).save(image_path)
        first = root / "a.npz"
        second = root / "b.npz"
        first.write_bytes(b"ok")
        second.write_bytes(b"ok")

        idx_a = _index_with_coords(
            rows=[[1.0, 0.0], [0.95, 0.3122499]],
            ids=["a_unique", "shared_a"],
            lats=[10.0, 20.0],
            lons=[10.0, 20.0],
        )
        idx_b = _index_with_coords(
            rows=[[0.96, 0.28], [0.94, 0.3411744]],
            ids=["shared_b", "b_unique"],
            lats=[20.0, 30.0],
            lons=[20.0, 30.0],
        )

        weighted = GeoRetrievalProvider(
            index_path=str(first),
            index_paths=[str(second)],
            top_k=1,
            per_index_top_k=2,
            index_score_norm="none",
            source_fusion_mode="weighted_score",
            min_score=-1.0,
        )
        weighted._ensure_indices = lambda: [  # type: ignore[method-assign]
            _loaded(first, "a_idx", idx_a),
            _loaded(second, "b_idx", idx_b),
        ]
        weighted._ensure_embedder = lambda: _StubEmbedder()  # type: ignore[method-assign]
        out_weighted = weighted.candidates(str(image_path))
        assert out_weighted
        assert out_weighted[0].match_id == "retrieval:a_idx:a_unique"

        rrf = GeoRetrievalProvider(
            index_path=str(first),
            index_paths=[str(second)],
            top_k=1,
            per_index_top_k=2,
            index_score_norm="none",
            source_fusion_mode="rrf",
            min_score=-1.0,
        )
        rrf._ensure_indices = weighted._ensure_indices  # type: ignore[method-assign]
        rrf._ensure_embedder = weighted._ensure_embedder  # type: ignore[method-assign]
        out_rrf = rrf.candidates(str(image_path))
        assert out_rrf
        assert abs(out_rrf[0].latitude - 20.0) < 1e-6
        assert abs(out_rrf[0].longitude - 20.0) < 1e-6


def test_query_expansion_can_promote_dense_nearby_cluster() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "q.jpg"
        Image.new("RGB", (16, 16), color=(25, 25, 25)).save(image_path)
        index_path = root / "idx.npz"
        index_path.write_bytes(b"ok")

        # Query is [1, 0]. "anchor" wins raw top-1, while b1/b2 form a dense
        # secondary cluster that query-expansion can pull the query towards.
        idx = _index(
            rows=[
                [1.0, 0.0],      # anchor
                [0.8, 0.6],      # b1
                [0.79, 0.61],    # b2
            ],
            ids=["anchor", "b1", "b2"],
        )

        base = GeoRetrievalProvider(
            index_path=str(index_path),
            top_k=1,
            min_score=-1.0,
            query_expansion_top_n=0,
        )
        base._ensure_indices = lambda: [_loaded(index_path, "idx", idx)]  # type: ignore[method-assign]
        base._ensure_embedder = lambda: _StubEmbedder()  # type: ignore[method-assign]
        out_base = base.candidates(str(image_path))
        assert out_base
        assert out_base[0].match_id == "retrieval:idx:anchor"

        expanded = GeoRetrievalProvider(
            index_path=str(index_path),
            top_k=1,
            min_score=-1.0,
            query_expansion_top_n=3,
            query_expansion_beta=0.9,
            query_expansion_alpha=0.1,
        )
        expanded._ensure_indices = base._ensure_indices  # type: ignore[method-assign]
        expanded._ensure_embedder = base._ensure_embedder  # type: ignore[method-assign]
        out_expanded = expanded.candidates(str(image_path))
        assert out_expanded
        assert out_expanded[0].match_id in {"retrieval:idx:b1", "retrieval:idx:b2"}


def test_local_match_rerank_can_promote_geometric_match(monkeypatch: pytest.MonkeyPatch) -> None:
    ranked = [
        GeoCandidate(latitude=48.85, longitude=2.35, retrieval_score=0.92, match_id="a", image_path="a.jpg"),
        GeoCandidate(latitude=48.86, longitude=2.36, retrieval_score=0.90, match_id="b", image_path="b.jpg"),
    ]

    class _DummyOrb:
        def detectAndCompute(self, _img, _mask):
            return [object() for _ in range(20)], np.ones((20, 32), dtype=np.uint8)

    class _DummyMatcher:
        pass

    class _DummyCV2:
        NORM_HAMMING = 6

        def ORB_create(self, **_kwargs):
            return _DummyOrb()

        def BFMatcher(self, *_args, **_kwargs):
            return _DummyMatcher()

    monkeypatch.setattr(retrieval_mod, "cv2", _DummyCV2(), raising=False)
    monkeypatch.setattr(retrieval_mod, "_resolve_candidate_image_path", lambda raw: Path(str(raw)))

    def _fake_score(*, cand_path: Path, **_kwargs) -> float:
        return 0.05 if cand_path.name == "a.jpg" else 0.95

    monkeypatch.setattr(retrieval_mod, "_score_local_feature_match", _fake_score)
    out = retrieval_mod._apply_local_match_rerank(
        ranked,
        query_gray=np.zeros((16, 16), dtype=np.uint8),
        top_n=2,
        weight=0.9,
        ratio_test=0.8,
        max_features=512,
    )
    assert out
    assert out[0].match_id == "b"


def test_local_match_rerank_skips_when_signal_is_weak(monkeypatch: pytest.MonkeyPatch) -> None:
    ranked = [
        GeoCandidate(latitude=48.85, longitude=2.35, retrieval_score=0.92, match_id="a", image_path="a.jpg"),
        GeoCandidate(latitude=48.86, longitude=2.36, retrieval_score=0.90, match_id="b", image_path="b.jpg"),
    ]

    class _DummyOrb:
        def detectAndCompute(self, _img, _mask):
            return [object() for _ in range(20)], np.ones((20, 32), dtype=np.uint8)

    class _DummyMatcher:
        pass

    class _DummyCV2:
        NORM_HAMMING = 6

        def ORB_create(self, **_kwargs):
            return _DummyOrb()

        def BFMatcher(self, *_args, **_kwargs):
            return _DummyMatcher()

    monkeypatch.setattr(retrieval_mod, "cv2", _DummyCV2(), raising=False)
    monkeypatch.setattr(retrieval_mod, "_resolve_candidate_image_path", lambda raw: Path(str(raw)))
    monkeypatch.setattr(retrieval_mod, "_score_local_feature_match", lambda **_kwargs: 0.05)
    out = retrieval_mod._apply_local_match_rerank(
        ranked,
        query_gray=np.zeros((16, 16), dtype=np.uint8),
        top_n=2,
        weight=0.9,
        ratio_test=0.8,
        max_features=512,
    )
    assert out
    assert [cand.match_id for cand in out[:2]] == ["a", "b"]


def test_local_match_rerank_skips_confident_override_without_strong_local_advantage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranked = [
        GeoCandidate(latitude=48.85, longitude=2.35, retrieval_score=0.99, match_id="a", image_path="a.jpg"),
        GeoCandidate(latitude=48.86, longitude=2.36, retrieval_score=0.70, match_id="b", image_path="b.jpg"),
    ]

    class _DummyOrb:
        def detectAndCompute(self, _img, _mask):
            return [object() for _ in range(20)], np.ones((20, 32), dtype=np.uint8)

    class _DummyMatcher:
        pass

    class _DummyCV2:
        NORM_HAMMING = 6

        def ORB_create(self, **_kwargs):
            return _DummyOrb()

        def BFMatcher(self, *_args, **_kwargs):
            return _DummyMatcher()

    monkeypatch.setattr(retrieval_mod, "cv2", _DummyCV2(), raising=False)
    monkeypatch.setattr(retrieval_mod, "_resolve_candidate_image_path", lambda raw: Path(str(raw)))

    def _fake_score(*, cand_path: Path, **_kwargs) -> float:
        return 0.30 if cand_path.name == "a.jpg" else 0.45

    monkeypatch.setattr(retrieval_mod, "_score_local_feature_match", _fake_score)
    out = retrieval_mod._apply_local_match_rerank(
        ranked,
        query_gray=np.zeros((16, 16), dtype=np.uint8),
        top_n=2,
        weight=0.9,
        ratio_test=0.8,
        max_features=512,
    )
    assert out
    assert [cand.match_id for cand in out[:2]] == ["a", "b"]


def test_local_match_rerank_allows_confident_override_when_local_advantage_is_strong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranked = [
        GeoCandidate(latitude=48.85, longitude=2.35, retrieval_score=0.95, match_id="a", image_path="a.jpg"),
        GeoCandidate(latitude=48.86, longitude=2.36, retrieval_score=0.84, match_id="b", image_path="b.jpg"),
    ]

    class _DummyOrb:
        def detectAndCompute(self, _img, _mask):
            return [object() for _ in range(20)], np.ones((20, 32), dtype=np.uint8)

    class _DummyMatcher:
        pass

    class _DummyCV2:
        NORM_HAMMING = 6

        def ORB_create(self, **_kwargs):
            return _DummyOrb()

        def BFMatcher(self, *_args, **_kwargs):
            return _DummyMatcher()

    monkeypatch.setattr(retrieval_mod, "cv2", _DummyCV2(), raising=False)
    monkeypatch.setattr(retrieval_mod, "_resolve_candidate_image_path", lambda raw: Path(str(raw)))

    def _fake_score(*, cand_path: Path, **_kwargs) -> float:
        return 0.30 if cand_path.name == "a.jpg" else 0.95

    monkeypatch.setattr(retrieval_mod, "_score_local_feature_match", _fake_score)
    out = retrieval_mod._apply_local_match_rerank(
        ranked,
        query_gray=np.zeros((16, 16), dtype=np.uint8),
        top_n=2,
        weight=0.9,
        ratio_test=0.8,
        max_features=512,
    )
    assert out
    assert out[0].match_id == "b"


def test_graph_support_rerank_can_promote_dense_cluster() -> None:
    ranked = [
        GeoCandidate(latitude=10.0, longitude=10.0, retrieval_score=0.95, match_id="isolated"),
        GeoCandidate(latitude=20.0, longitude=20.0, retrieval_score=0.86, match_id="cluster_a"),
        GeoCandidate(latitude=20.01, longitude=20.01, retrieval_score=0.85, match_id="cluster_b"),
        GeoCandidate(latitude=40.0, longitude=40.0, retrieval_score=0.80, match_id="far_tail"),
    ]

    out = retrieval_mod._apply_graph_support_rerank(
        ranked,
        top_n=4,
        sigma_km=3.0,
        score_alpha=0.4,
        support_beta=1.2,
        center_radius_km=0.0,
    )
    assert out
    assert out[0].match_id in {"cluster_a", "cluster_b"}

    centered = retrieval_mod._apply_graph_support_rerank(
        ranked,
        top_n=4,
        sigma_km=3.0,
        score_alpha=0.4,
        support_beta=1.2,
        center_radius_km=2.5,
    )
    assert centered
    assert centered[0].match_id in {"cluster_a", "cluster_b", "retrieval:graph_support_consensus"}


def test_kde_mode_refine_can_promote_dense_mode() -> None:
    ranked = [
        GeoCandidate(latitude=10.0, longitude=10.0, retrieval_score=0.98, match_id="isolated"),
        GeoCandidate(latitude=20.0, longitude=20.0, retrieval_score=0.90, match_id="cluster_a"),
        GeoCandidate(latitude=20.01, longitude=20.01, retrieval_score=0.89, match_id="cluster_b"),
        GeoCandidate(latitude=19.99, longitude=20.02, retrieval_score=0.88, match_id="cluster_c"),
    ]

    out = retrieval_mod._apply_kde_mode_refinement(
        ranked,
        top_n=4,
        sigma_km=2.0,
        score_power=1.0,
        margin_threshold=0.0,
        switch_radius_km=0.0,
        max_iters=8,
    )
    assert out
    assert out[0].match_id == "retrieval:kde_mode"
    assert out[0].latitude > 15.0


def test_kde_mode_refine_respects_confident_top1_guard() -> None:
    ranked = [
        GeoCandidate(latitude=10.0, longitude=10.0, retrieval_score=0.99, match_id="isolated"),
        GeoCandidate(latitude=20.0, longitude=20.0, retrieval_score=0.70, match_id="cluster_a"),
        GeoCandidate(latitude=20.01, longitude=20.01, retrieval_score=0.69, match_id="cluster_b"),
        GeoCandidate(latitude=19.99, longitude=20.02, retrieval_score=0.68, match_id="cluster_c"),
    ]

    out = retrieval_mod._apply_kde_mode_refinement(
        ranked,
        top_n=4,
        sigma_km=2.0,
        score_power=1.0,
        margin_threshold=0.2,
        switch_radius_km=2.0,
        max_iters=8,
    )
    assert out
    assert out[0].match_id == "isolated"


def test_kde_mode_refine_adaptive_mass_changes_solution() -> None:
    ranked = [
        GeoCandidate(latitude=10.0, longitude=10.0, retrieval_score=0.98, match_id="isolated"),
        GeoCandidate(latitude=20.0, longitude=20.0, retrieval_score=0.90, match_id="cluster_a"),
        GeoCandidate(latitude=20.01, longitude=20.01, retrieval_score=0.89, match_id="cluster_b"),
        GeoCandidate(latitude=19.99, longitude=20.02, retrieval_score=0.88, match_id="cluster_c"),
    ]

    full = retrieval_mod._apply_kde_mode_refinement(
        ranked,
        top_n=4,
        sigma_km=2.0,
        score_power=1.0,
        margin_threshold=0.0,
        switch_radius_km=0.0,
        max_iters=8,
        adaptive_mass=0.0,
    )
    narrow = retrieval_mod._apply_kde_mode_refinement(
        ranked,
        top_n=4,
        sigma_km=2.0,
        score_power=1.0,
        margin_threshold=0.0,
        switch_radius_km=0.0,
        max_iters=8,
        adaptive_mass=0.5,
    )
    assert full and narrow
    assert abs(float(full[0].latitude) - float(narrow[0].latitude)) > 0.01
