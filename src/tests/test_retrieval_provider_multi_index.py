from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.geo.retrieval_provider import GeoRetrievalProvider, LoadedRetrievalIndex, RetrievalIndex


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
