from __future__ import annotations

from pathlib import Path
import tempfile

from PIL import Image

from src.tools import train_retrieval_encoder as tool


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(120, 80, 40)).save(path)


def test_resolve_triplets_uses_query_and_reference_roots() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        query_dir = root / "eval" / "chips"
        ref_dir = root / "train" / "chips"
        _write_rgb(query_dir / "q.jpg")
        _write_rgb(ref_dir / "p.jpg")
        _write_rgb(ref_dir / "n.jpg")

        triplets = [
            {
                "query_path": "q.jpg",
                "triplet_weight": 2.5,
                "positives": [{"path": "p.jpg"}],
                "hard_negatives": [{"path": "n.jpg"}],
            }
        ]

        rows, stats = tool._resolve_triplets(
            triplets=triplets,
            query_images_dir=query_dir,
            reference_images_dir=ref_dir,
            max_triplets=0,
            max_positives_per_row=2,
            max_negatives_per_row=2,
            sample_weight_mode="triplet_weight",
            sample_weight_power=1.0,
            sample_weight_max=3.0,
        )

        assert stats["triplets_resolved"] == 1
        assert len(rows) == 1
        assert rows[0].query_path.endswith("q.jpg")
        assert rows[0].positive_paths[0].endswith("p.jpg")
        assert rows[0].negative_paths[0].endswith("n.jpg")
        assert rows[0].sample_weight == 2.5


def test_sample_batch_rows_selects_paths_and_weights() -> None:
    rows = [
        tool.ResolvedTriplet(
            query_path="q.jpg",
            positive_paths=("p1.jpg", "p2.jpg"),
            negative_paths=("n1.jpg", "n2.jpg"),
            sample_weight=1.5,
        )
    ]

    q_paths, p_paths, n_paths, weights = tool._sample_batch_rows(rows, batch_ids=[0], rng=__import__("random").Random(42))
    assert q_paths == ["q.jpg"]
    assert len(p_paths) == 1
    assert len(n_paths) == 1
    assert p_paths[0] in {"p1.jpg", "p2.jpg"}
    assert n_paths[0] in {"n1.jpg", "n2.jpg"}
    assert weights == [1.5]


def test_collect_unique_paths_dedupes_and_preserves_order() -> None:
    rows = [
        tool.ResolvedTriplet(
            query_path="q.jpg",
            positive_paths=("p1.jpg", "p2.jpg"),
            negative_paths=("n1.jpg",),
            sample_weight=1.0,
        ),
        tool.ResolvedTriplet(
            query_path="q.jpg",
            positive_paths=("p2.jpg", "p3.jpg"),
            negative_paths=("n2.jpg", "n1.jpg"),
            sample_weight=1.0,
        ),
    ]

    assert tool._collect_unique_paths(rows) == [
        "q.jpg",
        "p1.jpg",
        "p2.jpg",
        "n1.jpg",
        "p3.jpg",
        "n2.jpg",
    ]


def test_evaluate_rows_uses_feature_cache_for_projection_only_path() -> None:
    if tool.rp.torch is None:
        return

    torch = tool.rp.torch

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual_projection = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.visual_projection.weight.copy_(torch.eye(2))

    model = FakeModel()
    rows = [
        tool.ResolvedTriplet(
            query_path="q.jpg",
            positive_paths=("p1.jpg", "p2.jpg"),
            negative_paths=("n1.jpg", "n2.jpg"),
            sample_weight=2.0,
        )
    ]
    cache = {
        "q.jpg": torch.tensor([1.0, 0.0]),
        "p1.jpg": torch.tensor([1.0, 0.0]),
        "p2.jpg": torch.tensor([0.9, 0.1]),
        "n1.jpg": torch.tensor([0.0, 1.0]),
        "n2.jpg": torch.tensor([0.1, 0.9]),
    }

    stats = tool._evaluate_rows(
        model=model,
        processor=None,
        rows=rows,
        margin=0.1,
        device="cpu",
        feature_cache=cache,
    )

    assert stats["count"] == 1
    assert stats["triplet_satisfied_pct"] == 100.0
    assert stats["weighted_triplet_satisfied_pct"] == 100.0


def test_load_pretrained_falls_back_to_local_files_only() -> None:
    calls = []

    class FakeFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            calls.append((model_id, dict(kwargs)))
            if not kwargs.get("local_files_only"):
                raise RuntimeError("network_timeout")
            return {"model_id": model_id, "kwargs": dict(kwargs)}

    loaded = tool._load_pretrained(FakeFactory, "openai/clip-vit-large-patch14")

    assert calls == [
        ("openai/clip-vit-large-patch14", {}),
        ("openai/clip-vit-large-patch14", {"local_files_only": True}),
    ]
    assert loaded["kwargs"]["local_files_only"] is True
