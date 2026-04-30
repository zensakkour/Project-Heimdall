from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.tools import train_crossview_projection as tool


class _FakeEmbedder:
    def __init__(self, model_id: str, device: str, projection_path: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.projection_path = projection_path

    def embed_many(self, images):
        out = []
        for img in images:
            marker = int(np.asarray(img, dtype=np.uint8)[0, 0, 0])
            if marker == 10:
                out.append(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
            else:
                out.append(np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
        return out


@pytest.mark.skipif(tool.torch is None, reason="torch_not_available")
def test_train_crossview_projection_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    street_dir = tmp_path / "street"
    street_dir.mkdir()
    Image.fromarray(np.full((4, 4, 3), 10, dtype=np.uint8)).save(street_dir / "q1.png")
    Image.fromarray(np.full((4, 4, 3), 20, dtype=np.uint8)).save(street_dir / "q2.png")

    triplets_path = tmp_path / "triplets.jsonl"
    triplets_path.write_text(
        json.dumps(
            {
                "query_path": "q1.png",
                "positives": [{"path": "a1.png"}],
                "hard_negatives": [{"path": "a2.png"}],
                "triplet_weight": 2.0,
            }
        )
        + "\n"
        + json.dumps(
            {
                "query_path": "q2.png",
                "positives": [{"path": "a2.png"}],
                "hard_negatives": [{"path": "a1.png"}],
                "triplet_weight": 1.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    aerial_index = tmp_path / "aerial_index.npz"
    emb = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.savez(
        aerial_index,
        embeddings=emb,
        latitudes=np.asarray([48.0, 48.1], dtype=np.float32),
        longitudes=np.asarray([2.0, 2.1], dtype=np.float32),
        ids=np.asarray(["a1", "a2"], dtype=object),
        paths=np.asarray(["a1.png", "a2.png"], dtype=object),
        model_id=np.asarray("openai/clip-vit-large-patch14", dtype=np.str_),
    )

    monkeypatch.setattr(tool, "ClipEmbedder", _FakeEmbedder)

    output_path = tmp_path / "projection.npz"
    code = tool.main(
        [
            "--triplets",
            str(triplets_path),
            "--aerial-index",
            str(aerial_index),
            "--street-images-dir",
            str(street_dir),
            "--output",
            str(output_path),
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--learning-rate",
            "1e-3",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    assert output_path.exists()
    report = json.loads(output_path.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert report["triplets_used"] == 2
    with np.load(output_path, allow_pickle=False) as payload:
        assert payload["matrix"].shape == (3, 3)
        assert payload["bias"].shape == (3,)
