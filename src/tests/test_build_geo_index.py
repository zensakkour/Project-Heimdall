from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.tools import build_geo_index as mod


class _FakeEmbedder:
    def __init__(self, model_id: str, device: str, projection_path: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.projection_path = projection_path
        self.calls: list[int] = []

    def embed_many(self, images):
        self.calls.append(len(images))
        rows = []
        for idx, _ in enumerate(images):
            rows.append(np.array([float(idx + 1), 1.0], dtype=np.float32))
        return np.vstack(rows)


def test_build_index_batches_embeddings(monkeypatch, tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)
    metadata_path = tmp_path / "metadata.csv"
    rows = []
    for idx in range(20):
        image_path = images_dir / f"img_{idx}.jpg"
        Image.new("RGB", (8, 8), (idx, idx, idx)).save(image_path)
        rows.append(
            {
                "id": f"id_{idx}",
                "path": f"images/{image_path.name}",
                "lat": f"{48.85 + (idx * 0.0001):.8f}",
                "lon": f"{2.30 + (idx * 0.0001):.8f}",
            }
        )
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "path", "lat", "lon"])
        writer.writeheader()
        writer.writerows(rows)

    fake = _FakeEmbedder("fake/model", "cpu")
    monkeypatch.setattr(mod, "ClipEmbedder", lambda *args, **kwargs: fake)
    count = mod.build_index(
        images=sorted(images_dir.glob("*.jpg")),
        meta=mod._load_metadata(metadata_path),
        output=tmp_path / "index.npz",
        model_id="fake/model",
        root_dir=tmp_path,
        images_dir=images_dir,
        projection_path=None,
    )

    assert count == 20
    assert sum(fake.calls) == 20
    assert len(fake.calls) >= 1
    assert max(fake.calls) <= 32
