from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.tools import apply_projection_to_geo_index as tool


def test_apply_projection_to_geo_index() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        index_path = root / "index.npz"
        proj_path = root / "proj.npz"
        out_path = root / "out.npz"

        embeddings = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.savez(
            index_path,
            embeddings=embeddings,
            latitudes=np.asarray([48.0, 48.1, 48.2], dtype=np.float32),
            longitudes=np.asarray([2.0, 2.1, 2.2], dtype=np.float32),
            ids=np.asarray(["a", "b", "c"], dtype=object),
            paths=np.asarray(["a.jpg", "b.jpg", "c.jpg"], dtype=object),
            model_id=np.asarray("openai/clip-vit-large-patch14", dtype=np.str_),
        )

        weight = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        bias = np.asarray([0.1, -0.2], dtype=np.float32)
        np.savez(proj_path, matrix=weight, bias=bias)

        code = tool.main(
            [
                "--index",
                str(index_path),
                "--projection-path",
                str(proj_path),
                "--output",
                str(out_path),
            ]
        )
        assert code == 0
        assert out_path.exists()

        with np.load(out_path, allow_pickle=True) as data:
            arr = np.asarray(data["embeddings"], dtype=np.float32)
            assert arr.shape == (3, 2)
            norms = np.linalg.norm(arr, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5)
            assert "projection_path" in data
