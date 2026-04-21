"""
Apply a learned projection (matrix+bias) to an existing geo index.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_index(path: Path) -> dict:
    try:
        with np.load(path, allow_pickle=False) as data:
            required = ("embeddings", "latitudes", "longitudes", "ids", "paths")
            missing = [k for k in required if k not in data]
            if missing:
                raise ValueError(f"index_missing_keys:{','.join(missing)}")
            payload = {k: data[k] for k in data.files}
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise
        with np.load(path, allow_pickle=True) as data:
            required = ("embeddings", "latitudes", "longitudes", "ids", "paths")
            missing = [k for k in required if k not in data]
            if missing:
                raise ValueError(f"index_missing_keys:{','.join(missing)}")
            payload = {k: data[k] for k in data.files}
    return payload


def _load_projection(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            weight = data.get("matrix")
            if weight is None:
                weight = data.get("weight")
            bias = data.get("bias")
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise
        with np.load(path, allow_pickle=True) as data:
            weight = data.get("matrix")
            if weight is None:
                weight = data.get("weight")
            bias = data.get("bias")
    if weight is None:
        raise ValueError("projection_missing_matrix")
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2 or w.shape[0] <= 0 or w.shape[1] <= 0:
        raise ValueError("projection_matrix_invalid_shape")
    if not np.isfinite(w).all():
        raise ValueError("projection_matrix_not_finite")
    if bias is None:
        b = np.zeros(w.shape[0], dtype=np.float32)
    else:
        b = np.asarray(bias, dtype=np.float32).reshape(-1)
    if b.shape[0] != w.shape[0]:
        raise ValueError("projection_bias_dim_mismatch")
    if not np.isfinite(b).all():
        raise ValueError("projection_bias_not_finite")
    return w, b


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply projection to geo index embeddings.")
    parser.add_argument("--index", required=True, help="Input .npz index.")
    parser.add_argument("--projection-path", required=True, help="Projection .npz containing matrix+bias.")
    parser.add_argument("--output", required=True, help="Output .npz index path.")
    args = parser.parse_args(argv)

    index_path = Path(args.index)
    proj_path = Path(args.projection_path)
    out_path = Path(args.output)
    if not index_path.exists():
        raise FileNotFoundError(f"index_not_found:{index_path}")
    if not proj_path.exists():
        raise FileNotFoundError(f"projection_not_found:{proj_path}")

    payload = _load_index(index_path)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0:
        raise ValueError("index_embeddings_invalid_shape")
    if not np.isfinite(embeddings).all():
        raise ValueError("index_embeddings_not_finite")

    weight, bias = _load_projection(proj_path)
    if embeddings.shape[1] != weight.shape[1]:
        raise ValueError("projection_input_dim_mismatch")

    transformed = embeddings @ weight.T
    transformed = transformed + bias[None, :]
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    transformed = transformed / np.clip(norms, 1e-12, None)
    if not np.isfinite(transformed).all():
        raise ValueError("transformed_embeddings_not_finite")

    output_payload = dict(payload)
    output_payload["embeddings"] = transformed.astype(np.float32)
    output_payload["projection_path"] = np.asarray(str(proj_path).replace("\\", "/"), dtype=np.str_)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **output_payload)
    print(f"wrote projected index -> {out_path} ({transformed.shape[0]} rows, {transformed.shape[1]} dims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
