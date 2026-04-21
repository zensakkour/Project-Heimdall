"""
Descriptor database augmentation (DBA) for geo retrieval indices.

This transforms each index embedding by blending it with its nearest neighbors:
    x'_i = normalize(self_weight * x_i + sum_j w_ij * x_j)
where neighbors j are selected from top-k cosine neighbors in the index.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_index(path: Path) -> dict:
    required = ("embeddings", "latitudes", "longitudes", "ids", "paths")
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [k for k in required if k not in data]
            if missing:
                raise ValueError(f"index_missing_keys:{','.join(missing)}")
            payload = {k: data[k] for k in data.files}
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise
        with np.load(path, allow_pickle=True) as data:
            missing = [k for k in required if k not in data]
            if missing:
                raise ValueError(f"index_missing_keys:{','.join(missing)}")
            payload = {k: data[k] for k in data.files}
    return payload


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def augment_embeddings(
    embeddings: np.ndarray,
    *,
    neighbors: int,
    self_weight: float,
    min_similarity: float,
    temperature: float,
    latitudes: np.ndarray | None = None,
    longitudes: np.ndarray | None = None,
    max_geo_distance_km: float | None = None,
) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError("index_embeddings_invalid_shape")
    if not np.isfinite(arr).all():
        raise ValueError("index_embeddings_not_finite")

    arr = _normalize_rows(arr)
    n_rows = int(arr.shape[0])
    if n_rows <= 1 or neighbors <= 0:
        return arr

    k = min(max(0, int(neighbors)), n_rows - 1)
    if k == 0:
        return arr
    if not np.isfinite(self_weight) or self_weight < 0.0:
        raise ValueError("self_weight_invalid")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_invalid")

    sim = arr @ arr.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    top = np.take_along_axis(sim, idx, axis=1)

    if np.isfinite(min_similarity):
        mask = top >= float(min_similarity)
    else:
        mask = np.ones_like(top, dtype=bool)

    if max_geo_distance_km is not None and np.isfinite(max_geo_distance_km) and float(max_geo_distance_km) > 0.0:
        if latitudes is None or longitudes is None:
            raise ValueError("geo_radius_requires_latlon")
        lats = np.asarray(latitudes, dtype=np.float64).reshape(-1)
        lons = np.asarray(longitudes, dtype=np.float64).reshape(-1)
        if lats.shape[0] != n_rows or lons.shape[0] != n_rows:
            raise ValueError("latlon_length_mismatch")
        if not np.isfinite(lats).all() or not np.isfinite(lons).all():
            raise ValueError("latlon_not_finite")

        lat_q = lats[:, None]
        lon_q = lons[:, None]
        lat_k = lats[idx]
        lon_k = lons[idx]
        d_km = _haversine_km(lat_q, lon_q, lat_k, lon_k)
        mask = mask & (d_km <= float(max_geo_distance_km))

    logits = top / float(temperature)
    logits = np.where(mask, logits, -np.inf)
    row_max = np.max(logits, axis=1, keepdims=True)
    finite_rows = np.isfinite(row_max).reshape(-1)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exp_logits = np.exp(logits - row_max)
    exp_logits = np.where(mask, exp_logits, 0.0)
    denom = np.sum(exp_logits, axis=1, keepdims=True)
    denom = np.where(denom > 1e-12, denom, 1.0)
    weights = exp_logits / denom
    if not finite_rows.all():
        weights[~finite_rows, :] = 0.0

    augmented = arr * float(self_weight)
    for r in range(n_rows):
        nbr_idx = idx[r]
        nbr_w = weights[r].astype(np.float32, copy=False)
        if np.all(nbr_w <= 0.0):
            continue
        augmented[r] = augmented[r] + (nbr_w[:, None] * arr[nbr_idx]).sum(axis=0)

    augmented = _normalize_rows(augmented.astype(np.float32, copy=False))
    if not np.isfinite(augmented).all():
        raise ValueError("augmented_embeddings_not_finite")
    return augmented


def _haversine_km(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    r = 6371.0
    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    lat2 = np.radians(lat2_deg)
    lon2 = np.radians(lon2_deg)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon * 0.5) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    return (r * c).astype(np.float32, copy=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply database-side augmentation (DBA) to a geo index.")
    parser.add_argument("--index", required=True, help="Input .npz index.")
    parser.add_argument("--output", required=True, help="Output .npz index path.")
    parser.add_argument("--neighbors", type=int, default=10, help="Number of nearest neighbors per row.")
    parser.add_argument(
        "--self-weight",
        type=float,
        default=1.0,
        help="Weight assigned to original descriptor before adding neighbors.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=-1.0,
        help="Minimum cosine similarity for neighbor inclusion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Softmax temperature for neighbor weights.",
    )
    parser.add_argument(
        "--max-geo-distance-km",
        type=float,
        default=0.0,
        help="If >0, only use neighbors within this geographic radius (km).",
    )
    args = parser.parse_args(argv)

    index_path = Path(args.index)
    output_path = Path(args.output)
    if not index_path.exists():
        raise FileNotFoundError(f"index_not_found:{index_path}")

    payload = _load_index(index_path)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    latitudes = np.asarray(payload["latitudes"], dtype=np.float64)
    longitudes = np.asarray(payload["longitudes"], dtype=np.float64)
    max_geo_distance_km = float(args.max_geo_distance_km)
    augmented = augment_embeddings(
        embeddings,
        neighbors=int(args.neighbors),
        self_weight=float(args.self_weight),
        min_similarity=float(args.min_similarity),
        temperature=float(args.temperature),
        latitudes=latitudes,
        longitudes=longitudes,
        max_geo_distance_km=max_geo_distance_km,
    )

    output_payload = dict(payload)
    output_payload["embeddings"] = augmented
    output_payload["dba_neighbors"] = np.asarray(int(args.neighbors), dtype=np.int32)
    output_payload["dba_self_weight"] = np.asarray(float(args.self_weight), dtype=np.float32)
    output_payload["dba_min_similarity"] = np.asarray(float(args.min_similarity), dtype=np.float32)
    output_payload["dba_temperature"] = np.asarray(float(args.temperature), dtype=np.float32)
    output_payload["dba_max_geo_distance_km"] = np.asarray(max_geo_distance_km, dtype=np.float32)
    output_payload["dba_source_index"] = np.asarray(str(index_path).replace("\\", "/"), dtype=np.str_)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **output_payload)
    print(
        "wrote DBA index -> {path} ({rows} rows, {dims} dims, k={k})".format(
            path=output_path,
            rows=int(augmented.shape[0]),
            dims=int(augmented.shape[1]),
            k=int(args.neighbors),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
