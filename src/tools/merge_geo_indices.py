"""
Merge multiple geo retrieval indices into a single deduplicated index.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class IndexItem:
    embedding: np.ndarray
    latitude: float
    longitude: float
    match_id: str
    path: str


def _load_items(path: Path) -> List[IndexItem]:
    required = ("embeddings", "latitudes", "longitudes", "ids", "paths")
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{path}: missing keys: {','.join(missing)}")
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            latitudes = np.asarray(data["latitudes"], dtype=np.float64)
            longitudes = np.asarray(data["longitudes"], dtype=np.float64)
            ids = np.asarray(data["ids"])
            paths = np.asarray(data["paths"])
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{path}: missing keys: {','.join(missing)}")
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            latitudes = np.asarray(data["latitudes"], dtype=np.float64)
            longitudes = np.asarray(data["longitudes"], dtype=np.float64)
            ids = np.asarray(data["ids"])
            paths = np.asarray(data["paths"])

    if embeddings.ndim != 2:
        raise ValueError(f"{path}: embeddings must be 2D")
    n = embeddings.shape[0]
    if n <= 0:
        return []
    if latitudes.shape[0] != n or longitudes.shape[0] != n or ids.shape[0] != n or paths.shape[0] != n:
        raise ValueError(f"{path}: array length mismatch")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    items: List[IndexItem] = []
    for idx in range(n):
        lat = float(latitudes[idx])
        lon = float(longitudes[idx])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        items.append(
            IndexItem(
                embedding=embeddings[idx].astype(np.float32, copy=False),
                latitude=lat,
                longitude=lon,
                match_id=str(ids[idx]),
                path=str(paths[idx]),
            )
        )
    return items


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    return radius * c


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    return max(-1.0, min(1.0, dot))


def _grid_key(lat: float, lon: float, cell_deg: float) -> Tuple[int, int]:
    return int(math.floor(lat / cell_deg)), int(math.floor(lon / cell_deg))


def merge_indices(
    paths: Iterable[Path],
    dedupe_radius_m: float = 0.0,
    cosine_sim_threshold: float = 0.999,
    max_items: int = 0,
) -> Dict[str, np.ndarray]:
    merged: List[IndexItem] = []
    seen_exact = set()
    for path in paths:
        for item in _load_items(path):
            exact_key = (item.match_id, item.path)
            if exact_key in seen_exact:
                continue
            seen_exact.add(exact_key)
            merged.append(item)

    if dedupe_radius_m > 0.0 and merged:
        cell_deg = max(dedupe_radius_m / 111_320.0, 1e-6)
        grid: Dict[Tuple[int, int], List[int]] = {}
        kept: List[IndexItem] = []
        for item in merged:
            lat_key, lon_key = _grid_key(item.latitude, item.longitude, cell_deg)
            is_dup = False
            for dlat in (-1, 0, 1):
                for dlon in (-1, 0, 1):
                    neighbor = (lat_key + dlat, lon_key + dlon)
                    for idx in grid.get(neighbor, []):
                        current = kept[idx]
                        if _haversine_m(item.latitude, item.longitude, current.latitude, current.longitude) > dedupe_radius_m:
                            continue
                        if _cosine_similarity(item.embedding, current.embedding) >= cosine_sim_threshold:
                            is_dup = True
                            break
                    if is_dup:
                        break
                if is_dup:
                    break
            if is_dup:
                continue
            keep_idx = len(kept)
            kept.append(item)
            grid.setdefault((lat_key, lon_key), []).append(keep_idx)
        merged = kept

    if max_items > 0 and len(merged) > max_items:
        merged = merged[:max_items]

    if not merged:
        raise ValueError("merged index is empty")

    embeddings = np.stack([item.embedding for item in merged]).astype(np.float32)
    latitudes = np.asarray([item.latitude for item in merged], dtype=np.float64)
    longitudes = np.asarray([item.longitude for item in merged], dtype=np.float64)
    ids = np.asarray([item.match_id for item in merged], dtype=np.str_)
    paths_arr = np.asarray([item.path for item in merged], dtype=np.str_)
    return {
        "embeddings": embeddings,
        "latitudes": latitudes,
        "longitudes": longitudes,
        "ids": ids,
        "paths": paths_arr,
    }


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge geo retrieval indices.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input .npz index paths.")
    parser.add_argument("--output", required=True, help="Output merged .npz path.")
    parser.add_argument("--dedupe-radius-m", type=float, default=0.0, help="Spatial dedupe radius in meters.")
    parser.add_argument(
        "--cosine-sim-threshold",
        type=float,
        default=0.999,
        help="Embedding cosine threshold for spatial dedupe.",
    )
    parser.add_argument("--max-items", type=int, default=0, help="Optional cap on final merged size.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    input_paths = [Path(path) for path in args.inputs]
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing input indices: {', '.join(str(path) for path in missing)}")

    payload = merge_indices(
        input_paths,
        dedupe_radius_m=max(0.0, float(args.dedupe_radius_m)),
        cosine_sim_threshold=max(-1.0, min(1.0, float(args.cosine_sim_threshold))),
        max_items=max(0, int(args.max_items)),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Merged {len(input_paths)} indices -> {out_path} ({payload['embeddings'].shape[0]} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
