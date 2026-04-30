"""Create leakage-safe spatial splits for the realistic Paris dataset."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - optional acceleration
    np = None


def _optional_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _meters_per_degree_lat(latitude_deg: float) -> float:
    lat = math.radians(float(latitude_deg))
    return (
        111132.92
        - (559.82 * math.cos(2.0 * lat))
        + (1.175 * math.cos(4.0 * lat))
        - (0.0023 * math.cos(6.0 * lat))
    )


def _meters_per_degree_lon(latitude_deg: float) -> float:
    lat = math.radians(float(latitude_deg))
    return (
        (111412.84 * math.cos(lat))
        - (93.5 * math.cos(3.0 * lat))
        + (0.118 * math.cos(5.0 * lat))
    )


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = (math.sin(dphi / 2.0) ** 2) + (math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    return radius_m * c


def _load_pairs(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lat = _optional_float(row.get("lat") or row.get("latitude"))
            lon = _optional_float(row.get("lon") or row.get("longitude"))
            if lat is None or lon is None:
                continue
            out = dict(row)
            out["lat"] = f"{float(lat):.8f}"
            out["lon"] = f"{float(lon):.8f}"
            rows.append(out)
    return rows


def _assign_cell_ids(rows: Sequence[dict], *, cell_size_m: float) -> Tuple[List[dict], dict]:
    if not rows:
        return [], {"south": None, "west": None, "north": None, "east": None}

    latitudes = [float(row["lat"]) for row in rows]
    longitudes = [float(row["lon"]) for row in rows]
    south = min(latitudes)
    west = min(longitudes)
    north = max(latitudes)
    east = max(longitudes)
    avg_lat = (south + north) / 2.0
    lat_step = float(cell_size_m) / max(_meters_per_degree_lat(avg_lat), 1.0)
    lon_step = float(cell_size_m) / max(_meters_per_degree_lon(avg_lat), 1.0)
    lat_step = max(lat_step, 1e-6)
    lon_step = max(lon_step, 1e-6)

    out: List[dict] = []
    for row in rows:
        lat = float(row["lat"])
        lon = float(row["lon"])
        cell_row = int(math.floor((lat - south) / lat_step))
        cell_col = int(math.floor((lon - west) / lon_step))
        item = dict(row)
        item["cell_id"] = f"{cell_row}:{cell_col}"
        item["cell_row"] = cell_row
        item["cell_col"] = cell_col
        out.append(item)
    return out, {"south": south, "west": west, "north": north, "east": east}


def _resolve_sort_axis(bbox: dict, *, sort_axis: str) -> str:
    mode = str(sort_axis).strip().lower()
    if mode in {"lat", "lon"}:
        return mode
    south = _optional_float(bbox.get("south"))
    west = _optional_float(bbox.get("west"))
    north = _optional_float(bbox.get("north"))
    east = _optional_float(bbox.get("east"))
    if None in {south, west, north, east}:
        return "lon"
    avg_lat = (float(south) + float(north)) / 2.0
    width_m = abs(float(east) - float(west)) * _meters_per_degree_lon(avg_lat)
    height_m = abs(float(north) - float(south)) * _meters_per_degree_lat(avg_lat)
    return "lon" if width_m >= height_m else "lat"


def _sort_grouped_cells(
    grouped: Dict[str, List[dict]],
    *,
    sort_axis: str,
    seed: int,
) -> List[Tuple[str, List[dict]]]:
    items: List[Tuple[str, List[dict]]] = list(grouped.items())
    if not items:
        return []

    def key_lat(item: Tuple[str, List[dict]]) -> tuple[int, int, str]:
        rows = item[1]
        return (
            int(rows[0].get("cell_row", 0)),
            int(rows[0].get("cell_col", 0)),
            str(item[0]),
        )

    def key_lon(item: Tuple[str, List[dict]]) -> tuple[int, int, str]:
        rows = item[1]
        return (
            int(rows[0].get("cell_col", 0)),
            int(rows[0].get("cell_row", 0)),
            str(item[0]),
        )

    ordered = sorted(items, key=key_lat if sort_axis == "lat" else key_lon)
    rng = random.Random(int(seed))
    if rng.random() >= 0.5:
        ordered.reverse()
    return ordered


def _apply_boundary_buffer(
    splits: Dict[str, List[dict]],
    *,
    buffer_cells: int,
) -> Tuple[Dict[str, List[dict]], List[dict], Dict[str, int]]:
    if int(buffer_cells) <= 0:
        return {name: list(rows) for name, rows in splits.items()}, [], {"train": 0, "val": 0, "test": 0}

    cell_assignments: Dict[Tuple[int, int], str] = {}
    grouped_rows: Dict[Tuple[int, int], List[dict]] = {}
    for split_name, rows in splits.items():
        for row in rows:
            coord = (int(row.get("cell_row", 0)), int(row.get("cell_col", 0)))
            cell_assignments[coord] = split_name
            grouped_rows.setdefault(coord, []).append(row)

    excluded_cells: set[Tuple[int, int]] = set()
    margin = max(1, int(buffer_cells))
    for (cell_row, cell_col), split_name in cell_assignments.items():
        for d_row in range(-margin, margin + 1):
            for d_col in range(-margin, margin + 1):
                if d_row == 0 and d_col == 0:
                    continue
                neighbor = (cell_row + d_row, cell_col + d_col)
                other_split = cell_assignments.get(neighbor)
                if other_split is None or other_split == split_name:
                    continue
                excluded_cells.add((cell_row, cell_col))
                break
            if (cell_row, cell_col) in excluded_cells:
                break

    kept = {"train": [], "val": [], "test": []}
    excluded_rows: List[dict] = []
    per_split_excluded = {"train": 0, "val": 0, "test": 0}
    for coord, rows in grouped_rows.items():
        split_name = cell_assignments[coord]
        if coord in excluded_cells:
            excluded_rows.extend(rows)
            per_split_excluded[split_name] += len(rows)
            continue
        kept[split_name].extend(rows)
    return kept, excluded_rows, per_split_excluded


def spatial_split_pairs(
    rows: Sequence[dict],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    cell_size_m: float,
    seed: int,
    buffer_cells: int = 0,
    sort_axis: str = "auto",
) -> Tuple[Dict[str, List[dict]], dict]:
    total_ratio = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("split_ratios_must_sum_to_1")

    with_cells, bbox = _assign_cell_ids(rows, cell_size_m=cell_size_m)
    grouped: Dict[str, List[dict]] = {}
    for row in with_cells:
        grouped.setdefault(str(row["cell_id"]), []).append(row)

    resolved_axis = _resolve_sort_axis(bbox, sort_axis=sort_axis)
    items = _sort_grouped_cells(grouped, sort_axis=resolved_axis, seed=int(seed))

    total_rows = len(with_cells)
    train_target = int(round(total_rows * float(train_ratio)))
    val_target = int(round(total_rows * float(val_ratio)))

    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    cell_assignments: Dict[str, str] = {}
    for cell_id, cell_rows in items:
        if len(splits["train"]) < train_target:
            split_name = "train"
        elif len(splits["val"]) < val_target:
            split_name = "val"
        else:
            split_name = "test"
        splits[split_name].extend(cell_rows)
        cell_assignments[cell_id] = split_name

    splits, excluded_rows, excluded_per_split = _apply_boundary_buffer(
        splits,
        buffer_cells=int(max(0, buffer_cells)),
    )

    summary = {
        "total_pairs": total_rows,
        "split_counts": {name: len(rows_) for name, rows_ in splits.items()},
        "cell_counts": {
            name: sum(1 for assigned in cell_assignments.values() if assigned == name)
            for name in ("train", "val", "test")
        },
        "bbox": bbox,
        "seed": int(seed),
        "cell_size_m": float(cell_size_m),
        "buffer_cells": int(max(0, buffer_cells)),
        "sort_axis": resolved_axis,
        "excluded_pairs": len(excluded_rows),
        "excluded_pairs_by_split": excluded_per_split,
        "retained_pairs": sum(len(rows_) for rows_ in splits.values()),
        "retained_fraction": (
            float(sum(len(rows_) for rows_ in splits.values())) / float(total_rows)
            if total_rows > 0
            else 0.0
        ),
    }
    if excluded_rows:
        splits["excluded"] = excluded_rows
    return splits, summary


def min_cross_split_distance_m(splits: Dict[str, Sequence[dict]]) -> Optional[float]:
    names = ("train", "val", "test")
    if np is not None:
        minimum: Optional[float] = None
        cached: Dict[str, Optional[Tuple["np.ndarray", "np.ndarray"]]] = {}

        def get_coords(name: str) -> Optional[Tuple["np.ndarray", "np.ndarray"]]:
            if name in cached:
                return cached[name]
            rows = list(splits.get(name, []))
            if not rows:
                cached[name] = None
                return None
            lats = np.asarray([float(row["lat"]) for row in rows], dtype=np.float64)
            lons = np.asarray([float(row["lon"]) for row in rows], dtype=np.float64)
            cached[name] = (np.radians(lats), np.radians(lons))
            return cached[name]

        earth_radius_m = 6_371_000.0
        for idx, left_name in enumerate(names):
            left_coords = get_coords(left_name)
            if left_coords is None:
                continue
            left_lat, left_lon = left_coords
            for right_name in names[idx + 1 :]:
                right_coords = get_coords(right_name)
                if right_coords is None:
                    continue
                right_lat, right_lon = right_coords
                block = 512
                for start in range(0, int(left_lat.shape[0]), block):
                    lat1 = left_lat[start : start + block][:, None]
                    lon1 = left_lon[start : start + block][:, None]
                    dlat = right_lat[None, :] - lat1
                    dlon = right_lon[None, :] - lon1
                    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(right_lat[None, :]) * (
                        np.sin(dlon / 2.0) ** 2
                    )
                    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
                    chunk_min = float(np.min(c) * earth_radius_m)
                    if minimum is None or chunk_min < minimum:
                        minimum = chunk_min
        return minimum

    minimum: Optional[float] = None
    for idx, left_name in enumerate(names):
        left_rows = list(splits.get(left_name, []))
        for right_name in names[idx + 1 :]:
            right_rows = list(splits.get(right_name, []))
            for left in left_rows:
                for right in right_rows:
                    distance_m = haversine_m(
                        float(left["lat"]),
                        float(left["lon"]),
                        float(right["lat"]),
                        float(right["lon"]),
                    )
                    if minimum is None or distance_m < minimum:
                        minimum = float(distance_m)
    return minimum


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_split_file(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def sanity_check_split_dir(split_dir: Path) -> dict:
    splits = {
        "train": _load_split_file(split_dir / "train_pairs.csv"),
        "val": _load_split_file(split_dir / "val_pairs.csv"),
        "test": _load_split_file(split_dir / "test_pairs.csv"),
    }
    minimum = min_cross_split_distance_m(splits)
    return {
        "split_dir": str(split_dir),
        "min_cross_split_distance_m": minimum,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
    }


def build_realistic_splits(
    *,
    pairs_path: Path,
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    cell_size_m: float,
    seed: int,
    buffer_cells: int = 0,
    sort_axis: str = "auto",
) -> dict:
    rows = _load_pairs(pairs_path)
    splits, summary = spatial_split_pairs(
        rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        cell_size_m=cell_size_m,
        seed=seed,
        buffer_cells=buffer_cells,
        sort_axis=sort_axis,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "train_pairs.csv", splits["train"])
    _write_csv(out_dir / "val_pairs.csv", splits["val"])
    _write_csv(out_dir / "test_pairs.csv", splits["test"])
    if splits.get("excluded"):
        _write_csv(out_dir / "excluded_pairs.csv", splits["excluded"])

    min_distance = min_cross_split_distance_m(splits)
    summary["min_cross_split_distance_m"] = min_distance
    summary_path = out_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["split_summary_path"] = str(summary_path)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create leakage-safe spatial splits for the realistic Paris dataset.")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--cell-size-m", type=float, default=300.0)
    parser.add_argument("--buffer-cells", type=int, default=2)
    parser.add_argument("--sort-axis", default="auto", choices=["auto", "lat", "lon"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sanity-check-dir", default="")
    args = parser.parse_args(argv)

    if str(args.sanity_check_dir).strip():
        report = sanity_check_split_dir(Path(args.sanity_check_dir))
        print(json.dumps(report, indent=2))
        return 0

    if not str(args.pairs).strip() or not str(args.out).strip():
        raise ValueError("pairs_and_out_required")

    summary = build_realistic_splits(
        pairs_path=Path(args.pairs),
        out_dir=Path(args.out),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        cell_size_m=float(args.cell_size_m),
        seed=int(args.seed),
        buffer_cells=int(max(0, args.buffer_cells)),
        sort_axis=str(args.sort_axis),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
