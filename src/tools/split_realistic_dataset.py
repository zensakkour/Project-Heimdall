"""Create leakage-safe spatial splits for the realistic Paris dataset."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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
        out.append(item)
    return out, {"south": south, "west": west, "north": north, "east": east}


def spatial_split_pairs(
    rows: Sequence[dict],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    cell_size_m: float,
    seed: int,
) -> Tuple[Dict[str, List[dict]], dict]:
    total_ratio = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("split_ratios_must_sum_to_1")

    with_cells, bbox = _assign_cell_ids(rows, cell_size_m=cell_size_m)
    grouped: Dict[str, List[dict]] = {}
    for row in with_cells:
        grouped.setdefault(str(row["cell_id"]), []).append(row)

    items = list(grouped.items())
    rng = random.Random(int(seed))
    rng.shuffle(items)

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
    }
    return splits, summary


def min_cross_split_distance_m(splits: Dict[str, Sequence[dict]]) -> Optional[float]:
    names = ("train", "val", "test")
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
) -> dict:
    rows = _load_pairs(pairs_path)
    splits, summary = spatial_split_pairs(
        rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        cell_size_m=cell_size_m,
        seed=seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "train_pairs.csv", splits["train"])
    _write_csv(out_dir / "val_pairs.csv", splits["val"])
    _write_csv(out_dir / "test_pairs.csv", splits["test"])

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
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
