from __future__ import annotations

import csv
import json
from pathlib import Path

from src.tools.split_realistic_dataset import (
    build_realistic_splits,
    min_cross_split_distance_m,
    sanity_check_split_dir,
    spatial_split_pairs,
)


def test_spatial_split_pairs_keeps_cells_together() -> None:
    rows = [
        {"pair_id": "a", "lat": "48.85000000", "lon": "2.30000000"},
        {"pair_id": "b", "lat": "48.85010000", "lon": "2.30010000"},
        {"pair_id": "c", "lat": "48.86000000", "lon": "2.31000000"},
        {"pair_id": "d", "lat": "48.87000000", "lon": "2.32000000"},
    ]
    splits, summary = spatial_split_pairs(
        rows,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        cell_size_m=300.0,
        seed=42,
    )
    pair_to_split = {}
    for split_name, split_rows in splits.items():
        for row in split_rows:
            pair_to_split[row["pair_id"]] = split_name
    assert pair_to_split["a"] == pair_to_split["b"]
    assert summary["total_pairs"] == 4


def test_build_realistic_splits_writes_csv_and_summary(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.csv"
    pairs_path.write_text(
        "pair_id,street_id,street_path,aerial_id,aerial_path,lat,lon,heading_deg\n"
        "p1,s1,street/1.jpg,a1,aerial/1.png,48.8500,2.3000,90\n"
        "p2,s2,street/2.jpg,a2,aerial/2.png,48.8501,2.3001,91\n"
        "p3,s3,street/3.jpg,a3,aerial/3.png,48.8600,2.3100,92\n"
        "p4,s4,street/4.jpg,a4,aerial/4.png,48.8700,2.3200,93\n",
        encoding="utf-8",
    )

    summary = build_realistic_splits(
        pairs_path=pairs_path,
        out_dir=tmp_path / "splits",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        cell_size_m=300.0,
        seed=42,
    )

    assert (tmp_path / "splits" / "train_pairs.csv").exists()
    assert (tmp_path / "splits" / "val_pairs.csv").exists()
    assert (tmp_path / "splits" / "test_pairs.csv").exists()
    assert (tmp_path / "splits" / "split_summary.json").exists()
    payload = json.loads((tmp_path / "splits" / "split_summary.json").read_text(encoding="utf-8"))
    assert payload["total_pairs"] == 4
    assert "min_cross_split_distance_m" in payload
    train_rows = list(csv.DictReader((tmp_path / "splits" / "train_pairs.csv").open("r", encoding="utf-8")))
    assert train_rows
    assert summary["split_counts"]["train"] == len(train_rows)


def test_sanity_check_split_dir_reports_min_distance(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in {
        "train_pairs.csv": "pair_id,lat,lon\np1,48.8500,2.3000\n",
        "val_pairs.csv": "pair_id,lat,lon\np2,48.8600,2.3100\n",
        "test_pairs.csv": "pair_id,lat,lon\np3,48.8700,2.3200\n",
    }.items():
        (split_dir / name).write_text(rows, encoding="utf-8")

    report = sanity_check_split_dir(split_dir)
    assert report["min_cross_split_distance_m"] is not None
    assert report["split_counts"]["train"] == 1


def test_min_cross_split_distance_handles_empty_splits() -> None:
    minimum = min_cross_split_distance_m({"train": [], "val": [], "test": []})
    assert minimum is None
