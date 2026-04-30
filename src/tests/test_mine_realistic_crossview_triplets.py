from __future__ import annotations

from pathlib import Path

from src.tools import mine_realistic_crossview_triplets as tool


def test_mine_triplets_includes_exact_pair_and_ring_negatives(tmp_path: Path) -> None:
    pairs_csv = tmp_path / "pairs.csv"
    pairs_csv.write_text(
        "pair_id,street_id,street_path,aerial_id,aerial_path,lat,lon,heading_deg\n"
        "p1,s1,street/q1.jpg,a1,aerial/a1.png,48.8566,2.3522,90\n",
        encoding="utf-8",
    )
    aerial_csv = tmp_path / "aerial.csv"
    aerial_csv.write_text(
        "aerial_id,path,lat,lon\n"
        "a1,aerial/a1.png,48.8566,2.3522\n"
        "a2,aerial/a2.png,48.8570,2.3524\n"
        "a3,aerial/a3.png,48.8600,2.3600\n",
        encoding="utf-8",
    )

    triplets, summary = tool.mine_triplets(
        pairs=tool.load_pairs_csv(pairs_csv),
        aerial_records=tool.load_aerial_metadata(aerial_csv),
        positive_radius_m=80.0,
        negative_min_distance_m=300.0,
        negative_max_distance_m=5000.0,
        max_positives=3,
        max_negatives=5,
        limit=0,
        seed=42,
    )

    assert summary["triplets_written"] == 1
    assert len(triplets) == 1
    row = triplets[0]
    assert row["query_path"] == "street/q1.jpg"
    assert row["positive_ids"][0] == "a1"
    assert "a3" in row["negative_ids"]
    assert row["triplet_weight"] >= 1.0

