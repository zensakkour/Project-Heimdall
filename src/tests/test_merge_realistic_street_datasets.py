from __future__ import annotations

import csv
from pathlib import Path

from src.tools.merge_realistic_street_datasets import merge_street_datasets


def test_merge_street_datasets_copies_images_and_writes_metadata(tmp_path: Path) -> None:
    ds1 = tmp_path / "mapillary"
    ds2 = tmp_path / "panoramax"
    (ds1 / "images").mkdir(parents=True, exist_ok=True)
    (ds2 / "images").mkdir(parents=True, exist_ok=True)
    (ds1 / "images" / "a.jpg").write_bytes(b"a")
    (ds2 / "images" / "b.jpg").write_bytes(b"b")
    (ds1 / "metadata.csv").write_text(
        "image_id,path,lat,lon,heading_deg,captured_at,camera_type,width,height,quality_score,sequence,source,license_info\n"
        "img-a,images/a.jpg,48.85,2.30,90,,,,,,,mapillary,cc\n",
        encoding="utf-8",
    )
    (ds2 / "metadata.csv").write_text(
        "image_id,path,lat,lon,heading_deg,captured_at,camera_type,width,height,quality_score,sequence,source,license_info\n"
        "img-b,images/b.jpg,48.86,2.31,91,,,,,,,panoramax:ign,ol\n",
        encoding="utf-8",
    )

    summary = merge_street_datasets(
        metadata_paths=[ds1 / "metadata.csv", ds2 / "metadata.csv"],
        out_dir=tmp_path / "merged",
    )
    assert summary["merged_count"] == 2
    rows = list(csv.DictReader((tmp_path / "merged" / "metadata.csv").open("r", encoding="utf-8")))
    assert len(rows) == 2
    assert (tmp_path / "merged" / rows[0]["path"]).exists()
    assert (tmp_path / "merged" / rows[1]["path"]).exists()
