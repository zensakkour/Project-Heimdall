from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from src.tools.recover_combined_aerial_dataset import recover_combined_dataset


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_recover_combined_dataset_preseeds_existing_images_and_builds_split(tmp_path: Path) -> None:
    existing_images_dir = tmp_path / "existing" / "aerial" / "images"
    existing_images_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(existing_images_dir / "ign_geopf_ortho_demo.png")

    chunk_meta_dir = tmp_path / "chunkmeta"
    _write_csv(
        chunk_meta_dir / "street_combined_chunk_00.csv",
        [
            "image_id",
            "path",
            "lat",
            "lon",
            "heading_deg",
            "captured_at",
            "camera_type",
            "width",
            "height",
            "quality_score",
            "sequence",
            "source",
            "license_info",
        ],
        [
            {
                "image_id": "demo",
                "path": "images/demo.jpg",
                "lat": "48.85000000",
                "lon": "2.35000000",
                "heading_deg": "90.0",
                "captured_at": "",
                "camera_type": "perspective",
                "width": "640",
                "height": "480",
                "quality_score": "1.0",
                "sequence": "seq-1",
                "source": "panoramax",
                "license_info": "test",
            }
        ],
    )

    final_out_dir = tmp_path / "final"
    split_out_dir = final_out_dir / "splits_strict"
    summary = recover_combined_dataset(
        existing_images_dir=existing_images_dir,
        chunk_meta_dir=chunk_meta_dir,
        chunk_out_dir=tmp_path / "chunkpairs",
        final_out_dir=final_out_dir,
        split_out_dir=split_out_dir,
        provider="ign_geopf",
        crop_size_m=256.0,
        crop_px=512,
        allow_missing_aerial=False,
        seed=42,
        max_workers=1,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        cell_size_m=300.0,
        buffer_cells=2,
        sort_axis="auto",
    )

    assert summary["preseed"]["total_preseeded"] == 1
    assert summary["merged"]["aerial_rows"] == 1
    assert summary["merged"]["pair_rows"] == 1
    assert (final_out_dir / "aerial" / "images" / "ign_geopf_ortho_demo.png").exists()
    assert (final_out_dir / "pairs.csv").exists()
    assert summary["split"]["split_counts"]["train"] == 1
    assert summary["split"]["retained_pairs"] == 1
