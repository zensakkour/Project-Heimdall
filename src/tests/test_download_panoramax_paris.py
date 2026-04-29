from __future__ import annotations

import csv
from pathlib import Path

from src.tools.download_panoramax_paris import (
    build_search_url,
    download_panoramax_dataset,
    parse_panoramax_item,
)
from src.tools.download_mapillary_paris import GridCell


def test_parse_panoramax_item_extracts_heading_and_assets() -> None:
    image = parse_panoramax_item(
        {
            "id": "pic-1",
            "collection": "seq-1",
            "geometry": {"type": "Point", "coordinates": [2.30, 48.85]},
            "assets": {
                "sd": {"href": "https://example.com/pic-1.jpg"},
            },
            "links": [
                {
                    "rel": "license",
                    "href": "https://example.com/license",
                    "title": "License for this object (etalab-2.0)",
                },
                {"rel": "via", "instance_name": "ign", "href": "https://panoramax.ign.fr"},
            ],
            "properties": {
                "datetime": "2026-01-01T00:00:00Z",
                "view:azimuth": 123,
                "license": "etalab-2.0",
                "panoramax:horizontal_pixel_density": 42,
                "pers:interior_orientation": {"sensor_array_dimensions": [4096, 2160]},
            },
        }
    )
    assert image is not None
    assert image.heading_deg == 123.0
    assert image.source == "panoramax:ign"
    assert image.width == 4096
    assert image.license_info.startswith("etalab-2.0")


def test_build_search_url_encodes_bbox() -> None:
    url = build_search_url(
        endpoint="https://api.panoramax.xyz/api",
        cell=GridCell(row=0, col=0, south=48.8, west=2.2, north=48.81, east=2.21),
        limit=25,
    )
    assert "https://api.panoramax.xyz/api/search?" in url
    assert "bbox=2.20000000%2C48.80000000%2C2.21000000%2C48.81000000" in url
    assert "limit=25" in url


def test_download_panoramax_dataset_writes_metadata(tmp_path: Path) -> None:
    calls = {"count": 0}

    def fake_fetch_json(url: str) -> dict:
        calls["count"] += 1
        return {
            "features": [
                {
                    "id": "pic-1",
                    "collection": "seq-1",
                    "geometry": {"type": "Point", "coordinates": [2.30, 48.85]},
                    "assets": {"sd": {"href": "https://example.com/pic-1.jpg"}},
                    "links": [
                        {"rel": "license", "href": "https://example.com/license", "title": "License"},
                        {"rel": "via", "instance_name": "ign", "href": "https://panoramax.ign.fr"},
                    ],
                    "properties": {
                        "datetime": "2026-01-01T00:00:00Z",
                        "view:azimuth": 123,
                        "license": "etalab-2.0",
                        "panoramax:horizontal_pixel_density": 42,
                        "pers:interior_orientation": {"sensor_array_dimensions": [4096, 2160]},
                    },
                }
            ],
            "links": [],
        }

    summary = download_panoramax_dataset(
        bbox=(48.80, 2.30, 48.8005, 2.3005),
        out_dir=tmp_path / "street",
        grid_step_m=1000.0,
        street_per_cell=3,
        max_images=10,
        max_per_sequence=10,
        seed=42,
        fetch_json_fn=fake_fetch_json,
        download_bytes_fn=lambda url: b"jpg",
    )
    assert summary["selected_count"] == 1
    rows = list(csv.DictReader((tmp_path / "street" / "metadata.csv").open("r", encoding="utf-8")))
    assert rows[0]["source"] == "panoramax:ign"
    assert (tmp_path / "street" / "images" / "pic-1.jpg").exists()

