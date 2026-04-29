from __future__ import annotations

import csv
import io
from pathlib import Path

from PIL import Image

from src.tools.build_aerial_pairs import IgnGeopfProvider, OpenAerialMapProvider, _parse_oam_scene, _render_tms_crop, _render_wms_crop, build_aerial_pairs_dataset


def test_parse_oam_scene_extracts_resolution_tms_and_license() -> None:
    scene = _parse_oam_scene(
        {
            "_id": "scene-1",
            "title": "Example",
            "bbox": [2.20, 48.80, 2.40, 48.90],
            "gsd": 0.25,
            "properties": {
                "license": "CC-BY 4.0",
                "tms": "https://tiles.example.com/{z}/{x}/{y}.png",
                "resolution_in_meters": 0.2,
            },
        },
        lat=48.85,
        lon=2.30,
    )
    assert scene is not None
    assert scene.source_id == "scene-1"
    assert scene.resolution_m == 0.2
    assert scene.license_info == "CC-BY 4.0"


def test_openaerialmap_provider_picks_highest_resolution_scene() -> None:
    def fake_fetch_json(url: str) -> dict:
        return {
            "results": [
                {
                    "_id": "scene-bad",
                    "bbox": [2.20, 48.80, 2.40, 48.90],
                    "properties": {
                        "license": "CC-BY 4.0",
                        "tms": "https://tiles.example.com/bad/{z}/{x}/{y}.png",
                        "resolution_in_meters": 1.2,
                    },
                },
                {
                    "_id": "scene-good",
                    "bbox": [2.20, 48.80, 2.40, 48.90],
                    "properties": {
                        "license": "CC-BY 4.0",
                        "tms": "https://tiles.example.com/good/{z}/{x}/{y}.png",
                        "resolution_in_meters": 0.3,
                    },
                },
            ]
        }

    provider = OpenAerialMapProvider(fetch_json_fn=fake_fetch_json)
    scene = provider.best_scene_for_point(48.85, 2.30)
    assert scene is not None
    assert scene.source_id == "scene-good"


def test_render_tms_crop_builds_centered_crop() -> None:
    tile = Image.new("RGB", (256, 256), (10, 20, 30))
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    crop = _render_tms_crop(
        tms_url="https://tiles.example.com/{z}/{x}/{y}.png",
        lat=48.85,
        lon=2.30,
        crop_px=128,
        zoom=18,
        download_bytes_fn=lambda url: png_bytes,
    )
    assert crop.size == (128, 128)
    assert crop.getpixel((64, 64)) == (10, 20, 30)


def test_build_aerial_pairs_dataset_writes_rows_and_pairs(tmp_path: Path) -> None:
    street_dir = tmp_path / "street"
    street_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = street_dir / "metadata.csv"
    metadata_path.write_text(
        "image_id,path,lat,lon,heading_deg,captured_at,camera_type,width,height,quality_score,sequence,source,license_info\n"
        "street-1,images/street-1.jpg,48.85000000,2.30000000,90.0,,,,,,,mapillary,\n",
        encoding="utf-8",
    )

    tile = Image.new("RGB", (256, 256), (100, 120, 140))
    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    def fake_fetch_json(url: str) -> dict:
        return {
            "results": [
                {
                    "_id": "scene-1",
                    "title": "Example",
                    "bbox": [2.20, 48.80, 2.40, 48.90],
                    "properties": {
                        "license": "CC-BY 4.0",
                        "tms": "https://tiles.example.com/{z}/{x}/{y}.png",
                        "resolution_in_meters": 0.25,
                    },
                }
            ]
        }

    summary = build_aerial_pairs_dataset(
        street_metadata=metadata_path,
        out_dir=tmp_path / "dataset",
        provider_name="openaerialmap",
        crop_size_m=256.0,
        crop_px=128,
        allow_missing_aerial=False,
        seed=42,
        fetch_json_fn=fake_fetch_json,
        download_bytes_fn=lambda url: png_bytes,
    )

    assert summary["pairs_written"] == 1
    aerial_rows = list(csv.DictReader((tmp_path / "dataset" / "aerial" / "metadata.csv").open("r", encoding="utf-8")))
    assert aerial_rows[0]["status"] == "ok"
    pair_rows = list(csv.DictReader((tmp_path / "dataset" / "pairs.csv").open("r", encoding="utf-8")))
    assert pair_rows[0]["street_id"] == "street-1"
    crop_path = tmp_path / "dataset" / "aerial" / "images" / f"{aerial_rows[0]['aerial_id']}.png"
    assert crop_path.exists()


def test_build_aerial_pairs_dataset_marks_missing_when_no_scene(tmp_path: Path) -> None:
    street_dir = tmp_path / "street"
    street_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = street_dir / "metadata.csv"
    metadata_path.write_text(
        "image_id,path,lat,lon,heading_deg,captured_at,camera_type,width,height,quality_score,sequence,source,license_info\n"
        "street-1,images/street-1.jpg,48.85000000,2.30000000,90.0,,,,,,,mapillary,\n",
        encoding="utf-8",
    )

    summary = build_aerial_pairs_dataset(
        street_metadata=metadata_path,
        out_dir=tmp_path / "dataset",
        provider_name="openaerialmap",
        crop_size_m=256.0,
        crop_px=128,
        allow_missing_aerial=False,
        seed=42,
        fetch_json_fn=lambda url: {"results": []},
        download_bytes_fn=lambda url: b"",
    )

    assert summary["pairs_written"] == 0
    aerial_rows = list(csv.DictReader((tmp_path / "dataset" / "aerial" / "metadata.csv").open("r", encoding="utf-8")))
    assert aerial_rows[0]["status"] == "no_open_aerial_found"


def test_ign_geopf_provider_returns_scene() -> None:
    scene = IgnGeopfProvider().best_scene_for_point(48.85, 2.30)
    assert scene is not None
    assert scene.provider == "ign_geopf"
    assert "ORTHOIMAGERY.ORTHOPHOTOS" in scene.title
    assert scene.wms_url == "https://data.geopf.fr/wms-r"


def test_render_wms_crop_builds_request() -> None:
    tile = Image.new("RGB", (128, 128), (1, 2, 3))
    buf = io.BytesIO()
    tile.save(buf, format="JPEG")
    jpg_bytes = buf.getvalue()
    seen = {"url": ""}

    def fake_download(url: str) -> bytes:
        seen["url"] = url
        return jpg_bytes

    crop = _render_wms_crop(
        wms_url="https://data.geopf.fr/wms-r",
        lat=48.85,
        lon=2.30,
        crop_size_m=256.0,
        crop_px=128,
        download_bytes_fn=fake_download,
    )
    assert crop.size == (128, 128)
    assert "REQUEST=GetMap" in seen["url"]
    assert "CRS=CRS%3A84" in seen["url"]
