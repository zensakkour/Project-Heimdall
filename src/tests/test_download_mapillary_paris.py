from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.tools.download_mapillary_paris import (
    GridCell,
    _fetch_json_with_retry,
    _load_dotenv_value,
    build_mapillary_images_url,
    download_mapillary_dataset,
    parse_bbox,
    parse_mapillary_image,
    split_bbox_into_cells,
)


def test_parse_bbox_requires_south_west_north_east() -> None:
    assert parse_bbox("48.0,2.0,48.1,2.1") == (48.0, 2.0, 48.1, 2.1)
    with pytest.raises(ValueError):
        parse_bbox("48.0,2.0,48.0,2.1")


def test_split_bbox_into_cells_covers_bbox() -> None:
    cells = split_bbox_into_cells(48.8156, 2.2241, 48.8176, 2.2261, grid_step_m=80.0)
    assert len(cells) > 1
    assert cells[0].south == pytest.approx(48.8156)
    assert cells[-1].north == pytest.approx(48.8176)
    assert cells[-1].east == pytest.approx(2.2261)


def test_parse_mapillary_image_prefers_computed_fields() -> None:
    image = parse_mapillary_image(
        {
            "id": "123",
            "geometry": {"type": "Point", "coordinates": [2.30, 48.80]},
            "computed_geometry": {"type": "Point", "coordinates": [2.31, 48.81]},
            "compass_angle": 90.0,
            "computed_compass_angle": 135.0,
            "captured_at": "2026-01-01T00:00:00Z",
            "camera_type": "perspective",
            "width": 2048,
            "height": 1024,
            "thumb_1024_url": "https://example.com/thumb-1024.jpg",
            "thumb_2048_url": "https://example.com/thumb-2048.jpg",
            "quality_score": 0.88,
            "sequence": "seq-1",
        }
    )
    assert image is not None
    assert image.lat == pytest.approx(48.81)
    assert image.lon == pytest.approx(2.31)
    assert image.heading_deg == pytest.approx(135.0)
    assert image.thumb_url.endswith("thumb-2048.jpg")


def test_build_mapillary_images_url_encodes_bbox_and_fields() -> None:
    url = build_mapillary_images_url(
        GridCell(row=0, col=0, south=48.8, west=2.2, north=48.81, east=2.21),
        token="TOKEN",
        limit=10,
    )
    assert "access_token=TOKEN" in url
    assert "limit=10" in url
    assert "bbox=2.20000000%2C48.80000000%2C2.21000000%2C48.81000000" in url
    assert "computed_geometry" in url


def test_fetch_json_with_retry_retries_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return b'{"data": []}'

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("temporary")
        return _Response()

    sleeps: list[float] = []
    payload = _fetch_json_with_retry(
        "https://example.com",
        retries=3,
        sleep_fn=sleeps.append,
        opener=fake_urlopen,
    )
    assert payload == {"data": []}
    assert calls["count"] == 3
    assert len(sleeps) == 2


def test_load_dotenv_value_reads_key(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("# comment\nMAPILLARY_ACCESS_TOKEN='abc123'\nOTHER=x\n", encoding="utf-8")
    assert _load_dotenv_value("MAPILLARY_ACCESS_TOKEN", env_path=env_path) == "abc123"


def test_download_mapillary_dataset_writes_metadata_and_limits_sequences(tmp_path: Path) -> None:
    def fake_fetch_json(url: str) -> dict:
        return {
            "data": [
                {
                    "id": "img-1",
                    "computed_geometry": {"type": "Point", "coordinates": [2.3001, 48.8001]},
                    "computed_compass_angle": 12.0,
                    "captured_at": "2026-01-01T00:00:00Z",
                    "camera_type": "perspective",
                    "width": 2048,
                    "height": 1024,
                    "thumb_2048_url": "https://example.com/img-1.jpg",
                    "quality_score": 0.95,
                    "sequence": "seq-a",
                },
                {
                    "id": "img-2",
                    "computed_geometry": {"type": "Point", "coordinates": [2.3002, 48.8002]},
                    "computed_compass_angle": 14.0,
                    "captured_at": "2026-01-01T00:01:00Z",
                    "camera_type": "perspective",
                    "width": 2048,
                    "height": 1024,
                    "thumb_2048_url": "https://example.com/img-2.jpg",
                    "quality_score": 0.90,
                    "sequence": "seq-a",
                },
                {
                    "id": "img-3",
                    "computed_geometry": {"type": "Point", "coordinates": [2.3003, 48.8003]},
                    "computed_compass_angle": 16.0,
                    "captured_at": "2026-01-01T00:02:00Z",
                    "camera_type": "perspective",
                    "width": 1024,
                    "height": 768,
                    "thumb_1024_url": "https://example.com/img-3.jpg",
                    "quality_score": 0.80,
                    "sequence": "seq-b",
                },
                {
                    "id": "img-3",
                    "computed_geometry": {"type": "Point", "coordinates": [2.3003, 48.8003]},
                    "computed_compass_angle": 16.0,
                    "captured_at": "2026-01-01T00:02:00Z",
                    "camera_type": "perspective",
                    "width": 1024,
                    "height": 768,
                    "thumb_1024_url": "https://example.com/img-3.jpg",
                    "quality_score": 0.80,
                    "sequence": "seq-b",
                },
            ]
        }

    def fake_download_bytes(url: str) -> bytes:
        return f"downloaded:{url}".encode("utf-8")

    summary = download_mapillary_dataset(
        bbox=(48.80, 2.30, 48.8005, 2.3005),
        out_dir=tmp_path / "street",
        grid_step_m=1000.0,
        street_per_cell=3,
        max_images=10,
        max_per_sequence=10,
        seed=42,
        token="TOKEN",
        fetch_json_fn=fake_fetch_json,
        download_bytes_fn=fake_download_bytes,
    )

    assert summary["selected_count"] == 2
    metadata_path = tmp_path / "street" / "metadata.csv"
    assert metadata_path.exists()
    rows = list(csv.DictReader(metadata_path.open("r", encoding="utf-8")))
    assert [row["image_id"] for row in rows] == ["img-1", "img-3"]
    assert (tmp_path / "street" / "images" / "img-1.jpg").read_bytes().startswith(b"downloaded:")
    assert rows[0]["source"] == "mapillary"


def test_download_mapillary_dataset_dry_run_skips_writes(tmp_path: Path) -> None:
    def fake_fetch_json(url: str) -> dict:
        return {
            "data": [
                {
                    "id": "img-1",
                    "computed_geometry": {"type": "Point", "coordinates": [2.3001, 48.8001]},
                    "computed_compass_angle": 12.0,
                    "captured_at": "2026-01-01T00:00:00Z",
                    "camera_type": "perspective",
                    "width": 2048,
                    "height": 1024,
                    "thumb_2048_url": "https://example.com/img-1.jpg",
                    "quality_score": 0.95,
                    "sequence": "seq-a",
                }
            ]
        }

    summary = download_mapillary_dataset(
        bbox=(48.80, 2.30, 48.8005, 2.3005),
        out_dir=tmp_path / "street",
        grid_step_m=1000.0,
        street_per_cell=3,
        max_images=10,
        max_per_sequence=10,
        seed=42,
        dry_run=True,
        token="TOKEN",
        fetch_json_fn=fake_fetch_json,
        download_bytes_fn=lambda url: b"x",
    )

    assert summary["selected_count"] == 1
    assert not (tmp_path / "street" / "metadata.csv").exists()
    assert not (tmp_path / "street" / "images").exists()


def test_download_mapillary_dataset_reads_token_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("MAPILLARY_ACCESS_TOKEN=dotenv-token\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def fake_fetch_json(url: str) -> dict:
        assert "access_token=dotenv-token" in url
        return {
            "data": [
                {
                    "id": "img-1",
                    "computed_geometry": {"type": "Point", "coordinates": [2.3001, 48.8001]},
                    "thumb_2048_url": "https://example.com/img-1.jpg",
                }
            ]
        }

    summary = download_mapillary_dataset(
        bbox=(48.80, 2.30, 48.8005, 2.3005),
        out_dir=tmp_path / "street",
        grid_step_m=1000.0,
        street_per_cell=1,
        max_images=1,
        max_per_sequence=10,
        seed=42,
        dry_run=True,
        fetch_json_fn=fake_fetch_json,
        download_bytes_fn=lambda url: b"x",
    )
    assert summary["selected_count"] == 1


def test_download_mapillary_dataset_skips_failed_downloads(tmp_path: Path) -> None:
    def fake_fetch_json(url: str) -> dict:
        return {
            "data": [
                {
                    "id": "img-1",
                    "computed_geometry": {"type": "Point", "coordinates": [2.3001, 48.8001]},
                    "thumb_2048_url": "https://example.com/img-1.jpg",
                },
                {
                    "id": "img-2",
                    "computed_geometry": {"type": "Point", "coordinates": [2.3002, 48.8002]},
                    "thumb_2048_url": "https://example.com/img-2.jpg",
                },
            ]
        }

    def fake_download_bytes(url: str) -> bytes:
        if url.endswith("img-1.jpg"):
            raise RuntimeError("temporary")
        return b"ok"

    summary = download_mapillary_dataset(
        bbox=(48.80, 2.30, 48.8005, 2.3005),
        out_dir=tmp_path / "street",
        grid_step_m=1000.0,
        street_per_cell=3,
        max_images=10,
        max_per_sequence=10,
        seed=42,
        token="TOKEN",
        fetch_json_fn=fake_fetch_json,
        download_bytes_fn=fake_download_bytes,
    )

    assert summary["selected_count"] == 1
    assert summary["failed_downloads"] == 1
