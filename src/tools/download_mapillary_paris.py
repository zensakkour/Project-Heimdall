"""Download a realistic Paris street-image dataset from Mapillary."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


MAPILLARY_FIELDS: Tuple[str, ...] = (
    "id",
    "geometry",
    "computed_geometry",
    "compass_angle",
    "computed_compass_angle",
    "captured_at",
    "camera_type",
    "width",
    "height",
    "thumb_1024_url",
    "thumb_2048_url",
    "quality_score",
    "sequence",
)
MAPILLARY_IMAGES_ENDPOINT = "https://graph.mapillary.com/images"
MAX_MAPILLARY_LIMIT = 2000
USER_AGENT = "Project-Heimdall/1.0 (realistic Paris dataset bootstrap)"
LICENSE_INFO = "Mapillary platform imagery; verify current terms and attribution requirements separately."


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    south: float
    west: float
    north: float
    east: float


@dataclass(frozen=True)
class MapillaryImage:
    image_id: str
    lat: float
    lon: float
    heading_deg: Optional[float]
    captured_at: str
    camera_type: str
    width: int
    height: int
    quality_score: Optional[float]
    sequence: str
    thumb_url: str


def parse_bbox(raw: str) -> Tuple[float, float, float, float]:
    parts = [item.strip() for item in str(raw).split(",")]
    if len(parts) != 4:
        raise ValueError("bbox_must_be_south,west,north,east")
    south, west, north, east = (float(item) for item in parts)
    if north <= south:
        raise ValueError("bbox_north_must_be_gt_south")
    if east <= west:
        raise ValueError("bbox_east_must_be_gt_west")
    return south, west, north, east


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


def split_bbox_into_cells(
    south: float,
    west: float,
    north: float,
    east: float,
    *,
    grid_step_m: float,
) -> List[GridCell]:
    if float(grid_step_m) <= 0.0:
        raise ValueError("grid_step_m_must_be_positive")

    avg_lat = (float(south) + float(north)) / 2.0
    lat_step_deg = float(grid_step_m) / max(_meters_per_degree_lat(avg_lat), 1.0)
    lon_step_deg = float(grid_step_m) / max(_meters_per_degree_lon(avg_lat), 1.0)
    lat_step_deg = max(lat_step_deg, 1e-6)
    lon_step_deg = max(lon_step_deg, 1e-6)

    cells: List[GridCell] = []
    row = 0
    current_south = float(south)
    while current_south < float(north) - 1e-12:
        current_north = min(float(north), current_south + lat_step_deg)
        col = 0
        current_west = float(west)
        while current_west < float(east) - 1e-12:
            current_east = min(float(east), current_west + lon_step_deg)
            cells.append(
                GridCell(
                    row=row,
                    col=col,
                    south=current_south,
                    west=current_west,
                    north=current_north,
                    east=current_east,
                )
            )
            current_west = current_east
            col += 1
        current_south = current_north
        row += 1
    return cells


def _request_bytes_with_retry(
    url: str,
    *,
    retries: int = 3,
    timeout_s: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> bytes:
    last_error: Optional[Exception] = None
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)

    for attempt in range(max(1, int(retries))):
        req = urllib.request.Request(url, headers=request_headers)
        try:
            with opener(req, timeout=timeout_s) as response:  # nosec - read-only public API
                return bytes(response.read())
        except Exception as exc:  # pragma: no cover - exact transport error type is not important
            last_error = exc
            if attempt >= max(1, int(retries)) - 1:
                break
            sleep_fn(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"request_failed:{url}") from last_error


def _load_dotenv_value(key: str, *, env_path: Path = Path(".env")) -> str:
    if not env_path.exists():
        return ""
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    prefix = f"{str(key).strip()}="
    for raw_line in lines:
        line = str(raw_line).strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value.strip()
    return ""


def _fetch_json_with_retry(
    url: str,
    *,
    retries: int = 3,
    timeout_s: float = 30.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> dict:
    raw = _request_bytes_with_retry(
        url,
        retries=retries,
        timeout_s=timeout_s,
        sleep_fn=sleep_fn,
        opener=opener,
    )
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("mapillary_response_must_be_object")
    return payload


def _parse_point_geometry(raw: object) -> Optional[Tuple[float, float]]:
    if not isinstance(raw, dict):
        return None
    coords = raw.get("coordinates")
    if not isinstance(coords, Sequence) or len(coords) < 2:
        return None
    lon = coords[0]
    lat = coords[1]
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    return float(lat), float(lon)


def _optional_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _optional_int(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _sequence_id(raw: object) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        seq_id = raw.get("id")
        if isinstance(seq_id, str):
            return seq_id
    return ""


def parse_mapillary_image(raw: object) -> Optional[MapillaryImage]:
    if not isinstance(raw, dict):
        return None

    image_id = str(raw.get("id") or "").strip()
    if not image_id:
        return None

    location = _parse_point_geometry(raw.get("computed_geometry"))
    if location is None:
        location = _parse_point_geometry(raw.get("geometry"))
    if location is None:
        return None
    lat, lon = location

    heading = _optional_float(raw.get("computed_compass_angle"))
    if heading is None:
        heading = _optional_float(raw.get("compass_angle"))

    thumb_url = str(raw.get("thumb_2048_url") or raw.get("thumb_1024_url") or "").strip()
    if not thumb_url:
        return None

    return MapillaryImage(
        image_id=image_id,
        lat=lat,
        lon=lon,
        heading_deg=heading,
        captured_at=str(raw.get("captured_at") or ""),
        camera_type=str(raw.get("camera_type") or ""),
        width=_optional_int(raw.get("width")),
        height=_optional_int(raw.get("height")),
        quality_score=_optional_float(raw.get("quality_score")),
        sequence=_sequence_id(raw.get("sequence")),
        thumb_url=thumb_url,
    )


def build_mapillary_images_url(
    cell: GridCell,
    *,
    token: str,
    fields: Sequence[str] = MAPILLARY_FIELDS,
    limit: int = MAX_MAPILLARY_LIMIT,
) -> str:
    params = {
        "access_token": str(token),
        "bbox": f"{cell.west:.8f},{cell.south:.8f},{cell.east:.8f},{cell.north:.8f}",
        "fields": ",".join(str(item) for item in fields),
        "limit": str(max(1, min(MAX_MAPILLARY_LIMIT, int(limit)))),
    }
    return MAPILLARY_IMAGES_ENDPOINT + "?" + urllib.parse.urlencode(params)


def fetch_mapillary_cell_images(
    cell: GridCell,
    *,
    token: str,
    limit: int = MAX_MAPILLARY_LIMIT,
    fetch_json_fn: Callable[[str], dict],
) -> List[MapillaryImage]:
    payload = fetch_json_fn(build_mapillary_images_url(cell, token=token, limit=limit))
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        return []
    out: List[MapillaryImage] = []
    for row in rows:
        image = parse_mapillary_image(row)
        if image is not None:
            out.append(image)
    return out


def _candidate_sort_key(image: MapillaryImage) -> Tuple[float, int, int, str]:
    return (
        image.quality_score if image.quality_score is not None else -1.0,
        1 if "2048" in image.thumb_url else 0,
        1 if image.heading_deg is not None else 0,
        image.image_id,
    )


def select_cell_images(
    candidates: Sequence[MapillaryImage],
    *,
    street_per_cell: int,
    max_per_sequence: int,
    seen_image_ids: Optional[set[str]] = None,
    global_sequence_counts: Optional[Dict[str, int]] = None,
    rng: Optional[random.Random] = None,
) -> List[MapillaryImage]:
    if street_per_cell <= 0:
        return []
    seen_ids = seen_image_ids if seen_image_ids is not None else set()
    sequence_counts = global_sequence_counts if global_sequence_counts is not None else {}

    pool = list(candidates)
    if rng is not None:
        rng.shuffle(pool)
    pool.sort(key=_candidate_sort_key, reverse=True)

    selected: List[MapillaryImage] = []
    per_cell_sequence_counts: Dict[str, int] = {}
    for image in pool:
        if image.image_id in seen_ids:
            continue
        if image.sequence:
            if per_cell_sequence_counts.get(image.sequence, 0) >= 1:
                continue
            if sequence_counts.get(image.sequence, 0) >= max(1, int(max_per_sequence)):
                continue
        selected.append(image)
        if image.sequence:
            per_cell_sequence_counts[image.sequence] = per_cell_sequence_counts.get(image.sequence, 0) + 1
        if len(selected) >= int(street_per_cell):
            break
    return selected


def _image_extension_from_url(url: str) -> str:
    lower = str(url).lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if ext in lower:
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def _write_metadata_csv(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def download_mapillary_dataset(
    *,
    bbox: Tuple[float, float, float, float],
    out_dir: Path,
    grid_step_m: float,
    street_per_cell: int,
    max_images: int,
    seed: int,
    max_per_sequence: int = 12,
    dry_run: bool = False,
    token: Optional[str] = None,
    fetch_json_fn: Optional[Callable[[str], dict]] = None,
    download_bytes_fn: Optional[Callable[[str], bytes]] = None,
) -> dict:
    access_token = str(token or os.environ.get("MAPILLARY_ACCESS_TOKEN") or _load_dotenv_value("MAPILLARY_ACCESS_TOKEN") or "").strip()
    if not access_token:
        raise ValueError("MAPILLARY_ACCESS_TOKEN_missing")

    south, west, north, east = bbox
    cells = split_bbox_into_cells(south, west, north, east, grid_step_m=grid_step_m)
    rng = random.Random(int(seed))

    if fetch_json_fn is None:
        fetch_json_fn = lambda url: _fetch_json_with_retry(url)
    if download_bytes_fn is None:
        download_bytes_fn = lambda url: _request_bytes_with_retry(url)

    selected_rows: List[dict] = []
    seen_image_ids: set[str] = set()
    sequence_counts: Dict[str, int] = {}
    images_dir = out_dir / "images"
    failed_cells = 0
    failed_downloads = 0

    for cell in cells:
        if len(selected_rows) >= int(max_images):
            break
        try:
            candidates = fetch_mapillary_cell_images(
                cell,
                token=access_token,
                limit=MAX_MAPILLARY_LIMIT,
                fetch_json_fn=fetch_json_fn,
            )
        except Exception:
            failed_cells += 1
            continue
        picked = select_cell_images(
            candidates,
            street_per_cell=street_per_cell,
            max_per_sequence=max_per_sequence,
            seen_image_ids=seen_image_ids,
            global_sequence_counts=sequence_counts,
            rng=rng,
        )
        for image in picked:
            if len(selected_rows) >= int(max_images):
                break
            rel_path = f"images/{image.image_id}{_image_extension_from_url(image.thumb_url)}"
            if not dry_run:
                images_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / rel_path
                if not out_path.exists():
                    try:
                        out_path.write_bytes(download_bytes_fn(image.thumb_url))
                    except Exception:
                        failed_downloads += 1
                        continue
            selected_rows.append(
                {
                    "image_id": image.image_id,
                    "path": rel_path,
                    "lat": f"{image.lat:.8f}",
                    "lon": f"{image.lon:.8f}",
                    "heading_deg": "" if image.heading_deg is None else f"{image.heading_deg:.6f}",
                    "captured_at": image.captured_at,
                    "camera_type": image.camera_type,
                    "width": str(image.width),
                    "height": str(image.height),
                    "quality_score": "" if image.quality_score is None else f"{image.quality_score:.6f}",
                    "sequence": image.sequence,
                    "source": "mapillary",
                    "license_info": LICENSE_INFO,
                }
            )
            seen_image_ids.add(image.image_id)
            if image.sequence:
                sequence_counts[image.sequence] = sequence_counts.get(image.sequence, 0) + 1

    metadata_path = out_dir / "metadata.csv"
    if not dry_run and selected_rows:
        _write_metadata_csv(metadata_path, selected_rows)

    return {
        "bbox": {
            "south": float(south),
            "west": float(west),
            "north": float(north),
            "east": float(east),
        },
        "cells_total": len(cells),
        "cells_touched": len(cells) if len(selected_rows) < int(max_images) else None,
        "grid_step_m": float(grid_step_m),
        "street_per_cell": int(street_per_cell),
        "max_per_sequence": int(max_per_sequence),
        "max_images": int(max_images),
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "selected_count": len(selected_rows),
        "failed_cells": int(failed_cells),
        "failed_downloads": int(failed_downloads),
        "metadata_path": str(metadata_path),
        "images_dir": str(images_dir),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download a realistic Paris street-image dataset from Mapillary.")
    parser.add_argument("--bbox", required=True, help="Bounding box in south,west,north,east order.")
    parser.add_argument("--out", required=True, help="Output directory for street dataset files.")
    parser.add_argument("--grid-step-m", type=float, default=80.0)
    parser.add_argument("--street-per-cell", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=20000)
    parser.add_argument("--max-per-sequence", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary = download_mapillary_dataset(
        bbox=parse_bbox(args.bbox),
        out_dir=Path(args.out),
        grid_step_m=float(args.grid_step_m),
        street_per_cell=int(args.street_per_cell),
        max_images=int(args.max_images),
        max_per_sequence=int(args.max_per_sequence),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
    )

    if bool(args.dry_run):
        print(json.dumps(summary, indent=2))
        print(f"Dry run selected {summary['selected_count']} images.")
    else:
        print(json.dumps(summary, indent=2))
        print(f"Wrote {summary['selected_count']} images -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
