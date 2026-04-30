"""Download a realistic Paris street-image dataset from Panoramax."""
from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.tools.download_mapillary_paris import (
    GridCell,
    _request_bytes_with_retry,
    parse_bbox,
    split_bbox_into_cells,
)


PANORAMAX_ENDPOINT = "https://api.panoramax.xyz/api"
USER_AGENT = "Project-Heimdall/1.0 (realistic Paris Panoramax bootstrap)"


@dataclass(frozen=True)
class PanoramaxImage:
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
    image_url: str
    source: str
    license_info: str


def _fetch_json_with_retry(url: str) -> dict:
    payload = json.loads(_request_bytes_with_retry(url, headers={"User-Agent": USER_AGENT}).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("panoramax_response_must_be_object")
    return payload


def _optional_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _asset_href(assets: object, *names: str) -> str:
    if not isinstance(assets, dict):
        return ""
    for name in names:
        candidate = assets.get(name)
        if isinstance(candidate, dict):
            href = candidate.get("href")
            if isinstance(href, str) and href.strip():
                return href.strip()
    return ""


def _license_info(links: object, properties: object) -> str:
    parts: List[str] = []
    if isinstance(properties, dict):
        license_code = str(properties.get("license") or "").strip()
        if license_code:
            parts.append(license_code)
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            if str(link.get("rel") or "").strip() != "license":
                continue
            title = str(link.get("title") or "").strip()
            href = str(link.get("href") or "").strip()
            if title:
                parts.append(title)
            if href:
                parts.append(href)
            break
    return " | ".join(part for part in parts if part)


def _source_name(item: dict) -> str:
    links = item.get("links")
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            if str(link.get("rel") or "").strip() == "via":
                instance_name = str(link.get("instance_name") or "").strip()
                if instance_name:
                    return f"panoramax:{instance_name}"
    return "panoramax:federated"


def parse_panoramax_item(item: object) -> Optional[PanoramaxImage]:
    if not isinstance(item, dict):
        return None
    image_id = str(item.get("id") or "").strip()
    geometry = item.get("geometry") if isinstance(item.get("geometry"), dict) else {}
    coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if not image_id or not isinstance(coords, Sequence) or len(coords) < 2:
        return None
    lon = _optional_float(coords[0])
    lat = _optional_float(coords[1])
    if lon is None or lat is None:
        return None

    properties = item.get("properties") if isinstance(item.get("properties"), dict) else {}
    assets = item.get("assets") if isinstance(item.get("assets"), dict) else {}
    image_url = _asset_href(assets, "sd", "hd", "thumb")
    if not image_url:
        return None

    orientation = properties.get("pers:interior_orientation") if isinstance(properties, dict) else {}
    dims = orientation.get("sensor_array_dimensions") if isinstance(orientation, dict) else None
    width = int(dims[0]) if isinstance(dims, Sequence) and len(dims) >= 2 and _optional_float(dims[0]) else 0
    height = int(dims[1]) if isinstance(dims, Sequence) and len(dims) >= 2 and _optional_float(dims[1]) else 0
    heading = _optional_float(properties.get("view:azimuth")) if isinstance(properties, dict) else None
    quality = _optional_float(properties.get("panoramax:horizontal_pixel_density")) if isinstance(properties, dict) else None

    return PanoramaxImage(
        image_id=image_id,
        lat=float(lat),
        lon=float(lon),
        heading_deg=heading,
        captured_at=str(properties.get("datetime") or properties.get("datetimetz") or ""),
        camera_type="flat",
        width=width,
        height=height,
        quality_score=quality,
        sequence=str(item.get("collection") or ""),
        image_url=image_url,
        source=_source_name(item),
        license_info=_license_info(item.get("links"), properties),
    )


def build_search_url(
    *,
    endpoint: str,
    cell: GridCell,
    limit: int,
) -> str:
    params = {
        "bbox": f"{cell.west:.8f},{cell.south:.8f},{cell.east:.8f},{cell.north:.8f}",
        "limit": str(max(1, int(limit))),
    }
    return str(endpoint).rstrip("/") + "/search?" + urllib.parse.urlencode(params)


def _follow_next_url(payload: dict) -> Optional[str]:
    links = payload.get("links")
    if not isinstance(links, list):
        return None
    for link in links:
        if not isinstance(link, dict):
            continue
        if str(link.get("rel") or "").strip() == "next":
            href = str(link.get("href") or "").strip()
            return href or None
    return None


def fetch_panoramax_cell_images(
    cell: GridCell,
    *,
    endpoint: str,
    limit: int,
    fetch_json_fn: Callable[[str], dict],
) -> List[PanoramaxImage]:
    out: List[PanoramaxImage] = []
    seen_ids: set[str] = set()
    next_url: Optional[str] = build_search_url(endpoint=endpoint, cell=cell, limit=limit)
    while next_url:
        payload = fetch_json_fn(next_url)
        rows = payload.get("features", [])
        if not isinstance(rows, list):
            break
        for row in rows:
            image = parse_panoramax_item(row)
            if image is None or image.image_id in seen_ids:
                continue
            seen_ids.add(image.image_id)
            out.append(image)
        next_url = _follow_next_url(payload)
        if not rows:
            break
        if len(out) >= int(limit):
            break
    return out


def _image_extension(url: str) -> str:
    lower = str(url).lower()
    if ".webp" in lower:
        return ".webp"
    if ".png" in lower:
        return ".png"
    return ".jpg"


def _candidate_sort_key(image: PanoramaxImage) -> Tuple[float, int, str]:
    return (
        image.quality_score if image.quality_score is not None else -1.0,
        1 if image.heading_deg is not None else 0,
        image.image_id,
    )


def _write_metadata_csv(path: Path, rows: Iterable[dict]) -> None:
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


def _load_existing_metadata_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def download_panoramax_dataset(
    *,
    bbox: Tuple[float, float, float, float],
    out_dir: Path,
    grid_step_m: float,
    street_per_cell: int,
    max_images: int,
    max_per_sequence: int,
    seed: int,
    endpoint: str = PANORAMAX_ENDPOINT,
    dry_run: bool = False,
    fetch_json_fn: Optional[Callable[[str], dict]] = None,
    download_bytes_fn: Optional[Callable[[str], bytes]] = None,
) -> dict:
    fetch_json = fetch_json_fn or _fetch_json_with_retry
    download_bytes = download_bytes_fn or (lambda url: _request_bytes_with_retry(url, headers={"User-Agent": USER_AGENT}))
    cells = split_bbox_into_cells(*bbox, grid_step_m=grid_step_m)
    rng = random.Random(int(seed))
    images_dir = out_dir / "images"
    metadata_path = out_dir / "metadata.csv"

    seen_ids: set[str] = set()
    metadata_ids: set[str] = set()
    sequence_counts: Dict[str, int] = {}
    rows: List[dict] = []
    existing_rows = _load_existing_metadata_rows(metadata_path)
    for row in existing_rows:
        image_id = str(row.get("image_id") or "").strip()
        if not image_id:
            continue
        rows.append(row)
        seen_ids.add(image_id)
        metadata_ids.add(image_id)
        sequence = str(row.get("sequence") or "").strip()
        if sequence:
            sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
    existing_file_ids: set[str] = set()
    if images_dir.exists():
        existing_file_ids = {path.stem for path in images_dir.glob("*") if path.is_file()}

    for cell in cells:
        if len(rows) >= int(max_images):
            break
        candidates = fetch_panoramax_cell_images(
            cell,
            endpoint=endpoint,
            limit=max(50, street_per_cell * 20),
            fetch_json_fn=fetch_json,
        )
        pool = list(candidates)
        rng.shuffle(pool)
        pool.sort(key=_candidate_sort_key, reverse=True)
        per_cell_sequences: set[str] = set()
        taken = 0
        for image in pool:
            if image.image_id in metadata_ids:
                continue
            if image.sequence:
                if image.sequence in per_cell_sequences:
                    continue
                if sequence_counts.get(image.sequence, 0) >= int(max_per_sequence):
                    continue
            rel_path = f"images/{image.image_id}{_image_extension(image.image_url)}"
            if not dry_run:
                images_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / rel_path
                if not out_path.exists():
                    out_path.write_bytes(download_bytes(image.image_url))
            rows.append(
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
                    "source": image.source,
                    "license_info": image.license_info,
                }
            )
            seen_ids.add(image.image_id)
            metadata_ids.add(image.image_id)
            if image.sequence:
                sequence_counts[image.sequence] = sequence_counts.get(image.sequence, 0) + 1
                per_cell_sequences.add(image.sequence)
            taken += 1
            if len(rows) >= int(max_images) or taken >= int(street_per_cell):
                break

    if not dry_run and rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_metadata_csv(metadata_path, rows)

    return {
        "bbox": {"south": bbox[0], "west": bbox[1], "north": bbox[2], "east": bbox[3]},
        "endpoint": str(endpoint),
        "cells_total": len(cells),
        "grid_step_m": float(grid_step_m),
        "street_per_cell": int(street_per_cell),
        "max_images": int(max_images),
        "max_per_sequence": int(max_per_sequence),
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "selected_count": len(rows),
        "metadata_path": str(metadata_path),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download a realistic Paris street-image dataset from Panoramax.")
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid-step-m", type=float, default=80.0)
    parser.add_argument("--street-per-cell", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=20000)
    parser.add_argument("--max-per-sequence", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--endpoint", default=PANORAMAX_ENDPOINT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary = download_panoramax_dataset(
        bbox=parse_bbox(args.bbox),
        out_dir=Path(args.out),
        grid_step_m=float(args.grid_step_m),
        street_per_cell=int(args.street_per_cell),
        max_images=int(args.max_images),
        max_per_sequence=int(args.max_per_sequence),
        seed=int(args.seed),
        endpoint=str(args.endpoint),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
