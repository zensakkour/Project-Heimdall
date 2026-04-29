"""Build aerial crops and positive pairs for the realistic Paris street dataset."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


OAM_META_ENDPOINT = "https://api.openaerialmap.org/meta"
USER_AGENT = "Project-Heimdall/1.0 (realistic Paris aerial pairing)"
IGN_GEOPF_ORTHO_URL = (
    "https://data.geopf.fr/wmts?"
    "SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&"
    "FORMAT=image/jpeg&TILEMATRIXSET=PM_0_19&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}"
)


@dataclass(frozen=True)
class StreetRecord:
    image_id: str
    path: str
    lat: float
    lon: float
    heading_deg: Optional[float]


@dataclass(frozen=True)
class AerialScene:
    source_id: str
    provider: str
    title: str
    bbox: Tuple[float, float, float, float]
    resolution_m: float
    license_info: str
    tms_url: str


class AerialProvider:
    def best_scene_for_point(self, lat: float, lon: float) -> Optional[AerialScene]:
        raise NotImplementedError


def _request_bytes_with_retry(
    url: str,
    *,
    retries: int = 3,
    timeout_s: float = 30.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(max(1, int(retries))):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with opener(req, timeout=timeout_s) as response:  # nosec - read-only public API
                return bytes(response.read())
        except Exception as exc:  # pragma: no cover - exact transport error type is not important
            last_error = exc
            if attempt >= max(1, int(retries)) - 1:
                break
            sleep_fn(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"request_failed:{url}") from last_error


def _fetch_json_with_retry(
    url: str,
    *,
    retries: int = 3,
    timeout_s: float = 30.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> dict:
    payload = json.loads(
        _request_bytes_with_retry(
            url,
            retries=retries,
            timeout_s=timeout_s,
            sleep_fn=sleep_fn,
            opener=opener,
        ).decode("utf-8")
    )
    if not isinstance(payload, dict):
        raise ValueError("oam_response_must_be_object")
    return payload


def _optional_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _optional_heading(value: object) -> Optional[float]:
    number = _optional_float(value)
    if number is None:
        return None
    return float(number)


def _load_street_metadata(path: Path) -> List[StreetRecord]:
    rows: List[StreetRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = str(row.get("image_id") or "").strip()
            rel_path = str(row.get("path") or "").strip()
            lat = _optional_float(row.get("lat") or row.get("latitude"))
            lon = _optional_float(row.get("lon") or row.get("longitude"))
            if not image_id or not rel_path or lat is None or lon is None:
                continue
            rows.append(
                StreetRecord(
                    image_id=image_id,
                    path=rel_path,
                    lat=float(lat),
                    lon=float(lon),
                    heading_deg=_optional_heading(row.get("heading_deg")),
                )
            )
    return rows


def _point_bbox(lon: float, lat: float, epsilon_deg: float = 0.0006) -> Tuple[float, float, float, float]:
    eps = max(float(epsilon_deg), 1e-6)
    return lon - eps, lat - eps, lon + eps, lat + eps


def _point_in_bbox(lat: float, lon: float, bbox: Sequence[object]) -> bool:
    if len(bbox) < 4:
        return False
    west = _optional_float(bbox[0])
    south = _optional_float(bbox[1])
    east = _optional_float(bbox[2])
    north = _optional_float(bbox[3])
    if west is None or south is None or east is None or north is None:
        return False
    return bool(west <= lon <= east and south <= lat <= north)


def _scene_resolution_m(row: dict) -> Optional[float]:
    properties = row.get("properties") if isinstance(row.get("properties"), dict) else {}
    if isinstance(properties, dict):
        for key in ("resolution_in_meters",):
            value = _optional_float(properties.get(key))
            if value is not None:
                return value
        value = properties.get("resolution")
        if isinstance(value, Sequence) and value:
            first = _optional_float(value[0])
            second = _optional_float(value[1]) if len(value) > 1 else first
            values = [item for item in (first, second) if item is not None]
            if values:
                return float(max(values))
    return _optional_float(row.get("gsd"))


def _parse_oam_scene(row: object, *, lat: float, lon: float) -> Optional[AerialScene]:
    if not isinstance(row, dict):
        return None
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or not _point_in_bbox(lat, lon, bbox):
        return None

    properties = row.get("properties") if isinstance(row.get("properties"), dict) else {}
    tms_url = str(properties.get("tms") or "").strip() if isinstance(properties, dict) else ""
    if not tms_url:
        return None

    resolution_m = _scene_resolution_m(row)
    if resolution_m is None:
        return None

    license_info = ""
    if isinstance(properties, dict):
        license_info = str(properties.get("license") or "").strip()

    source_id = str(row.get("_id") or row.get("uuid") or "").strip()
    if not source_id:
        return None

    west = float(bbox[0])
    south = float(bbox[1])
    east = float(bbox[2])
    north = float(bbox[3])
    return AerialScene(
        source_id=source_id,
        provider="openaerialmap",
        title=str(row.get("title") or ""),
        bbox=(west, south, east, north),
        resolution_m=float(resolution_m),
        license_info=license_info or "CC-BY 4.0",
        tms_url=tms_url,
    )


class OpenAerialMapProvider(AerialProvider):
    def __init__(self, *, fetch_json_fn: Optional[Callable[[str], dict]] = None) -> None:
        self._fetch_json = fetch_json_fn or (lambda url: _fetch_json_with_retry(url))

    def best_scene_for_point(self, lat: float, lon: float) -> Optional[AerialScene]:
        west, south, east, north = _point_bbox(lon, lat)
        params = {
            "bbox": f"{west:.8f},{south:.8f},{east:.8f},{north:.8f}",
            "limit": "50",
        }
        url = OAM_META_ENDPOINT + "?" + urllib.parse.urlencode(params)
        payload = self._fetch_json(url)
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            return None
        scenes = [
            scene
            for scene in (_parse_oam_scene(row, lat=lat, lon=lon) for row in rows)
            if scene is not None
        ]
        if not scenes:
            return None
        scenes.sort(key=lambda item: (item.resolution_m, item.source_id))
        return scenes[0]


class IgnGeopfProvider(AerialProvider):
    def best_scene_for_point(self, lat: float, lon: float) -> Optional[AerialScene]:
        return AerialScene(
            source_id="ign_geopf_ortho",
            provider="ign_geopf",
            title="IGN ORTHOIMAGERY.ORTHOPHOTOS",
            bbox=(-180.0, -85.0, 180.0, 85.0),
            resolution_m=0.20,
            license_info="IGN / Geoportail orthophotos; verify attribution for final publication.",
            tms_url=IGN_GEOPF_ORTHO_URL,
        )


def make_provider(
    provider_name: str,
    *,
    fetch_json_fn: Optional[Callable[[str], dict]] = None,
) -> AerialProvider:
    normalized = str(provider_name).strip().lower()
    if normalized == "openaerialmap":
        return OpenAerialMapProvider(fetch_json_fn=fetch_json_fn)
    if normalized == "ign_geopf":
        return IgnGeopfProvider()
    raise ValueError(f"unsupported_provider:{provider_name}")


def _target_zoom(lat: float, crop_size_m: float, crop_px: int) -> int:
    target_m_per_px = float(crop_size_m) / max(1.0, float(crop_px))
    target_m_per_px = max(target_m_per_px, 0.05)
    initial_resolution = 156543.03392804097 * math.cos(math.radians(float(lat)))
    zoom = math.ceil(math.log2(max(initial_resolution / target_m_per_px, 1.0)))
    return int(min(22, max(0, zoom)))


def _lonlat_to_world_pixel(lon: float, lat: float, zoom: int) -> Tuple[float, float]:
    lat_clamped = max(min(float(lat), 85.05112878), -85.05112878)
    scale = 256.0 * (2**int(zoom))
    x = (float(lon) + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(lat_clamped))
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * scale
    return x, y


def _render_tms_crop(
    *,
    tms_url: str,
    lat: float,
    lon: float,
    crop_px: int,
    zoom: int,
    download_bytes_fn: Callable[[str], bytes],
) -> Image.Image:
    center_x, center_y = _lonlat_to_world_pixel(lon, lat, zoom)
    half = float(crop_px) / 2.0
    left = center_x - half
    top = center_y - half
    right = center_x + half
    bottom = center_y + half

    tile_left = int(math.floor(left / 256.0))
    tile_top = int(math.floor(top / 256.0))
    tile_right = int(math.floor((right - 1.0) / 256.0))
    tile_bottom = int(math.floor((bottom - 1.0) / 256.0))
    max_tile = (2**int(zoom)) - 1

    canvas = Image.new("RGB", ((tile_right - tile_left + 1) * 256, (tile_bottom - tile_top + 1) * 256))
    for tile_x in range(tile_left, tile_right + 1):
        wrapped_x = tile_x % (2**int(zoom))
        for tile_y in range(tile_top, tile_bottom + 1):
            clamped_y = min(max(tile_y, 0), max_tile)
            tile_url = (
                str(tms_url)
                .replace("{z}", str(int(zoom)))
                .replace("{x}", str(int(wrapped_x)))
                .replace("{y}", str(int(clamped_y)))
            )
            tile_img = Image.open(io.BytesIO(download_bytes_fn(tile_url))).convert("RGB")
            canvas.paste(tile_img, ((tile_x - tile_left) * 256, (tile_y - tile_top) * 256))

    crop_left = int(round(left - (tile_left * 256.0)))
    crop_top = int(round(top - (tile_top * 256.0)))
    crop_right = crop_left + int(crop_px)
    crop_bottom = crop_top + int(crop_px)
    return canvas.crop((crop_left, crop_top, crop_right, crop_bottom))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_bool(text: str) -> bool:
    normalized = str(text).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid_bool:{text}")


def build_aerial_pairs_dataset(
    *,
    street_metadata: Path,
    out_dir: Path,
    provider_name: str,
    crop_size_m: float,
    crop_px: int,
    allow_missing_aerial: bool,
    seed: int,
    fetch_json_fn: Optional[Callable[[str], dict]] = None,
    download_bytes_fn: Optional[Callable[[str], bytes]] = None,
) -> dict:
    _ = seed  # Reserved for future deterministic provider sampling.
    provider = make_provider(provider_name, fetch_json_fn=fetch_json_fn)
    downloader = download_bytes_fn or (lambda url: _request_bytes_with_retry(url))

    street_rows = _load_street_metadata(street_metadata)
    aerial_dir = out_dir / "aerial" / "images"
    aerial_metadata_path = out_dir / "aerial" / "metadata.csv"
    pairs_path = out_dir / "pairs.csv"

    aerial_rows: List[dict] = []
    pair_rows: List[dict] = []
    missing_count = 0

    for street in street_rows:
        scene = provider.best_scene_for_point(street.lat, street.lon)
        if scene is None:
            missing_count += 1
            aerial_rows.append(
                {
                    "aerial_id": f"missing_{street.image_id}",
                    "path": "",
                    "lat": f"{street.lat:.8f}",
                    "lon": f"{street.lon:.8f}",
                    "source": "",
                    "provider": str(provider_name),
                    "resolution_m": "",
                    "crop_size_m": f"{float(crop_size_m):.2f}",
                    "crop_px": str(int(crop_px)),
                    "license_info": "",
                    "paired_street_id": street.image_id,
                    "status": "no_open_aerial_found",
                }
            )
            if not allow_missing_aerial:
                continue
            pair_rows.append(
                {
                    "pair_id": f"{street.image_id}__missing",
                    "street_id": street.image_id,
                    "street_path": street.path,
                    "aerial_id": "",
                    "aerial_path": "",
                    "lat": f"{street.lat:.8f}",
                    "lon": f"{street.lon:.8f}",
                    "heading_deg": "" if street.heading_deg is None else f"{street.heading_deg:.6f}",
                }
            )
            continue

        zoom = _target_zoom(street.lat, crop_size_m, crop_px)
        image = _render_tms_crop(
            tms_url=scene.tms_url,
            lat=street.lat,
            lon=street.lon,
            crop_px=int(crop_px),
            zoom=zoom,
            download_bytes_fn=downloader,
        )
        aerial_id = f"{scene.source_id}_{street.image_id}"
        rel_path = f"aerial/images/{aerial_id}.png"
        out_path = out_dir / rel_path
        aerial_dir.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="PNG")

        aerial_rows.append(
            {
                "aerial_id": aerial_id,
                "path": rel_path,
                "lat": f"{street.lat:.8f}",
                "lon": f"{street.lon:.8f}",
                "source": scene.source_id,
                "provider": scene.provider,
                "resolution_m": f"{scene.resolution_m:.6f}",
                "crop_size_m": f"{float(crop_size_m):.2f}",
                "crop_px": str(int(crop_px)),
                "license_info": scene.license_info,
                "paired_street_id": street.image_id,
                "status": "ok",
            }
        )
        pair_rows.append(
            {
                "pair_id": f"{street.image_id}__{aerial_id}",
                "street_id": street.image_id,
                "street_path": street.path,
                "aerial_id": aerial_id,
                "aerial_path": rel_path,
                "lat": f"{street.lat:.8f}",
                "lon": f"{street.lon:.8f}",
                "heading_deg": "" if street.heading_deg is None else f"{street.heading_deg:.6f}",
            }
        )

    _write_csv(
        aerial_metadata_path,
        [
            "aerial_id",
            "path",
            "lat",
            "lon",
            "source",
            "provider",
            "resolution_m",
            "crop_size_m",
            "crop_px",
            "license_info",
            "paired_street_id",
            "status",
        ],
        aerial_rows,
    )
    _write_csv(
        pairs_path,
        ["pair_id", "street_id", "street_path", "aerial_id", "aerial_path", "lat", "lon", "heading_deg"],
        pair_rows,
    )

    return {
        "street_metadata": str(street_metadata),
        "provider": str(provider_name),
        "crop_size_m": float(crop_size_m),
        "crop_px": int(crop_px),
        "allow_missing_aerial": bool(allow_missing_aerial),
        "street_records": len(street_rows),
        "aerial_rows": len(aerial_rows),
        "pairs_written": len(pair_rows),
        "missing_aerial_count": int(missing_count),
        "aerial_metadata": str(aerial_metadata_path),
        "pairs_csv": str(pairs_path),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build aerial crops and street-to-aerial positive pairs.")
    parser.add_argument("--street-metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--provider", default="openaerialmap")
    parser.add_argument("--crop-size-m", type=float, default=256.0)
    parser.add_argument("--crop-px", type=int, default=512)
    parser.add_argument("--allow-missing-aerial", default="false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    summary = build_aerial_pairs_dataset(
        street_metadata=Path(args.street_metadata),
        out_dir=Path(args.out),
        provider_name=args.provider,
        crop_size_m=float(args.crop_size_m),
        crop_px=int(args.crop_px),
        allow_missing_aerial=_parse_bool(args.allow_missing_aerial),
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
