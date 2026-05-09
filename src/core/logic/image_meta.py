"""
Image metadata helpers (EXIF).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

from PIL import Image, ExifTags


_EXIF_TAGS = ExifTags.TAGS
_GPS_TAGS = ExifTags.GPSTAGS


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _dms_to_degrees(dms: tuple[Any, Any, Any]) -> float:
    degrees = _to_float(dms[0])
    minutes = _to_float(dms[1])
    seconds = _to_float(dms[2])
    return degrees + (minutes / 60.0) + (seconds / 3600.0)


def _get_exif(image_path: str) -> dict[str, Any]:
    try:
        with Image.open(image_path) as img:
            raw = img._getexif() or {}
    except Exception:
        return {}

    exif: dict[str, Any] = {}
    for tag_id, value in raw.items():
        tag = _EXIF_TAGS.get(tag_id, tag_id)
        if tag == "GPSInfo" and isinstance(value, dict):
            gps: dict[str, Any] = {}
            for gps_id, gps_value in value.items():
                gps_tag = _GPS_TAGS.get(gps_id, gps_id)
                gps[gps_tag] = gps_value
            exif[tag] = gps
        else:
            exif[tag] = value
    return exif


def extract_gps(image_path: str) -> Optional[Tuple[float, float]]:
    exif = _get_exif(image_path)
    gps = exif.get("GPSInfo")
    if not isinstance(gps, dict):
        return None

    lat = gps.get("GPSLatitude")
    lat_ref = gps.get("GPSLatitudeRef")
    lon = gps.get("GPSLongitude")
    lon_ref = gps.get("GPSLongitudeRef")
    if not (lat and lat_ref and lon and lon_ref):
        return None

    latitude = _dms_to_degrees(lat)
    longitude = _dms_to_degrees(lon)
    if str(lat_ref).upper() == "S":
        latitude = -latitude
    if str(lon_ref).upper() == "W":
        longitude = -longitude
    return latitude, longitude


def extract_capture_time(image_path: str) -> Optional[datetime]:
    exif = _get_exif(image_path)
    for key in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
        value = exif.get(key)
        if not value:
            continue
        parsed = parse_capture_time(value)
        if parsed is not None:
            return parsed
    return _extract_sidecar_capture_time(image_path)


def _extract_sidecar_capture_time(image_path: str) -> Optional[datetime]:
    path = Path(image_path)
    candidates = [
        Path(str(path) + ".meta.json"),
        path.with_suffix(".meta.json"),
        Path(str(path) + ".geo.json"),
        path.with_suffix(".geo.json"),
        Path(str(path) + ".geoloc.json"),
        path.with_suffix(".geoloc.json"),
        Path(str(path) + ".detections.json"),
        path.with_suffix(".detections.json"),
    ]
    for sidecar in candidates:
        if not sidecar.exists():
            continue
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            continue
        parsed = _capture_time_from_payload(raw)
        if parsed is not None:
            return parsed
    return None


def _capture_time_from_payload(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, dict):
        return None
    for key in (
        "captured_at",
        "capture_time",
        "datetime",
        "datetimetz",
        "DateTimeOriginal",
        "DateTime",
        "DateTimeDigitized",
    ):
        parsed = parse_capture_time(raw.get(key))
        if parsed is not None:
            return parsed
    for key in ("image", "metadata", "meta", "exif"):
        nested = raw.get(key)
        if isinstance(nested, dict):
            parsed = _capture_time_from_payload(nested)
            if parsed is not None:
                return parsed
    return None


def parse_capture_time(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return _parse_epoch_capture_time(float(value))
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.isdigit():
        return _parse_epoch_capture_time(float(text))
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(iso_text)
    except ValueError:
        return None


def _parse_epoch_capture_time(value: float) -> Optional[datetime]:
    if value <= 0.0:
        return None
    # Mapillary exports milliseconds, while some feeds export seconds.
    seconds = value / 1000.0 if value > 10_000_000_000 else value
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


