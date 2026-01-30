"""
Image metadata helpers (EXIF).
"""
from __future__ import annotations

from datetime import datetime
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
        try:
            return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
        except Exception:
            continue
    return None
