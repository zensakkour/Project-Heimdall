"""Download open geotagged image sets from Wikimedia Commons."""
from __future__ import annotations

import argparse
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple


OPEN_SAMPLES: List[Tuple[str, str, float, float]] = [
    ("Golden Gate Bridge, Aerial View", "Golden_Gate_Bridge-Aerial_View.jpg", 37.8199, -122.4783),
    ("Aerial view of the Sydney Opera House", "Aerial_view_of_the_Sydney_Opera_House.jpg", -33.8568, 151.2153),
    (
        "Aerial photograph of Eiffel Tower and Front de Seine, Paris 2005",
        "Aerial_photograph_of_Eiffel_Tower_and_Front_de_Seine,_Paris_2005.jpg",
        48.8584,
        2.2945,
    ),
    ("Foto aerea del Colosseo", "Foto_aerea_del_Colosseo.jpg", 41.8902, 12.4922),
    (
        "NYC skyline with Statue of Liberty at dusk (aerial)",
        "New_York_City_skyline_with_Statue_of_Liberty_at_dusk_aerial_2018.jpg",
        40.6892,
        -74.0445,
    ),
]

WORLD_ANCHORS: List[Tuple[str, float, float]] = [
    ("San Francisco", 37.7749, -122.4194),
    ("Los Angeles", 34.0522, -118.2437),
    ("New York", 40.7128, -74.0060),
    ("Chicago", 41.8781, -87.6298),
    ("Mexico City", 19.4326, -99.1332),
    ("Rio de Janeiro", -22.9068, -43.1729),
    ("Buenos Aires", -34.6037, -58.3816),
    ("London", 51.5074, -0.1278),
    ("Paris", 48.8566, 2.3522),
    ("Berlin", 52.5200, 13.4050),
    ("Rome", 41.9028, 12.4964),
    ("Madrid", 40.4168, -3.7038),
    ("Lisbon", 38.7223, -9.1393),
    ("Amsterdam", 52.3676, 4.9041),
    ("Istanbul", 41.0082, 28.9784),
    ("Cairo", 30.0444, 31.2357),
    ("Cape Town", -33.9249, 18.4241),
    ("Nairobi", -1.2921, 36.8219),
    ("Lagos", 6.5244, 3.3792),
    ("Dubai", 25.2048, 55.2708),
    ("Mumbai", 19.0760, 72.8777),
    ("Delhi", 28.6139, 77.2090),
    ("Bangkok", 13.7563, 100.5018),
    ("Singapore", 1.3521, 103.8198),
    ("Jakarta", -6.2088, 106.8456),
    ("Hong Kong", 22.3193, 114.1694),
    ("Tokyo", 35.6762, 139.6503),
    ("Seoul", 37.5665, 126.9780),
    ("Beijing", 39.9042, 116.4074),
    ("Sydney", -33.8688, 151.2093),
    ("Melbourne", -37.8136, 144.9631),
    ("Auckland", -36.8509, 174.7645),
]


def _fetch_json(url: str) -> Optional[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "Project-Heimdall/1.0 (data bootstrap)"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:  # nosec - public read-only API endpoint
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _parse_commons_geosearch_payload(payload: dict) -> List[Dict[str, object]]:
    pages = payload.get("query", {}).get("pages", []) if isinstance(payload, dict) else []
    out: List[Dict[str, object]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        title = str(page.get("title") or "")
        if not title.startswith("File:"):
            continue
        coords = page.get("coordinates")
        if not isinstance(coords, list) or not coords:
            continue
        first = coords[0]
        if not isinstance(first, dict):
            continue
        lat = first.get("lat")
        lon = first.get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue
        imageinfo = page.get("imageinfo")
        if not isinstance(imageinfo, list) or not imageinfo:
            continue
        info = imageinfo[0]
        if not isinstance(info, dict):
            continue
        file_url = info.get("url")
        if not isinstance(file_url, str) or not file_url:
            continue
        description_url = info.get("descriptionurl")
        extmeta = info.get("extmetadata") if isinstance(info.get("extmetadata"), dict) else {}
        license_short = ""
        license_url = ""
        if isinstance(extmeta, dict):
            lic = extmeta.get("LicenseShortName")
            if isinstance(lic, dict) and isinstance(lic.get("value"), str):
                license_short = lic["value"]
            licu = extmeta.get("LicenseUrl")
            if isinstance(licu, dict) and isinstance(licu.get("value"), str):
                license_url = licu["value"]
        out.append(
            {
                "title": title[5:],
                "url": file_url,
                "source_url": description_url or file_url,
                "latitude": float(lat),
                "longitude": float(lon),
                "license": license_short,
                "license_url": license_url,
            }
        )
    return out


def _fetch_geosearch_candidates(latitude: float, longitude: float, radius_m: int, limit: int) -> List[Dict[str, object]]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "geosearch",
        "ggscoord": f"{latitude}|{longitude}",
        "ggsradius": str(max(1000, min(10000, int(radius_m)))),
        "ggslimit": str(max(1, min(50, int(limit)))),
        "ggsnamespace": "6",
        "prop": "coordinates|imageinfo",
        "colimit": "1",
        "iiprop": "url|extmetadata",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    payload = _fetch_json(url)
    if payload is None:
        return []
    return _parse_commons_geosearch_payload(payload)


def _safe_ext(url: str) -> str:
    lower = url.lower()
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        if lower.endswith(ext):
            return ext
    return ".jpg"


def _download(url: str, out_path: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "Project-Heimdall/1.0 (local test)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # nosec - public file URL
            out_path.write_bytes(resp.read())
        return True
    except Exception:
        return False


def download_open_geo(output_dir: Path, limit: int, seed: int = 42, per_anchor: int = 20, radius_m: int = 8000) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.csv"

    rows = []
    rng = random.Random(seed)
    anchors = WORLD_ANCHORS[:]
    rng.shuffle(anchors)
    seen_urls = set()

    for _city, lat_anchor, lon_anchor in anchors:
        if len(rows) >= limit:
            break
        candidates = _fetch_geosearch_candidates(lat_anchor, lon_anchor, radius_m=radius_m, limit=per_anchor)
        rng.shuffle(candidates)
        for item in candidates:
            if len(rows) >= limit:
                break
            url = str(item.get("url") or "")
            if not url or url in seen_urls:
                continue
            title = str(item.get("title") or "")
            lat = float(item.get("latitude"))
            lon = float(item.get("longitude"))
            ext = _safe_ext(url)
            out_name = f"{Path(title).stem}{ext}"
            out_path = images_dir / out_name
            if not _download(url, out_path):
                continue
            seen_urls.add(url)
            rows.append(
                {
                    "path": f"images/{out_name}",
                    "latitude": lat,
                    "longitude": lon,
                    "title": title,
                    "license": str(item.get("license") or ""),
                    "license_url": str(item.get("license_url") or ""),
                    "source_url": str(item.get("source_url") or url),
                }
            )

    if len(rows) < min(limit, 10):
        samples = OPEN_SAMPLES[:]
        rng.shuffle(samples)
        for title, filename, lat, lon in samples:
            if len(rows) >= limit:
                break
            url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(filename)
            if url in seen_urls:
                continue
            ext = _safe_ext(url)
            out_name = f"{Path(filename).stem}{ext}"
            out_path = images_dir / out_name
            if not _download(url, out_path):
                continue
            seen_urls.add(url)
            rows.append(
                {
                    "path": f"images/{out_name}",
                    "latitude": lat,
                    "longitude": lon,
                    "title": title,
                    "license": "",
                    "license_url": "",
                    "source_url": url,
                }
            )

    unique_rows = []
    seen_paths = set()
    for row in rows:
        path = row["path"]
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique_rows.append(row)
    rows = unique_rows

    if not rows:
        return 0

    with meta_path.open("w", encoding="utf-8") as handle:
        handle.write("path,latitude,longitude,title,license,license_url,source_url\n")
        for row in rows:
            handle.write(
                f"{row['path']},{row['latitude']},{row['longitude']},"
                f"\"{row['title'].replace('\"', '\"\"')}\","
                f"\"{row['license'].replace('\"', '\"\"')}\","
                f"\"{row['license_url'].replace('\"', '\"\"')}\","
                f"\"{row['source_url'].replace('\"', '\"\"')}\"\n"
            )
    return len(rows)


def _download_curated(output_dir: Path, limit: int) -> int:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.csv"
    rows = []
    for title, filename, lat, lon in OPEN_SAMPLES:
        if len(rows) >= limit:
            break
        url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(filename)
        ext = _safe_ext(url)
        out_name = f"{Path(filename).stem}{ext}"
        out_path = images_dir / out_name
        if not _download(url, out_path):
            continue
        rows.append(
            {
                "path": f"images/{out_name}",
                "latitude": lat,
                "longitude": lon,
                "title": title,
                "license": "",
                "license_url": "",
                "source_url": url,
            }
        )
    if not rows:
        return 0
    with meta_path.open("w", encoding="utf-8") as handle:
        handle.write("path,latitude,longitude,title,license,license_url,source_url\n")
        for row in rows:
            handle.write(
                f"{row['path']},{row['latitude']},{row['longitude']},"
                f"\"{row['title'].replace('\"', '\"\"')}\","
                f"\"{row['license'].replace('\"', '\"\"')}\","
                f"\"{row['license_url'].replace('\"', '\"\"')}\","
                f"\"{row['source_url'].replace('\"', '\"\"')}\"\n"
            )
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download open geotagged images from Wikimedia Commons.")
    parser.add_argument("--limit", type=int, default=100, help="Number of images to download")
    parser.add_argument("--output", default="data/open_geo", help="Output folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for anchor/candidate shuffle.")
    parser.add_argument("--per-anchor", type=int, default=20, help="Max candidates fetched per world anchor.")
    parser.add_argument("--radius-m", type=int, default=8000, help="Geosearch radius in meters (1000..10000).")
    parser.add_argument(
        "--mode",
        choices=["api", "curated"],
        default="api",
        help="`api` fetches broader data using Wikimedia geosearch; `curated` uses fixed fallback samples.",
    )
    args = parser.parse_args()

    if args.mode == "curated":
        count = _download_curated(Path(args.output), args.limit)
    else:
        count = download_open_geo(
            Path(args.output),
            args.limit,
            seed=args.seed,
            per_anchor=args.per_anchor,
            radius_m=args.radius_m,
        )
    if count == 0:
        print("No images downloaded.")
        return 1
    print(f"Downloaded {count} images to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
