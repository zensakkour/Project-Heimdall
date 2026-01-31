"""Download a small open geotagged image set from Wikimedia Commons."""
from __future__ import annotations

import argparse
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Tuple


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


def download_open_geo(output_dir: Path, limit: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.csv"

    rows = []
    samples = OPEN_SAMPLES[:]
    random.shuffle(samples)
    for title, filename, lat, lon in samples:
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
    args = parser.parse_args()

    count = download_open_geo(Path(args.output), args.limit)
    if count == 0:
        print("No images downloaded.")
        return 1
    print(f"Downloaded {count} images to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
