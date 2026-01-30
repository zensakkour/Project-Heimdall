"""
Download sample aerial images/videos from Wikimedia Commons (thumbnails for images).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


OUT = Path("data/samples")
OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "Category:Aerial photographs of airports",
    "Category:Aerial photographs of seaports",
    "Category:Aerial photographs of harbours",
    "Category:Aerial photographs of airfields",
]

VIDEO_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/1/13/Port_of_Rotterdam_aerial_view.webm",
    "https://upload.wikimedia.org/wikipedia/commons/1/19/Aerial_flight_over_San_Diego_Harbor.webm",
]

USER_AGENT = "Mozilla/5.0"


def api_get(params: dict[str, str | int]) -> dict:
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download(url: str, filename: Path, retries: int = 3, base_sleep: int = 5) -> int:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
            filename.write_bytes(data)
            return len(data)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "Too many requests" in msg:
                time.sleep(base_sleep * (attempt + 1))
                continue
            raise
    raise RuntimeError("Rate limited. Try again later.")


def download_images(limit: int = 20, thumb_width: int = 1280) -> int:
    images = []
    seen = set()
    for cat in CATEGORIES:
        data = api_get(
            {
                "action": "query",
                "generator": "categorymembers",
                "gcmtitle": cat,
                "gcmtype": "file",
                "gcmlimit": 50,
                "prop": "imageinfo",
                "iiprop": "url|mime|thumburl",
                "iiurlwidth": thumb_width,
                "format": "json",
            }
        )
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title")
            if not title or title in seen:
                continue
            seen.add(title)
            info = page.get("imageinfo")
            if not info:
                continue
            info = info[0]
            mime = info.get("mime", "")
            thumb = info.get("thumburl")
            if not thumb or not mime.startswith("image/"):
                continue
            images.append((title, thumb, mime))
        if len(images) >= limit:
            break

    count = 0
    for idx, (_, url, mime) in enumerate(images[:limit], 1):
        ext = ".jpg" if mime.endswith("jpeg") else ".png"
        path = OUT / f"real_auto_{idx:02d}{ext}"
        if path.exists():
            count += 1
            continue
        try:
            size = download(url, path)
            print(path, size)
        except Exception as exc:
            print(f"Stopped due to error: {exc}")
            break
        count += 1
        time.sleep(3)
    return count


def download_videos() -> int:
    count = 0
    for idx, url in enumerate(VIDEO_URLS, 1):
        ext = ".webm"
        path = OUT / f"real_auto_video_{idx:02d}{ext}"
        if path.exists():
            count += 1
            continue
        try:
            size = download(url, path, retries=5, base_sleep=8)
            print(path, size)
        except Exception as exc:
            print(f"Stopped due to error: {exc}")
            break
        count += 1
        time.sleep(5)
    return count


def main() -> int:
    print("Downloading images...")
    download_images()
    print("Downloading videos...")
    download_videos()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
