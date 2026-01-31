"""
Download DOTA v1.0 (Ultralytics-hosted) dataset zip.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DOTA v1.0 dataset zip.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Download URL for DOTA v1.0 zip")
    parser.add_argument("--output", default="data/dota/DOTAv1.zip", help="Output zip path")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {out} ...")
    urllib.request.urlretrieve(args.url, out)  # nosec - trusted dataset URL
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


