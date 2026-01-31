"""
Download GeoSpot Base model weights from Hugging Face.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download GeoSpot Base model weights.")
    parser.add_argument("--repo", default="sdan/geospot-base", help="Hugging Face repo id")
    parser.add_argument(
        "--cache-dir",
        default="data/models/geospot-base",
        help="Directory to store model weights",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as exc:
        raise SystemExit("huggingface_hub is required. pip install huggingface_hub") from exc

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=args.repo, filename="model.safetensors", cache_dir=str(cache_dir))
    print(f"Downloaded to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


