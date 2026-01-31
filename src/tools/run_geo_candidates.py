"""
Offline runner for geolocation candidates.
"""
from __future__ import annotations

import argparse
import json

from src.core.geo.geoclip_provider import GeoCLIPProvider


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate geo candidates for an image.")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--model", help="Path to GeoCLIP/GeoFT model")
    parser.add_argument("--model-id", default="sdan/geospot-base", help="Hugging Face model id")
    parser.add_argument("--cache-dir", default="data/models/geospot-base", help="Cache dir for model")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    provider = GeoCLIPProvider(
        model_path=args.model,
        model_id=args.model_id,
        model_cache_dir=args.cache_dir,
        top_n=args.top_n,
    )
    candidates = provider.candidates(args.image)
    payload = [
        {
            "latitude": c.latitude,
            "longitude": c.longitude,
            "retrieval_score": c.retrieval_score,
            "match_id": c.match_id,
        }
        for c in candidates
    ]
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


