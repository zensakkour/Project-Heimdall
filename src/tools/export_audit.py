"""
Export per-image audit JSON with intermediate signals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.geo.geoclip_provider import GeoCLIPProvider
from src.core.logic.fusion import fuse_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Export audit JSON per image.")
    parser.add_argument("images", nargs="+", help="Image paths")
    parser.add_argument("--output-dir", default="runs/audit", help="Output directory")
    args = parser.parse_args()

    provider = GeoCLIPProvider()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in args.images:
        candidates = provider.candidates(image_path)
        fused = fuse_candidates(image_path, candidates, detections=[])
        payload = {
            "image": image_path,
            "candidates": [
                {
                    "latitude": c.latitude,
                    "longitude": c.longitude,
                    "retrieval_score": c.retrieval_score,
                    "match_id": c.match_id,
                }
                for c in candidates
            ],
            "fusion": None,
        }
        if fused is not None:
            payload["fusion"] = {
                "mean_latitude": fused.mean_latitude,
                "mean_longitude": fused.mean_longitude,
                "uncertainty_radius_m": fused.uncertainty_radius_m,
            }
        out_path = out_dir / (Path(image_path).stem + ".audit.json")
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


