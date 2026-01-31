"""
Sequence evaluation harness (stub).
"""
from __future__ import annotations

import argparse
import json

from src.core.geo.geoclip_provider import GeoCLIPProvider
from src.core.logic.fusion import fuse_candidates
from src.core.logic.tracking import TrackState, associate_track


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a simple sequence eval.")
    parser.add_argument("images", nargs="+", help="Image paths")
    args = parser.parse_args()

    provider = GeoCLIPProvider()
    tracks: list[TrackState] = []
    outputs = []

    for idx, image_path in enumerate(args.images):
        candidates = provider.candidates(image_path)
        fused = fuse_candidates(image_path, candidates, detections=[])  # no detections in stub
        if fused is None:
            continue
        track = associate_track(tracks, fused)
        if track is None:
            track = TrackState(track_id=f"track-{idx}", fused_history=[])
            tracks.append(track)
        track.fused_history.append(fused)
        outputs.append({
            "image": image_path,
            "track_id": track.track_id,
            "mean_latitude": fused.mean_latitude,
            "mean_longitude": fused.mean_longitude,
        })

    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


