"""
Benchmark detector throughput and detection counts on a list of images.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core.detection.factory import create_detector
from core.logic.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark detector on images.")
    parser.add_argument("--config", default="config/defaults.json", help="Path to config JSON")
    parser.add_argument("--images", nargs="+", help="Image paths to benchmark")
    parser.add_argument("--output", default="outputs_benchmark.json", help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    detector = create_detector(cfg.detector)
    if detector is None:
        raise SystemExit("Detector is not configured. Set detector.weights_path in config.")

    images = [Path(p) for p in (args.images or [])]
    if not images:
        raise SystemExit("No images provided.")

    results = []
    total_time = 0.0
    for image in images:
        start = time.perf_counter()
        dets = detector.predict(str(image))
        elapsed = time.perf_counter() - start
        total_time += elapsed
        results.append(
            {
                "image": str(image),
                "detections": len(dets),
                "time_s": round(elapsed, 4),
            }
        )

    avg = total_time / len(images)
    payload = {
        "images": results,
        "avg_time_s": round(avg, 4),
        "avg_fps": round(1.0 / avg, 2) if avg > 0 else 0.0,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
