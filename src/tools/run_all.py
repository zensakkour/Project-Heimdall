"""
Full pipeline runner (stub).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.core.detection.factory import create_detector
from src.core.geo import GeoCLIPProvider, GeoLocator
from src.core.logic.config import FusionConfig, load_config
from src.core.logic.pipeline import HeimdallPipeline
from src.core.logic.serialize import assessment_to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full pipeline for a list of images.")
    parser.add_argument("images", nargs="+", help="Image paths")
    parser.add_argument("--output", default="runs/results.jsonl")
    parser.add_argument("--config", help="Path to config JSON")
    args = parser.parse_args()

    cfg = None
    if args.config:
        cfg = load_config(args.config)
    else:
        default_path = Path("src/config/defaults.json")
        if default_path.exists():
            cfg = load_config(str(default_path))

    detector = create_detector(cfg.detector) if cfg else None
    geolocator = None
    candidate_provider = None
    fusion_cfg = None
    if cfg:
        geolocator = GeoLocator(
            cfg.geolocator.model_path,
            use_sidecar=cfg.geolocator.use_sidecar,
            use_exif=cfg.geolocator.use_exif,
        )
        candidate_provider = GeoCLIPProvider(
            model_path=cfg.geolocator.model_path,
            model_id=cfg.geolocator.model_id,
            model_cache_dir=cfg.geolocator.model_cache_dir,
            top_n=cfg.geolocator.top_n,
            use_sidecar=cfg.geolocator.use_sidecar,
            use_exif=cfg.geolocator.use_exif,
        )
        fusion_cfg = cfg.fusion
    elif args.images:
        candidate_provider = GeoCLIPProvider()
        fusion_cfg = FusionConfig()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = HeimdallPipeline(
        detector=detector,
        geolocator=geolocator,
        candidate_provider=candidate_provider,
        fusion_config=fusion_cfg,
        score_config=cfg.score if cfg else None,
        verification_config=cfg.verification if cfg else None,
    )

    with out_path.open("w", encoding="utf-8") as handle:
        for image_path in args.images:
            payload = {
                "image": image_path,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "result": None,
            }
            result = pipeline.run(image_path)
            payload["result"] = assessment_to_dict(result)
            handle.write(json.dumps(payload) + "\n")

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


