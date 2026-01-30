"""
Minimal CLI to run the pipeline on a local image.
"""
from __future__ import annotations

import argparse
import json

from core.detection.factory import create_detector
from core.geo import GeoLocator
from core.logic.config import (
    DetectorConfig,
    GeoConfig,
    HeimdallConfig,
    ScoreConfig,
    VerificationConfig,
    load_config,
)
from core.logic.pipeline import HeimdallPipeline
from core.logic.serialize import assessment_to_dict


def build_pipeline(
    detector_cfg: DetectorConfig | None,
    geo_cfg: GeoConfig | None,
    score_cfg: ScoreConfig | None,
    verification_cfg: VerificationConfig | None,
) -> HeimdallPipeline:
    detector = create_detector(detector_cfg)
    geolocator = None
    if geo_cfg:
        geolocator = GeoLocator(
            geo_cfg.model_path,
            use_sidecar=geo_cfg.use_sidecar,
            use_exif=geo_cfg.use_exif,
        )
    return HeimdallPipeline(
        detector=detector,
        geolocator=geolocator,
        score_config=score_cfg,
        verification_config=verification_cfg,
    )


def build_pipeline_from_config(cfg: HeimdallConfig) -> HeimdallPipeline:
    return build_pipeline(cfg.detector, cfg.geolocator, cfg.score, cfg.verification)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Heimdall pipeline on an image.")
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--weights", help="Path to YOLOv11-OBB weights")
    parser.add_argument("--geo-model", help="Path to GeoFT/GeoCLIP model")
    parser.add_argument("--config", help="Path to config JSON (overrides --weights/--geo-model)")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    parser.add_argument("--no-shadow", action="store_true", help="Disable shadow heuristics")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config)
        pipeline = build_pipeline_from_config(cfg)
    else:
        detector_cfg = DetectorConfig(weights_path=args.weights)
        geo_cfg = GeoConfig(model_path=args.geo_model)
        ver_cfg = VerificationConfig()
        if args.no_shadow:
            ver_cfg = VerificationConfig(use_shadow=False, use_shadow_length=False, use_shadow_heading=False)
        pipeline = build_pipeline(detector_cfg, geo_cfg, None, ver_cfg)

    result = pipeline.run(args.image)

    if args.json:
        payload = {"image": args.image, "result": assessment_to_dict(result)}
        print(json.dumps(payload))
    else:
        print("Detections:", len(result.detections))
        print("Geo:", result.geo)
        print("Verification:", result.verification)
        print("Score:", result.score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
