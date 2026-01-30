"""
Batch runner: process a directory of images and emit JSON lines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def iter_images(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch run Heimdall pipeline.")
    parser.add_argument("input", help="Directory of images")
    parser.add_argument("--weights", help="Path to YOLOv11-OBB weights")
    parser.add_argument("--geo-model", help="Path to GeoFT/GeoCLIP model")
    parser.add_argument("--config", help="Path to config JSON (overrides --weights/--geo-model)")
    parser.add_argument("--output", default="outputs.jsonl", help="Output JSONL file")
    parser.add_argument("--no-shadow", action="store_true", help="Disable shadow heuristics")
    args = parser.parse_args()

    root = Path(args.input)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Input directory not found: {root}")

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

    images = iter_images(root)

    with open(args.output, "w", encoding="utf-8") as f:
        for image_path in images:
            result = pipeline.run(str(image_path))
            payload = {
                "image": str(image_path),
                "result": assessment_to_dict(result),
            }
            f.write(json.dumps(payload) + "\n")

    print(f"Processed {len(images)} images -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
