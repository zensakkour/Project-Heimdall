"""
Batch runner: process a directory of images and emit JSON lines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.detection.factory import create_detector
from src.core.geo import GeoCLIPProvider, GeoLocator, GeoRetrievalProvider, MultiCandidateProvider
from src.core.logic.config import (
    DetectorConfig,
    GeoConfig,
    HeimdallConfig,
    FusionConfig,
    ScoreConfig,
    VerificationConfig,
    load_config,
)
from src.core.logic.pipeline import HeimdallPipeline
from src.core.logic.serialize import assessment_to_dict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_pipeline(
    detector_cfg: DetectorConfig | None,
    geo_cfg: GeoConfig | None,
    fusion_cfg: FusionConfig | None,
    score_cfg: ScoreConfig | None,
    verification_cfg: VerificationConfig | None,
) -> HeimdallPipeline:
    detector = create_detector(detector_cfg)
    geolocator = None
    candidate_provider = None
    if geo_cfg:
        geolocator = GeoLocator(
            geo_cfg.model_path,
            use_sidecar=geo_cfg.use_sidecar,
            use_exif=geo_cfg.use_exif,
        )
        retrieval_provider = GeoRetrievalProvider(
            index_path=geo_cfg.retrieval_index_path,
            model_id=geo_cfg.retrieval_model_id or "openai/clip-vit-large-patch14",
            top_k=geo_cfg.retrieval_top_k,
            min_score=geo_cfg.retrieval_min_score,
        )
        geoclip_provider = GeoCLIPProvider(
            model_path=geo_cfg.model_path,
            model_id=geo_cfg.model_id,
            model_cache_dir=geo_cfg.model_cache_dir,
            top_n=geo_cfg.top_n,
            use_sidecar=geo_cfg.use_sidecar,
            use_exif=geo_cfg.use_exif,
        )
        if geo_cfg.retrieval_index_path:
            candidate_provider = MultiCandidateProvider(
                [retrieval_provider, geoclip_provider],
                dedupe_radius_m=geo_cfg.candidate_dedupe_radius_m,
                max_candidates=geo_cfg.candidate_max_results,
            )
        else:
            candidate_provider = geoclip_provider
    return HeimdallPipeline(
        detector=detector,
        geolocator=geolocator,
        candidate_provider=candidate_provider,
        fusion_config=fusion_cfg,
        score_config=score_cfg,
        verification_config=verification_cfg,
    )


def build_pipeline_from_config(cfg: HeimdallConfig) -> HeimdallPipeline:
    return build_pipeline(cfg.detector, cfg.geolocator, cfg.fusion, cfg.score, cfg.verification)


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
        fusion_cfg = FusionConfig()
        ver_cfg = VerificationConfig()
        if args.no_shadow:
            ver_cfg = VerificationConfig(use_shadow=False, use_shadow_length=False, use_shadow_heading=False)
        pipeline = build_pipeline(detector_cfg, geo_cfg, fusion_cfg, None, ver_cfg)

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


