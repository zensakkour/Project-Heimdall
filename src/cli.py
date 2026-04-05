"""
Minimal CLI to run the pipeline on a local image.
"""
from __future__ import annotations

import argparse
import json

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
        fusion_cfg = FusionConfig()
        ver_cfg = VerificationConfig()
        if args.no_shadow:
            ver_cfg = VerificationConfig(use_shadow=False, use_shadow_length=False, use_shadow_heading=False)
        pipeline = build_pipeline(detector_cfg, geo_cfg, fusion_cfg, None, ver_cfg)

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


