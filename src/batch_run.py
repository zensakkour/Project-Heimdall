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
    has_retrieval_index,
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
            index_paths=geo_cfg.retrieval_index_paths,
            index_weights=geo_cfg.retrieval_index_weights,
            index_model_ids=geo_cfg.retrieval_index_model_ids,
            model_id=geo_cfg.retrieval_model_id or "openai/clip-vit-large-patch14",
            projection_path=geo_cfg.retrieval_projection_path,
            top_k=geo_cfg.retrieval_top_k,
            per_index_top_k=geo_cfg.retrieval_per_index_top_k,
            index_score_norm=geo_cfg.retrieval_index_score_norm,
            source_balance_beta=geo_cfg.retrieval_source_balance_beta,
            min_score=geo_cfg.retrieval_min_score,
            min_keep_topk=geo_cfg.retrieval_min_keep_topk,
            diversity_radius_km=geo_cfg.retrieval_diversity_radius_km,
            diversity_lambda=geo_cfg.retrieval_diversity_lambda,
            diversity_min_keep=geo_cfg.retrieval_diversity_min_keep,
            locality_radius_km=geo_cfg.retrieval_locality_radius_km,
            locality_weight=geo_cfg.retrieval_locality_weight,
            consensus_top_n=geo_cfg.retrieval_consensus_top_n,
            consensus_radius_km=geo_cfg.retrieval_consensus_radius_km,
            consensus_score_power=geo_cfg.retrieval_consensus_score_power,
            query_tta_degrees=geo_cfg.retrieval_query_tta_degrees,
            query_tta_modes=geo_cfg.retrieval_query_tta_modes,
            query_tta_scales=geo_cfg.retrieval_query_tta_scales,
            query_tta_auto_modality=geo_cfg.retrieval_query_tta_auto_modality,
            query_tta_reduce=geo_cfg.retrieval_query_tta_reduce,
            query_expansion_top_n=geo_cfg.retrieval_query_expansion_top_n,
            query_expansion_beta=geo_cfg.retrieval_query_expansion_beta,
            query_expansion_alpha=geo_cfg.retrieval_query_expansion_alpha,
            tta_agreement_top_n=geo_cfg.retrieval_tta_agreement_top_n,
            tta_agreement_weight=geo_cfg.retrieval_tta_agreement_weight,
            local_match_top_n=geo_cfg.retrieval_local_match_top_n,
            local_match_weight=geo_cfg.retrieval_local_match_weight,
            local_match_ratio=geo_cfg.retrieval_local_match_ratio,
            local_match_max_features=geo_cfg.retrieval_local_match_max_features,
            graph_rerank_top_n=geo_cfg.retrieval_graph_rerank_top_n,
            graph_rerank_sigma_km=geo_cfg.retrieval_graph_rerank_sigma_km,
            graph_rerank_score_alpha=geo_cfg.retrieval_graph_rerank_score_alpha,
            graph_rerank_support_beta=geo_cfg.retrieval_graph_rerank_support_beta,
            graph_rerank_center_radius_km=geo_cfg.retrieval_graph_rerank_center_radius_km,
            kde_refine_top_n=geo_cfg.retrieval_kde_refine_top_n,
            kde_refine_sigma_km=geo_cfg.retrieval_kde_refine_sigma_km,
            kde_refine_score_power=geo_cfg.retrieval_kde_refine_score_power,
            kde_refine_margin_threshold=geo_cfg.retrieval_kde_refine_margin_threshold,
            kde_refine_switch_radius_km=geo_cfg.retrieval_kde_refine_switch_radius_km,
            kde_refine_max_iters=geo_cfg.retrieval_kde_refine_max_iters,
            kde_refine_adaptive_mass=geo_cfg.retrieval_kde_refine_adaptive_mass,
            source_fusion_mode=geo_cfg.retrieval_source_fusion_mode,
        )
        geoclip_provider = GeoCLIPProvider(
            model_path=geo_cfg.model_path,
            model_id=geo_cfg.model_id,
            model_cache_dir=geo_cfg.model_cache_dir,
            top_n=geo_cfg.top_n,
            use_sidecar=geo_cfg.use_sidecar,
            use_exif=geo_cfg.use_exif,
        )
        if has_retrieval_index(geo_cfg):
            candidate_provider = MultiCandidateProvider(
                [retrieval_provider, geoclip_provider],
                dedupe_radius_m=geo_cfg.candidate_dedupe_radius_m,
                source_balance_beta=geo_cfg.candidate_source_balance_beta,
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
