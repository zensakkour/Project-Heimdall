"""
Full pipeline runner (stub).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.core.detection.factory import create_detector
from src.core.geo import GeoCLIPProvider, GeoLocator, GeoRetrievalProvider, MultiCandidateProvider
from src.core.logic.config import FusionConfig, has_retrieval_index, load_config
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
        retrieval_provider = GeoRetrievalProvider(
            index_path=cfg.geolocator.retrieval_index_path,
            index_paths=cfg.geolocator.retrieval_index_paths,
            index_weights=cfg.geolocator.retrieval_index_weights,
            index_model_ids=cfg.geolocator.retrieval_index_model_ids,
            model_id=cfg.geolocator.retrieval_model_id or "openai/clip-vit-large-patch14",
            top_k=cfg.geolocator.retrieval_top_k,
            per_index_top_k=cfg.geolocator.retrieval_per_index_top_k,
            index_score_norm=cfg.geolocator.retrieval_index_score_norm,
            source_balance_beta=cfg.geolocator.retrieval_source_balance_beta,
            min_score=cfg.geolocator.retrieval_min_score,
            min_keep_topk=cfg.geolocator.retrieval_min_keep_topk,
            diversity_radius_km=cfg.geolocator.retrieval_diversity_radius_km,
            diversity_lambda=cfg.geolocator.retrieval_diversity_lambda,
            diversity_min_keep=cfg.geolocator.retrieval_diversity_min_keep,
            locality_radius_km=cfg.geolocator.retrieval_locality_radius_km,
            locality_weight=cfg.geolocator.retrieval_locality_weight,
            consensus_top_n=cfg.geolocator.retrieval_consensus_top_n,
            consensus_radius_km=cfg.geolocator.retrieval_consensus_radius_km,
            consensus_score_power=cfg.geolocator.retrieval_consensus_score_power,
            query_tta_degrees=cfg.geolocator.retrieval_query_tta_degrees,
            query_tta_reduce=cfg.geolocator.retrieval_query_tta_reduce,
        )
        geoclip_provider = GeoCLIPProvider(
            model_path=cfg.geolocator.model_path,
            model_id=cfg.geolocator.model_id,
            model_cache_dir=cfg.geolocator.model_cache_dir,
            top_n=cfg.geolocator.top_n,
            use_sidecar=cfg.geolocator.use_sidecar,
            use_exif=cfg.geolocator.use_exif,
        )
        if has_retrieval_index(cfg.geolocator):
            candidate_provider = MultiCandidateProvider(
                [retrieval_provider, geoclip_provider],
                dedupe_radius_m=cfg.geolocator.candidate_dedupe_radius_m,
                source_balance_beta=cfg.geolocator.candidate_source_balance_beta,
                max_candidates=cfg.geolocator.candidate_max_results,
            )
        else:
            candidate_provider = geoclip_provider
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
