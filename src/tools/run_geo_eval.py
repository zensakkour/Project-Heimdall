"""
Evaluate geo localization accuracy against a metadata CSV (path, latitude, longitude).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.core.geo import GeoCLIPProvider, GeoLocator, GeoRetrievalProvider, MultiCandidateProvider
from src.core.logic.config import HeimdallConfig, has_retrieval_index, load_config

if TYPE_CHECKING:
    from src.core.logic.pipeline import HeimdallPipeline


def build_pipeline(cfg: Optional[HeimdallConfig]) -> "HeimdallPipeline":
    from src.core.detection.factory import create_detector
    from src.core.logic.pipeline import HeimdallPipeline

    if cfg is None:
        return HeimdallPipeline()
    detector = create_detector(cfg.detector)
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
        source_fusion_mode=cfg.geolocator.retrieval_source_fusion_mode,
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
        query_tta_modes=cfg.geolocator.retrieval_query_tta_modes,
        query_tta_scales=cfg.geolocator.retrieval_query_tta_scales,
        query_tta_auto_modality=cfg.geolocator.retrieval_query_tta_auto_modality,
        query_tta_reduce=cfg.geolocator.retrieval_query_tta_reduce,
        query_expansion_top_n=cfg.geolocator.retrieval_query_expansion_top_n,
        query_expansion_beta=cfg.geolocator.retrieval_query_expansion_beta,
        query_expansion_alpha=cfg.geolocator.retrieval_query_expansion_alpha,
        tta_agreement_top_n=cfg.geolocator.retrieval_tta_agreement_top_n,
        tta_agreement_weight=cfg.geolocator.retrieval_tta_agreement_weight,
        local_match_top_n=cfg.geolocator.retrieval_local_match_top_n,
        local_match_weight=cfg.geolocator.retrieval_local_match_weight,
        local_match_ratio=cfg.geolocator.retrieval_local_match_ratio,
        local_match_max_features=cfg.geolocator.retrieval_local_match_max_features,
        graph_rerank_top_n=cfg.geolocator.retrieval_graph_rerank_top_n,
        graph_rerank_sigma_km=cfg.geolocator.retrieval_graph_rerank_sigma_km,
        graph_rerank_score_alpha=cfg.geolocator.retrieval_graph_rerank_score_alpha,
        graph_rerank_support_beta=cfg.geolocator.retrieval_graph_rerank_support_beta,
        graph_rerank_center_radius_km=cfg.geolocator.retrieval_graph_rerank_center_radius_km,
        kde_refine_top_n=cfg.geolocator.retrieval_kde_refine_top_n,
        kde_refine_sigma_km=cfg.geolocator.retrieval_kde_refine_sigma_km,
        kde_refine_score_power=cfg.geolocator.retrieval_kde_refine_score_power,
        kde_refine_margin_threshold=cfg.geolocator.retrieval_kde_refine_margin_threshold,
        kde_refine_switch_radius_km=cfg.geolocator.retrieval_kde_refine_switch_radius_km,
        kde_refine_max_iters=cfg.geolocator.retrieval_kde_refine_max_iters,
        kde_refine_adaptive_mass=cfg.geolocator.retrieval_kde_refine_adaptive_mass,
    )
    geoclip_provider = GeoCLIPProvider(
        model_path=cfg.geolocator.model_path,
        model_id=cfg.geolocator.model_id,
        model_cache_dir=cfg.geolocator.model_cache_dir,
        encoder_name=cfg.geolocator.encoder_name,
        top_n=cfg.geolocator.top_n,
        use_sidecar=cfg.geolocator.use_sidecar,
        use_exif=cfg.geolocator.use_exif,
        score_scale=cfg.geolocator.geospot_score_scale,
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
    return HeimdallPipeline(
        detector=detector,
        geolocator=geolocator,
        candidate_provider=candidate_provider,
        fusion_config=cfg.fusion,
        score_config=cfg.score,
        verification_config=cfg.verification,
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def predict_latlon(result) -> Optional[tuple[float, float]]:
    fusion = getattr(result, "fusion", None)
    if fusion and fusion.mean_latitude is not None and fusion.mean_longitude is not None:
        return fusion.mean_latitude, fusion.mean_longitude
    candidates = getattr(result, "candidates", []) or []
    if not candidates:
        return None
    top = candidates[0]
    return top.latitude, top.longitude


def predict_latlon_retrieval(
    image_path: str, provider: Optional[GeoRetrievalProvider]
) -> tuple[Optional[tuple[float, float]], Optional[float], Optional[str]]:
    if provider is None:
        return None, None, "index_not_configured"
    candidates = provider.candidates(image_path)
    if not candidates:
        return None, None, provider.last_error or "no_candidates"
    top = candidates[0]
    return (top.latitude, top.longitude), top.retrieval_score, provider.last_error


def build_retrieval_provider(cfg: Optional[HeimdallConfig]) -> Optional[GeoRetrievalProvider]:
    if cfg is None or not has_retrieval_index(cfg.geolocator):
        return None
    return GeoRetrievalProvider(
        index_path=cfg.geolocator.retrieval_index_path,
        index_paths=cfg.geolocator.retrieval_index_paths,
        index_weights=cfg.geolocator.retrieval_index_weights,
        index_model_ids=cfg.geolocator.retrieval_index_model_ids,
        model_id=cfg.geolocator.retrieval_model_id or "openai/clip-vit-large-patch14",
        top_k=cfg.geolocator.retrieval_top_k,
        per_index_top_k=cfg.geolocator.retrieval_per_index_top_k,
        index_score_norm=cfg.geolocator.retrieval_index_score_norm,
        source_fusion_mode=cfg.geolocator.retrieval_source_fusion_mode,
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
        query_tta_modes=cfg.geolocator.retrieval_query_tta_modes,
        query_tta_scales=cfg.geolocator.retrieval_query_tta_scales,
        query_tta_auto_modality=cfg.geolocator.retrieval_query_tta_auto_modality,
        query_tta_reduce=cfg.geolocator.retrieval_query_tta_reduce,
        query_expansion_top_n=cfg.geolocator.retrieval_query_expansion_top_n,
        query_expansion_beta=cfg.geolocator.retrieval_query_expansion_beta,
        query_expansion_alpha=cfg.geolocator.retrieval_query_expansion_alpha,
        tta_agreement_top_n=cfg.geolocator.retrieval_tta_agreement_top_n,
        tta_agreement_weight=cfg.geolocator.retrieval_tta_agreement_weight,
        local_match_top_n=cfg.geolocator.retrieval_local_match_top_n,
        local_match_weight=cfg.geolocator.retrieval_local_match_weight,
        local_match_ratio=cfg.geolocator.retrieval_local_match_ratio,
        local_match_max_features=cfg.geolocator.retrieval_local_match_max_features,
        graph_rerank_top_n=cfg.geolocator.retrieval_graph_rerank_top_n,
        graph_rerank_sigma_km=cfg.geolocator.retrieval_graph_rerank_sigma_km,
        graph_rerank_score_alpha=cfg.geolocator.retrieval_graph_rerank_score_alpha,
        graph_rerank_support_beta=cfg.geolocator.retrieval_graph_rerank_support_beta,
        graph_rerank_center_radius_km=cfg.geolocator.retrieval_graph_rerank_center_radius_km,
        kde_refine_top_n=cfg.geolocator.retrieval_kde_refine_top_n,
        kde_refine_sigma_km=cfg.geolocator.retrieval_kde_refine_sigma_km,
        kde_refine_score_power=cfg.geolocator.retrieval_kde_refine_score_power,
        kde_refine_margin_threshold=cfg.geolocator.retrieval_kde_refine_margin_threshold,
        kde_refine_switch_radius_km=cfg.geolocator.retrieval_kde_refine_switch_radius_km,
        kde_refine_max_iters=cfg.geolocator.retrieval_kde_refine_max_iters,
        kde_refine_adaptive_mass=cfg.geolocator.retrieval_kde_refine_adaptive_mass,
    )


def resolve_image_path(images_dir: Path, rel_path: str) -> Path:
    rel_path = rel_path.replace("\\", "/")
    direct = images_dir / rel_path
    if direct.exists():
        return direct
    parent = images_dir.parent / rel_path
    if parent.exists():
        return parent
    if rel_path.startswith("chips/"):
        trimmed = rel_path.split("chips/", 1)[1]
        candidate = images_dir / trimmed
        if candidate.exists():
            return candidate
    return direct


def normalize_scope(raw_scope: Optional[str]) -> str:
    if not raw_scope:
        return ""
    value = str(raw_scope).strip().upper().replace("-", "_")
    if "PARIS" in value:
        return "PARIS"
    if value in {"US", "USA"} or "OPEN_GEO" in value:
        return "US"
    if value == "GLOBAL":
        return "GLOBAL"
    if value == "UNKNOWN":
        return "UNKNOWN"
    return value


def load_profile_scope(config_path: str) -> str:
    path = Path(config_path)
    scope_raw: Optional[str] = None
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                value = payload.get("profile_scope")
                scope_raw = value if isinstance(value, str) else None
        except Exception:
            scope_raw = None
    scope = normalize_scope(scope_raw)
    if scope:
        return scope
    name = path.stem.lower()
    if "paris" in name:
        return "PARIS"
    if "open_geo" in name or name.startswith("us") or "_us" in name:
        return "US"
    return ""


def infer_dataset_scope(images_dir: Path, metadata_path: Path) -> str:
    blob = f"{images_dir.as_posix().lower()} {metadata_path.as_posix().lower()}"
    if "spacenet_paris" in blob or "/paris/" in blob or "_paris" in blob:
        return "PARIS"
    if "open_geo" in blob or "/us/" in blob or "_us" in blob:
        return "US"
    return "UNKNOWN"


def validate_scope_alignment(
    profile_scope: str,
    dataset_scope: str,
    *,
    allow_scope_mismatch: bool,
) -> Optional[str]:
    profile = normalize_scope(profile_scope)
    dataset = normalize_scope(dataset_scope)
    if not profile or profile == "GLOBAL":
        return None
    if dataset in {"", "UNKNOWN"}:
        return None
    if profile == dataset:
        return None
    warning = (
        f"profile/data scope mismatch: profile_scope='{profile}' "
        f"but inferred dataset_scope='{dataset}'"
    )
    if not allow_scope_mismatch:
        raise ValueError(f"{warning}. Use --allow-scope-mismatch to override.")
    return warning


def main(argv: Optional[list[str]] = None) -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description="Geo evaluation against metadata CSV.")
    parser.add_argument("--images-dir", required=True, help="Directory containing images.")
    parser.add_argument("--metadata", required=True, help="CSV with path, latitude, longitude.")
    parser.add_argument("--output", default="src/dashboard/data/geo_eval.json", help="Output JSON file.")
    parser.add_argument("--progress", default="", help="Optional progress JSON path.")
    parser.add_argument("--retrieval-only", action="store_true", help="Use retrieval-only scoring.")
    parser.add_argument("--diag-samples", type=int, default=5, help="Number of sample diagnostics.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0=all).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--config", default="src/config/defaults.json", help="Config file.")
    parser.add_argument(
        "--allow-scope-mismatch",
        action="store_true",
        help="Allow evaluation when config profile_scope and inferred dataset scope disagree.",
    )
    args = parser.parse_args(argv)

    profile_scope = load_profile_scope(args.config) if args.config else ""
    metadata_path = Path(args.metadata)
    images_dir = Path(args.images_dir)
    dataset_scope = infer_dataset_scope(images_dir, metadata_path)
    scope_warning = validate_scope_alignment(
        profile_scope,
        dataset_scope,
        allow_scope_mismatch=bool(args.allow_scope_mismatch),
    )

    cfg = load_config(args.config) if args.config else None
    pipeline = None
    retrieval_provider = None
    if not args.retrieval_only:
        pipeline = build_pipeline(cfg)
    else:
        retrieval_provider = build_retrieval_provider(cfg)

    df = pd.read_csv(metadata_path)
    if not {"path", "latitude", "longitude"}.issubset(df.columns):
        raise ValueError("metadata must include columns: path, latitude, longitude")

    records = df[["path", "latitude", "longitude"]].to_dict("records")
    random.Random(args.seed).shuffle(records)
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    total = len(records)

    distances = []
    missing = 0
    null_pred = 0
    retrieval_scores = []
    diagnostics = []
    progress_path = Path(args.progress) if args.progress else None
    for idx, item in enumerate(records, start=1):
        rel_path = str(item["path"])
        image_path = (
            Path(rel_path)
            if Path(rel_path).is_absolute()
            else resolve_image_path(images_dir, rel_path)
        )
        if not image_path.exists():
            missing += 1
            continue
        pred = None
        top_score = None
        provider_error = None
        if args.retrieval_only:
            pred, top_score, provider_error = predict_latlon_retrieval(str(image_path), retrieval_provider)
            if top_score is not None:
                retrieval_scores.append(float(top_score))
        else:
            if pipeline is None:
                pred = None
            else:
                result = pipeline.run(str(image_path))
                pred = predict_latlon(result)
        if pred is None:
            null_pred += 1
            distances.append(None)
            continue
        gt_lat = float(item["latitude"])
        gt_lon = float(item["longitude"])
        dist = haversine_km(gt_lat, gt_lon, pred[0], pred[1])
        distances.append(dist)
        if len(diagnostics) < max(0, args.diag_samples):
            diagnostics.append(
                {
                    "image": str(image_path),
                    "gt_lat": gt_lat,
                    "gt_lon": gt_lon,
                    "pred_lat": pred[0],
                    "pred_lon": pred[1],
                    "dist_km": dist,
                    "retrieval_score": top_score,
                    "provider_error": provider_error,
                }
            )

        if progress_path and (idx % 25 == 0 or idx == total):
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(
                json.dumps({"total": total, "processed": idx}, indent=2),
                encoding="utf-8",
            )

    valid = [d for d in distances if d is not None]
    valid.sort()
    evaluated = len(valid)

    def pct_within(km: float) -> float:
        if evaluated == 0:
            return 0.0
        count = sum(1 for d in valid if d <= km)
        return 100.0 * count / evaluated

    def percentile(p: float) -> Optional[float]:
        if not valid:
            return None
        k = max(0, min(len(valid) - 1, int(round((p / 100.0) * (len(valid) - 1)))))
        return float(valid[k])

    report = {
        "config": args.config,
        "profile_scope": profile_scope or None,
        "dataset_scope": dataset_scope,
        "scope_warning": scope_warning,
        "allow_scope_mismatch": bool(args.allow_scope_mismatch),
        "retrieval_only": bool(args.retrieval_only),
        "images_dir": str(images_dir),
        "metadata": str(metadata_path),
        "index_path": cfg.geolocator.retrieval_index_path if cfg else None,
        "index_paths": list(cfg.geolocator.retrieval_index_paths) if cfg else None,
        "index_weights": list(cfg.geolocator.retrieval_index_weights) if cfg else None,
        "index_model_ids": list(cfg.geolocator.retrieval_index_model_ids) if cfg else None,
        "retrieval_per_index_top_k": cfg.geolocator.retrieval_per_index_top_k if cfg else None,
        "retrieval_index_score_norm": cfg.geolocator.retrieval_index_score_norm if cfg else None,
        "retrieval_source_fusion_mode": cfg.geolocator.retrieval_source_fusion_mode if cfg else None,
        "retrieval_source_balance_beta": cfg.geolocator.retrieval_source_balance_beta if cfg else None,
        "candidate_source_balance_beta": cfg.geolocator.candidate_source_balance_beta if cfg else None,
        "retrieval_query_tta_degrees": list(cfg.geolocator.retrieval_query_tta_degrees) if cfg else None,
        "retrieval_query_tta_modes": list(cfg.geolocator.retrieval_query_tta_modes) if cfg else None,
        "retrieval_query_tta_scales": list(cfg.geolocator.retrieval_query_tta_scales) if cfg else None,
        "retrieval_query_tta_auto_modality": cfg.geolocator.retrieval_query_tta_auto_modality if cfg else None,
        "retrieval_query_tta_reduce": cfg.geolocator.retrieval_query_tta_reduce if cfg else None,
        "retrieval_query_expansion_top_n": cfg.geolocator.retrieval_query_expansion_top_n if cfg else None,
        "retrieval_query_expansion_beta": cfg.geolocator.retrieval_query_expansion_beta if cfg else None,
        "retrieval_query_expansion_alpha": cfg.geolocator.retrieval_query_expansion_alpha if cfg else None,
        "retrieval_tta_agreement_top_n": cfg.geolocator.retrieval_tta_agreement_top_n if cfg else None,
        "retrieval_tta_agreement_weight": cfg.geolocator.retrieval_tta_agreement_weight if cfg else None,
        "retrieval_local_match_top_n": cfg.geolocator.retrieval_local_match_top_n if cfg else None,
        "retrieval_local_match_weight": cfg.geolocator.retrieval_local_match_weight if cfg else None,
        "retrieval_local_match_ratio": cfg.geolocator.retrieval_local_match_ratio if cfg else None,
        "retrieval_local_match_max_features": cfg.geolocator.retrieval_local_match_max_features if cfg else None,
        "retrieval_graph_rerank_top_n": cfg.geolocator.retrieval_graph_rerank_top_n if cfg else None,
        "retrieval_graph_rerank_sigma_km": cfg.geolocator.retrieval_graph_rerank_sigma_km if cfg else None,
        "retrieval_graph_rerank_score_alpha": cfg.geolocator.retrieval_graph_rerank_score_alpha if cfg else None,
        "retrieval_graph_rerank_support_beta": cfg.geolocator.retrieval_graph_rerank_support_beta if cfg else None,
        "retrieval_graph_rerank_center_radius_km": cfg.geolocator.retrieval_graph_rerank_center_radius_km if cfg else None,
        "retrieval_kde_refine_top_n": cfg.geolocator.retrieval_kde_refine_top_n if cfg else None,
        "retrieval_kde_refine_sigma_km": cfg.geolocator.retrieval_kde_refine_sigma_km if cfg else None,
        "retrieval_kde_refine_score_power": cfg.geolocator.retrieval_kde_refine_score_power if cfg else None,
        "retrieval_kde_refine_margin_threshold": cfg.geolocator.retrieval_kde_refine_margin_threshold if cfg else None,
        "retrieval_kde_refine_switch_radius_km": cfg.geolocator.retrieval_kde_refine_switch_radius_km if cfg else None,
        "retrieval_kde_refine_max_iters": cfg.geolocator.retrieval_kde_refine_max_iters if cfg else None,
        "retrieval_kde_refine_adaptive_mass": cfg.geolocator.retrieval_kde_refine_adaptive_mass if cfg else None,
        "retrieval_min_keep_topk": cfg.geolocator.retrieval_min_keep_topk if cfg else None,
        "retrieval_consensus_top_n": cfg.geolocator.retrieval_consensus_top_n if cfg else None,
        "retrieval_consensus_radius_km": cfg.geolocator.retrieval_consensus_radius_km if cfg else None,
        "retrieval_consensus_score_power": cfg.geolocator.retrieval_consensus_score_power if cfg else None,
        "use_adaptive_outlier_guard": cfg.fusion.use_adaptive_outlier_guard if cfg else None,
        "outlier_guard_strength": cfg.fusion.outlier_guard_strength if cfg else None,
        "outlier_guard_min_scale_km": cfg.fusion.outlier_guard_min_scale_km if cfg else None,
        "outlier_guard_mad_scale": cfg.fusion.outlier_guard_mad_scale if cfg else None,
        "total": total,
        "evaluated": evaluated,
        "missing_files": missing,
        "null_predictions": null_pred,
        "retrieval_score_mean": float(sum(retrieval_scores) / len(retrieval_scores))
        if retrieval_scores
        else None,
        "retrieval_score_min": float(min(retrieval_scores)) if retrieval_scores else None,
        "retrieval_score_max": float(max(retrieval_scores)) if retrieval_scores else None,
        "mean_km": float(sum(valid) / evaluated) if evaluated else None,
        "median_km": float(valid[evaluated // 2]) if evaluated else None,
        "p90_km": float(valid[int(evaluated * 0.9) - 1]) if evaluated else None,
        "p10_km": percentile(10),
        "p25_km": percentile(25),
        "p75_km": percentile(75),
        "p95_km": percentile(95),
        "within_1km_pct": pct_within(1.0),
        "within_2km_pct": pct_within(2.0),
        "within_5km_pct": pct_within(5.0),
        "within_10km_pct": pct_within(10.0),
        "within_50km_pct": pct_within(50.0),
        "samples": diagnostics,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if progress_path:
        progress_path.write_text(
            json.dumps({"total": total, "processed": total}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
