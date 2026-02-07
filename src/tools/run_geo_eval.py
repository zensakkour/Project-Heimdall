"""
Evaluate geo localization accuracy against a metadata CSV (path, latitude, longitude).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.detection.factory import create_detector
from src.core.geo import GeoCLIPProvider, GeoLocator, GeoRetrievalProvider, MultiCandidateProvider
from src.core.logic.config import HeimdallConfig, load_config
from src.core.logic.pipeline import HeimdallPipeline


def build_pipeline(cfg: Optional[HeimdallConfig]) -> HeimdallPipeline:
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
        model_id=cfg.geolocator.retrieval_model_id or "openai/clip-vit-large-patch14",
        top_k=cfg.geolocator.retrieval_top_k,
        min_score=cfg.geolocator.retrieval_min_score,
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
    if cfg.geolocator.retrieval_index_path:
        candidate_provider = MultiCandidateProvider([retrieval_provider, geoclip_provider])
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
    image_path: str, cfg: Optional[HeimdallConfig]
) -> tuple[Optional[tuple[float, float]], Optional[float], Optional[str]]:
    if cfg is None or not cfg.geolocator.retrieval_index_path:
        return None, None, "index_not_configured"
    provider = GeoRetrievalProvider(
        index_path=cfg.geolocator.retrieval_index_path,
        model_id=cfg.geolocator.retrieval_model_id or "openai/clip-vit-large-patch14",
        top_k=cfg.geolocator.retrieval_top_k,
        min_score=cfg.geolocator.retrieval_min_score,
    )
    candidates = provider.candidates(image_path)
    if not candidates:
        return None, None, provider.last_error or "no_candidates"
    top = candidates[0]
    return (top.latitude, top.longitude), top.retrieval_score, provider.last_error


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


def main(argv: Optional[list[str]] = None) -> None:
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
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else None
    pipeline = None
    if not args.retrieval_only:
        pipeline = build_pipeline(cfg)

    metadata_path = Path(args.metadata)
    images_dir = Path(args.images_dir)
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
            pred, top_score, provider_error = predict_latlon_retrieval(str(image_path), cfg)
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
        "retrieval_only": bool(args.retrieval_only),
        "images_dir": str(images_dir),
        "metadata": str(metadata_path),
        "index_path": cfg.geolocator.retrieval_index_path if cfg else None,
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
