"""
Config loading for pipeline components.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DetectorConfig:
    weights_path: Optional[str] = None
    min_confidence: float = 0.25
    nms_iou: float = 0.5
    max_detections: int = 100
    min_area_px: float = 16.0
    class_agnostic_nms: bool = False
    use_tta: bool = False
    use_sidecar: bool = True
    use_classic: bool = False
    imgsz: int = 1280


@dataclass(frozen=True)
class GeoConfig:
    model_path: Optional[str] = None
    model_id: Optional[str] = None
    model_cache_dir: Optional[str] = None
    encoder_name: Optional[str] = None
    use_sidecar: bool = True
    use_exif: bool = True
    top_n: int = 5
    geospot_score_scale: float = 1.0
    retrieval_index_path: Optional[str] = None
    retrieval_model_id: Optional[str] = None
    retrieval_top_k: int = 10
    retrieval_min_score: float = 0.2
    candidate_dedupe_radius_m: float = 300.0
    candidate_max_results: int = 80


@dataclass(frozen=True)
class VerificationConfig:
    use_shadow: bool = True
    use_shadow_length: bool = True
    use_shadow_heading: bool = True


@dataclass(frozen=True)
class FusionConfig:
    retrieval_temperature: float = 0.2
    retrieval_score_norm: str = "none"
    source_prior_retrieval: float = 1.0
    source_prior_geoclip: float = 1.0
    source_prior_exif: float = 1.0
    use_spatial_consensus: bool = True
    spatial_sigma_km: float = 2.0
    spatial_consensus_weight: float = 1.0
    shadow_sigma_deg: float = 20.0
    terrain_sigma: float = 100.0
    use_shadow: bool = True
    use_terrain: bool = False
    credible_mass: float = 0.9
    min_credible_candidates: int = 2
    use_top_cluster_for_stats: bool = True
    credible_cluster_radius_km: float = 500.0
    min_credible_cluster_weight: float = 0.35
    top_k: int = 5


@dataclass(frozen=True)
class ScoreConfig:
    detection_weight: float = 0.4
    geo_weight: float = 0.4
    shadow_weight: float = 0.1
    topo_weight: float = 0.1
    detection_top_k: int = 3


@dataclass(frozen=True)
class HeimdallConfig:
    detector: DetectorConfig
    geolocator: GeoConfig
    fusion: FusionConfig
    score: ScoreConfig
    verification: VerificationConfig


def load_config(path: str) -> HeimdallConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    det = raw.get("detector", {})
    geo = raw.get("geolocator", {})
    fusion = raw.get("fusion", {})
    score = raw.get("score", {})
    ver = raw.get("verification", {})
    return HeimdallConfig(
        detector=DetectorConfig(
            weights_path=det.get("weights_path"),
            min_confidence=det.get("min_confidence", 0.25),
            nms_iou=det.get("nms_iou", 0.5),
            max_detections=det.get("max_detections", 100),
            min_area_px=det.get("min_area_px", 16.0),
            class_agnostic_nms=det.get("class_agnostic_nms", False),
            use_tta=det.get("use_tta", False),
            use_sidecar=det.get("use_sidecar", True),
            use_classic=det.get("use_classic", False),
            imgsz=det.get("imgsz", 1280),
        ),
        geolocator=GeoConfig(
            model_path=geo.get("model_path"),
            model_id=geo.get("model_id"),
            model_cache_dir=geo.get("model_cache_dir"),
            encoder_name=geo.get("encoder_name"),
            use_sidecar=geo.get("use_sidecar", True),
            use_exif=geo.get("use_exif", True),
            top_n=geo.get("top_n", 5),
            geospot_score_scale=geo.get("geospot_score_scale", 1.0),
            retrieval_index_path=geo.get("retrieval_index_path"),
            retrieval_model_id=geo.get("retrieval_model_id"),
            retrieval_top_k=geo.get("retrieval_top_k", 10),
            retrieval_min_score=geo.get("retrieval_min_score", 0.2),
            candidate_dedupe_radius_m=geo.get("candidate_dedupe_radius_m", 300.0),
            candidate_max_results=geo.get("candidate_max_results", 80),
        ),
        fusion=FusionConfig(
            retrieval_temperature=fusion.get("retrieval_temperature", 0.2),
            retrieval_score_norm=fusion.get("retrieval_score_norm", "none"),
            source_prior_retrieval=fusion.get("source_prior_retrieval", 1.0),
            source_prior_geoclip=fusion.get("source_prior_geoclip", 1.0),
            source_prior_exif=fusion.get("source_prior_exif", 1.0),
            use_spatial_consensus=fusion.get("use_spatial_consensus", True),
            spatial_sigma_km=fusion.get("spatial_sigma_km", 2.0),
            spatial_consensus_weight=fusion.get("spatial_consensus_weight", 1.0),
            shadow_sigma_deg=fusion.get("shadow_sigma_deg", 20.0),
            terrain_sigma=fusion.get("terrain_sigma", 100.0),
            use_shadow=fusion.get("use_shadow", True),
            use_terrain=fusion.get("use_terrain", False),
            credible_mass=fusion.get("credible_mass", 0.9),
            min_credible_candidates=fusion.get("min_credible_candidates", 2),
            use_top_cluster_for_stats=fusion.get("use_top_cluster_for_stats", True),
            credible_cluster_radius_km=fusion.get("credible_cluster_radius_km", 500.0),
            min_credible_cluster_weight=fusion.get("min_credible_cluster_weight", 0.35),
            top_k=fusion.get("top_k", 5),
        ),
        score=ScoreConfig(
            detection_weight=score.get("detection_weight", 0.4),
            geo_weight=score.get("geo_weight", 0.4),
            shadow_weight=score.get("shadow_weight", 0.1),
            topo_weight=score.get("topo_weight", 0.1),
            detection_top_k=score.get("detection_top_k", 3),
        ),
        verification=VerificationConfig(
            use_shadow=ver.get("use_shadow", True),
            use_shadow_length=ver.get("use_shadow_length", True),
            use_shadow_heading=ver.get("use_shadow_heading", True),
        ),
    )
