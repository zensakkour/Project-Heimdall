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


@dataclass(frozen=True)
class VerificationConfig:
    use_shadow: bool = True
    use_shadow_length: bool = True
    use_shadow_heading: bool = True


@dataclass(frozen=True)
class FusionConfig:
    retrieval_temperature: float = 0.2
    shadow_sigma_deg: float = 20.0
    terrain_sigma: float = 100.0
    use_shadow: bool = True
    use_terrain: bool = False
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
        ),
        fusion=FusionConfig(
            retrieval_temperature=fusion.get("retrieval_temperature", 0.2),
            shadow_sigma_deg=fusion.get("shadow_sigma_deg", 20.0),
            terrain_sigma=fusion.get("terrain_sigma", 100.0),
            use_shadow=fusion.get("use_shadow", True),
            use_terrain=fusion.get("use_terrain", False),
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
