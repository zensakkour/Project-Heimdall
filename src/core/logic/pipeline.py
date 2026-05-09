"""
Core pipeline wiring (stub): detection -> geolocation -> verification -> score.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from src.core.detection import Detector
from src.core.geo import CandidateProvider, GeoLocator

from .config import FusionConfig, ScoreConfig, VerificationConfig
from .fusion import fuse_candidates
from .score import compute_score
from .shadow_extract import enrich_detections_with_shadows
from .types import Assessment, Detection, GeoEstimate, Verification
from .verify import run_verification


class HeimdallPipeline:
    """Pipeline skeleton. Plug in real models later."""

    def __init__(
        self,
        detector: Optional[Detector] = None,
        geolocator: Optional[GeoLocator] = None,
        candidate_provider: Optional[CandidateProvider] = None,
        fusion_config: Optional[FusionConfig] = None,
        score_config: Optional[ScoreConfig] = None,
        verification_config: Optional[VerificationConfig] = None,
        detector_backend: Optional[str] = None,
    ) -> None:
        self.detector = detector
        self.geolocator = geolocator
        self.candidate_provider = candidate_provider
        self.fusion_config = fusion_config
        self.score_config = score_config
        self.verification_config = verification_config
        self.detector_backend = detector_backend

    def detect(self, image_path: str) -> List[Detection]:
        if self.detector is None:
            return []
        return self.detector.predict(image_path)

    def geolocate(self, image_path: str) -> Optional[GeoEstimate]:
        if self.geolocator is None:
            return None
        return self.geolocator.predict(image_path)

    def verify(
        self, image_path: str, geo: Optional[GeoEstimate], detections: List[Detection]
    ) -> Optional[Verification]:
        return run_verification(image_path, geo, detections, self.verification_config)

    def run(self, image_path: str, capture_time: Optional[datetime] = None) -> Assessment:
        detections = self.detect(image_path)
        detections = enrich_detections_with_shadows(image_path, detections)
        geo = self.geolocate(image_path)
        verification = self.verify(image_path, geo, detections)
        candidates = []
        fusion = None
        if self.candidate_provider is not None:
            candidates = self.candidate_provider.candidates(image_path)
            if candidates:
                fusion = fuse_candidates(
                    image_path=image_path,
                    candidates=candidates,
                    detections=detections,
                    config=self.fusion_config,
                    capture_time=capture_time,
                )
                if geo is None and fusion is not None and fusion.candidates:
                    top = fusion.candidates[0]
                    match_id = top.candidate.match_id
                    geo = GeoEstimate(
                        latitude=fusion.mean_latitude,
                        longitude=fusion.mean_longitude,
                        confidence=max(0.05, min(0.95, top.posterior_weight)),
                        landmarks=None if match_id is None else [match_id],
                        uncertainty_m=fusion.uncertainty_radius_m,
                    )
        score = compute_score(detections, geo, verification, self.score_config)
        return Assessment(
            detections=detections,
            geo=geo,
            verification=verification,
            score=score,
            candidates=candidates,
            fusion=fusion,
            backend=self.detector_backend,
        )

