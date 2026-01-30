"""
Core pipeline wiring (stub): detection -> geolocation -> verification -> score.
"""
from __future__ import annotations

from typing import List, Optional

from core.detection import Detector
from core.geo import GeoLocator

from .config import ScoreConfig, VerificationConfig
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
        score_config: Optional[ScoreConfig] = None,
        verification_config: Optional[VerificationConfig] = None,
    ) -> None:
        self.detector = detector
        self.geolocator = geolocator
        self.score_config = score_config
        self.verification_config = verification_config

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

    def run(self, image_path: str) -> Assessment:
        detections = self.detect(image_path)
        detections = enrich_detections_with_shadows(image_path, detections)
        geo = self.geolocate(image_path)
        verification = self.verify(image_path, geo, detections)
        score = compute_score(detections, geo, verification, self.score_config)
        return Assessment(
            detections=detections,
            geo=geo,
            verification=verification,
            score=score,
        )
