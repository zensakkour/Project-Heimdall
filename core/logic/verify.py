"""
Shadow/topographic verification stubs.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .astro import sun_position, tan_safe
from .image_meta import extract_capture_time
from .config import VerificationConfig
from .types import Detection, GeoEstimate, Verification


def verify_shadow(
    image_path: str,
    geo: Optional[GeoEstimate],
    detections: Sequence[Detection],
    config: VerificationConfig,
) -> Tuple[bool, str]:
    if not config.use_shadow:
        return True, "shadow:disabled"
    if geo is None:
        return False, "shadow:missing_geo"
    captured = extract_capture_time(image_path)
    if captured is None:
        return False, "shadow:missing_time"
    azimuth, elevation = sun_position(captured, geo.latitude, geo.longitude)
    if elevation <= 0.0:
        return False, f"shadow:night el={elevation:.1f}"
    if elevation < 5.0:
        return False, f"shadow:low_sun el={elevation:.1f}"

    expected_shadow = (azimuth + 180.0) % 360.0
    observed = [d for d in detections if d.shadow_azimuth_deg is not None]
    if not observed:
        return True, f"shadow:time_only az={azimuth:.1f} el={elevation:.1f}"

    diffs = []
    heading_conflict = False
    for det in observed:
        diff = _angular_diff(expected_shadow, float(det.shadow_azimuth_deg))
        diffs.append(diff)
        if config.use_shadow_heading and det.heading_deg is not None:
            if _angular_diff(float(det.heading_deg), float(det.shadow_azimuth_deg)) < 12.0:
                heading_conflict = True
        if config.use_shadow_length and det.shadow_length_ratio is not None:
            if not _shadow_length_ok(elevation, float(det.shadow_length_ratio)):
                return False, f"shadow:length_mismatch el={elevation:.1f}"

    mean_diff = sum(diffs) / len(diffs)
    if heading_conflict:
        return False, f"shadow:conflicts_heading diff={mean_diff:.1f}"
    if mean_diff <= 25.0:
        return True, f"shadow:ok diff={mean_diff:.1f} az={azimuth:.1f}"
    return False, f"shadow:diff={mean_diff:.1f} az={azimuth:.1f}"


def verify_topography(image_path: str, geo: Optional[GeoEstimate]) -> Tuple[bool, str]:
    if geo is None:
        return False, "topo:missing_geo"
    if geo.confidence < 0.5:
        return False, f"topo:low_conf {geo.confidence:.2f}"
    return True, "topo:stub_confidence_ok"


def run_verification(
    image_path: str,
    geo: Optional[GeoEstimate],
    detections: Sequence[Detection],
    config: Optional[VerificationConfig] = None,
) -> Optional[Verification]:
    if geo is None:
        return None
    cfg = config or VerificationConfig()
    shadow_ok, shadow_note = verify_shadow(image_path, geo, detections, cfg)
    topo_ok, topo_note = verify_topography(image_path, geo)
    notes = "; ".join([shadow_note, topo_note])
    return Verification(shadow_ok=shadow_ok, topo_ok=topo_ok, notes=notes)


def _angular_diff(a: float, b: float) -> float:
    delta = abs((a - b) % 360.0)
    return min(delta, 360.0 - delta)


def _shadow_length_ok(elevation_deg: float, ratio: float) -> bool:
    if elevation_deg <= 0.0 or ratio <= 0.0:
        return False
    elev_rad = elevation_deg * 3.141592653589793 / 180.0
    expected = 1.0 / max(0.01, tan_safe(elev_rad))
    low = expected / 3.0
    high = expected * 3.0
    return low <= ratio <= high
