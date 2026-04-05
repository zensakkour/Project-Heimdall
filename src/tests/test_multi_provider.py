"""
Tests for multi-provider candidate aggregation.
"""
from __future__ import annotations

from src.core.geo.multi_provider import MultiCandidateProvider
from src.core.logic.types import GeoCandidate


class _StaticProvider:
    def __init__(self, values=None, error=None, raises=None) -> None:
        self._values = values or []
        self.last_error = error
        self._raises = raises

    def candidates(self, image_path: str):
        del image_path
        if self._raises is not None:
            raise RuntimeError(self._raises)
        return list(self._values)


def test_multi_provider_dedupes_nearby_candidates() -> None:
    provider_a = _StaticProvider(
        values=[
            GeoCandidate(latitude=48.856600, longitude=2.352200, retrieval_score=0.62, match_id="retrieval:1"),
            GeoCandidate(latitude=40.712800, longitude=-74.006000, retrieval_score=0.51, match_id="retrieval:2"),
        ]
    )
    provider_b = _StaticProvider(
        values=[
            GeoCandidate(latitude=48.856605, longitude=2.352210, retrieval_score=0.68, match_id="geoclip"),
            GeoCandidate(latitude=120.0, longitude=0.0, retrieval_score=0.99, match_id="invalid"),
        ]
    )
    provider = MultiCandidateProvider(
        [provider_a, provider_b],
        dedupe_radius_m=50.0,
        max_candidates=10,
    )

    out = provider.candidates("dummy.jpg")

    assert len(out) == 2
    assert out[0].retrieval_score > 0.8
    assert abs(out[0].latitude - 48.856602) < 1e-4
    assert abs(out[0].longitude - 2.352205) < 1e-4
    assert out[1].match_id == "retrieval:2"


def test_multi_provider_collects_provider_errors() -> None:
    provider = MultiCandidateProvider(
        [
            _StaticProvider(raises="provider_failed"),
            _StaticProvider(values=[], error="index_not_found"),
        ]
    )

    out = provider.candidates("dummy.jpg")

    assert out == []
    assert provider.last_error is not None
    assert "provider_failed" in provider.last_error
    assert "index_not_found" in provider.last_error


def test_multi_provider_source_balance_can_prevent_single_source_domination() -> None:
    provider_a = _StaticProvider(
        values=[
            GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.99, match_id="retrieval:a:1"),
            GeoCandidate(latitude=40.7128, longitude=-74.0060, retrieval_score=0.98, match_id="retrieval:a:2"),
        ]
    )
    provider_b = _StaticProvider(
        values=[
            GeoCandidate(latitude=35.6764, longitude=139.6500, retrieval_score=0.97, match_id="geoclip"),
        ]
    )
    plain = MultiCandidateProvider(
        [provider_a, provider_b],
        dedupe_radius_m=0.0,
        source_balance_beta=0.0,
        max_candidates=2,
    )
    out_plain = plain.candidates("dummy.jpg")
    assert [cand.match_id for cand in out_plain] == ["retrieval:a:1", "retrieval:a:2"]

    balanced = MultiCandidateProvider(
        [provider_a, provider_b],
        dedupe_radius_m=0.0,
        source_balance_beta=0.8,
        max_candidates=2,
    )
    out_balanced = balanced.candidates("dummy.jpg")
    ids = {cand.match_id for cand in out_balanced}
    assert "retrieval:a:1" in ids
    assert "geoclip" in ids
