"""Combine multiple candidate providers."""
from __future__ import annotations

from typing import Iterable, List, Optional

from src.core.logic.types import GeoCandidate


class MultiCandidateProvider:
    def __init__(self, providers: Iterable[object]) -> None:
        self.providers = [p for p in providers if p is not None]
        self.last_error: Optional[str] = None

    def candidates(self, image_path: str) -> List[GeoCandidate]:
        results: List[GeoCandidate] = []
        errors = []
        for provider in self.providers:
            try:
                results.extend(provider.candidates(image_path))
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(str(exc))
                continue
            err = getattr(provider, "last_error", None)
            if err:
                errors.append(str(err))
        self.last_error = "; ".join(errors) if errors else None
        return results
