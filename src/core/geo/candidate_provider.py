from __future__ import annotations

from typing import List, Protocol

from src.core.logic.types import GeoCandidate


class CandidateProvider(Protocol):
    def candidates(self, image_path: str) -> List[GeoCandidate]:
        ...


