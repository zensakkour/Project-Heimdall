"""
Detector protocol for pipeline typing.
"""
from __future__ import annotations

from typing import List, Protocol

from core.logic.types import Detection


class Detector(Protocol):
    def predict(self, image_path: str) -> List[Detection]:
        ...
