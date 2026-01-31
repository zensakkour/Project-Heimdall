"""
Embedding-based retrieval for geo candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from src.core.logic.types import GeoCandidate

try:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CLIPModel = None
    CLIPProcessor = None


@dataclass
class RetrievalIndex:
    embeddings: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    ids: np.ndarray
    paths: np.ndarray


class ClipEmbedder:
    def __init__(self, model_id: str, device: str) -> None:
        if CLIPModel is None or CLIPProcessor is None:
            raise RuntimeError("transformers not available")
        self.device = device
        self.model_id = model_id
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        feats = _extract_tensor(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


def _extract_tensor(output):
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state.mean(dim=1)
    if isinstance(output, dict):
        if "pooler_output" in output and output["pooler_output"] is not None:
            return output["pooler_output"]
        if "image_embeds" in output and output["image_embeds"] is not None:
            return output["image_embeds"]
        if "last_hidden_state" in output and output["last_hidden_state"] is not None:
            return output["last_hidden_state"].mean(dim=1)
    return output


def load_index(index_path: Path) -> RetrievalIndex:
    data = np.load(index_path, allow_pickle=True)
    return RetrievalIndex(
        embeddings=data["embeddings"],
        latitudes=data["latitudes"],
        longitudes=data["longitudes"],
        ids=data["ids"],
        paths=data["paths"],
    )


class GeoRetrievalProvider:
    def __init__(
        self,
        index_path: Optional[str],
        model_id: str = "openai/clip-vit-large-patch14",
        top_k: int = 10,
        min_score: float = 0.2,
    ) -> None:
        self.index_path = Path(index_path) if index_path else None
        self.model_id = model_id
        self.top_k = top_k
        self.min_score = min_score
        self._index: Optional[RetrievalIndex] = None
        self._embedder: Optional[ClipEmbedder] = None
        self.last_error: Optional[str] = None

    def candidates(self, image_path: str) -> List[GeoCandidate]:
        if self.index_path is None:
            self.last_error = "index_not_configured"
            return []
        if not self.index_path.exists():
            self.last_error = "index_not_found"
            return []
        try:
            index = self._ensure_index()
            embedder = self._ensure_embedder()
        except Exception as exc:
            self.last_error = str(exc)
            return []

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            query = embedder.embed(image)
            scores = (index.embeddings @ query.T).squeeze(1)
            top_idx = np.argsort(scores)[::-1][: self.top_k]
            results: List[GeoCandidate] = []
            for idx in top_idx:
                score = float(scores[idx])
                if score < self.min_score:
                    continue
                results.append(
                    GeoCandidate(
                        latitude=float(index.latitudes[idx]),
                        longitude=float(index.longitudes[idx]),
                        retrieval_score=score,
                        match_id=str(index.ids[idx]) if index.ids[idx] is not None else None,
                    )
                )
            self.last_error = None
            return results
        except Exception as exc:
            self.last_error = str(exc)
            return []

    def _ensure_index(self) -> RetrievalIndex:
        if self._index is None:
            self._index = load_index(self.index_path)
        return self._index

    def _ensure_embedder(self) -> ClipEmbedder:
        if self._embedder is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedder = ClipEmbedder(self.model_id, device)
        return self._embedder
