"""
Embedding-based retrieval for geo candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from src.core.logic.types import GeoCandidate

try:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CLIPModel = None
    CLIPProcessor = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


@dataclass
class RetrievalIndex:
    embeddings: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    ids: np.ndarray
    paths: np.ndarray


class ClipEmbedder:
    def __init__(self, model_id: str, device: str) -> None:
        if CLIPModel is None or CLIPProcessor is None or torch is None:
            raise RuntimeError("transformers not available")
        self.device = device
        self.model_id = model_id
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    def embed(self, image: Image.Image) -> np.ndarray:
        if torch is None:
            raise RuntimeError("torch_not_available")
        with torch.no_grad():
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
    required = ("embeddings", "latitudes", "longitudes", "ids", "paths")
    with np.load(index_path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"index_missing_keys:{','.join(missing)}")

        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        latitudes = np.asarray(data["latitudes"], dtype=np.float64)
        longitudes = np.asarray(data["longitudes"], dtype=np.float64)
        ids = np.asarray(data["ids"])
        paths = np.asarray(data["paths"])

    if embeddings.ndim != 2:
        raise ValueError("index_embeddings_must_be_2d")
    if embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0:
        raise ValueError("index_embeddings_empty")

    count = embeddings.shape[0]
    if latitudes.shape[0] != count or longitudes.shape[0] != count or ids.shape[0] != count or paths.shape[0] != count:
        raise ValueError("index_array_length_mismatch")

    if not np.isfinite(embeddings).all():
        raise ValueError("index_embeddings_not_finite")
    if not np.isfinite(latitudes).all() or not np.isfinite(longitudes).all():
        raise ValueError("index_coordinates_not_finite")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    return RetrievalIndex(
        embeddings=embeddings,
        latitudes=latitudes,
        longitudes=longitudes,
        ids=ids,
        paths=paths,
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
            if query.ndim == 1:
                query = query.reshape(1, -1)
            query = query.astype(np.float32, copy=False)
            if query.shape[1] != index.embeddings.shape[1]:
                raise ValueError("query_embedding_dim_mismatch")
            query_norm = np.linalg.norm(query, axis=1, keepdims=True)
            query = query / np.clip(query_norm, 1e-12, None)
            scores = (index.embeddings @ query.T).squeeze(1)
            if scores.size == 0:
                self.last_error = "index_empty"
                return []
            top_k = min(max(1, int(self.top_k)), int(scores.size))
            if top_k == scores.size:
                top_idx = np.argsort(scores)[::-1]
            else:
                unsorted_idx = np.argpartition(scores, -top_k)[-top_k:]
                top_idx = unsorted_idx[np.argsort(scores[unsorted_idx])[::-1]]
            results: List[GeoCandidate] = []
            for idx in top_idx:
                score = float(scores[idx])
                if score < self.min_score:
                    continue
                lat = float(index.latitudes[idx])
                lon = float(index.longitudes[idx])
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    continue
                results.append(
                    GeoCandidate(
                        latitude=lat,
                        longitude=lon,
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
            device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
            self._embedder = ClipEmbedder(self.model_id, device)
        return self._embedder
