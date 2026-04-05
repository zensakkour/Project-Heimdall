"""
Embedding-based retrieval for geo candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.core.logic.types import GeoCandidate

try:
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel, CLIPProcessor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AutoImageProcessor = None
    AutoModel = None
    AutoProcessor = None
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
    model_id: Optional[str] = None


@dataclass(frozen=True)
class LoadedRetrievalIndex:
    source: str
    path: Path
    model_id: str
    index: RetrievalIndex


class ClipEmbedder:
    def __init__(self, model_id: str, device: str) -> None:
        if torch is None:
            raise RuntimeError("transformers not available")
        self.device = device
        self.model_id = model_id
        self.processor = None
        self.model = None

        # Prefer native CLIP loader when possible for compatibility with existing indices.
        if CLIPModel is not None and CLIPProcessor is not None:
            try:
                self.processor = CLIPProcessor.from_pretrained(model_id)
                self.model = CLIPModel.from_pretrained(model_id)
            except Exception:
                self.processor = None
                self.model = None

        # Fallback enables stronger non-CLIP image backbones (e.g., SigLIP family).
        if self.model is None or self.processor is None:
            if AutoModel is None:
                raise RuntimeError("transformers_not_available")
            if AutoImageProcessor is not None:
                try:
                    self.processor = AutoImageProcessor.from_pretrained(model_id)
                except Exception:
                    self.processor = None
            if self.processor is None:
                if AutoProcessor is None:
                    raise RuntimeError("image_processor_not_available")
                self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id)

        self.model.to(device)
        self.model.eval()

    def embed(self, image: Image.Image) -> np.ndarray:
        if torch is None:
            raise RuntimeError("torch_not_available")
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}
            if hasattr(self.model, "get_image_features"):
                feats = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)
                feats = _extract_tensor(outputs)
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
    try:
        with np.load(index_path, allow_pickle=False) as data:
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"index_missing_keys:{','.join(missing)}")

            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            latitudes = np.asarray(data["latitudes"], dtype=np.float64)
            longitudes = np.asarray(data["longitudes"], dtype=np.float64)
            ids = np.asarray(data["ids"])
            paths = np.asarray(data["paths"])
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise
        with np.load(index_path, allow_pickle=True) as data:
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

    model_id = _parse_index_model_id(index_path)

    return RetrievalIndex(
        embeddings=embeddings,
        latitudes=latitudes,
        longitudes=longitudes,
        ids=ids,
        paths=paths,
        model_id=model_id,
    )


def _parse_index_model_id(index_path: Path) -> Optional[str]:
    try:
        with np.load(index_path, allow_pickle=False) as data:
            raw = data.get("model_id")
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            return None
        with np.load(index_path, allow_pickle=True) as data:
            raw = data.get("model_id")
    except Exception:
        return None

    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        if raw.ndim == 0:
            text = str(raw.item()).strip()
        elif raw.size > 0:
            text = str(raw.reshape(-1)[0]).strip()
        else:
            text = ""
    else:
        text = str(raw).strip()
    return text or None


def _normalize_index_paths(index_path: Optional[str], index_paths: Optional[Sequence[str]]) -> List[Path]:
    ordered: List[Path] = []
    seen = set()
    raw_paths: List[str] = []
    if index_path:
        raw_paths.append(str(index_path))
    if index_paths:
        raw_paths.extend(str(item) for item in index_paths if item)
    for raw in raw_paths:
        text = raw.strip()
        if not text:
            continue
        key = str(Path(text))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(Path(text))
    return ordered


def _normalize_index_weights(weights: Optional[Sequence[float]], n_indices: int) -> List[float]:
    if n_indices <= 0:
        return []
    out = [1.0 for _ in range(n_indices)]
    if not weights:
        return out
    for idx in range(min(n_indices, len(weights))):
        raw = weights[idx]
        if not isinstance(raw, (int, float)):
            continue
        val = float(raw)
        if not math.isfinite(val) or val <= 0.0:
            continue
        out[idx] = val
    return out


def _normalize_index_model_ids(
    model_ids: Optional[Sequence[str]],
    *,
    n_indices: int,
    default_model_id: str,
) -> List[str]:
    out = [default_model_id for _ in range(n_indices)]
    if not model_ids:
        return out
    for idx in range(min(n_indices, len(model_ids))):
        raw = model_ids[idx]
        text = str(raw).strip() if raw is not None else ""
        if text:
            out[idx] = text
    return out


def _normalize_index_score_norm(mode: object) -> str:
    text = str(mode).strip().lower()
    if text not in {"auto", "none", "minmax", "zscore_sigmoid", "rank_exp"}:
        return "auto"
    return text


def _normalize_scores_for_source(scores: np.ndarray, mode: str) -> np.ndarray:
    reduce_mode = _normalize_index_score_norm(mode)
    if reduce_mode == "none":
        return np.asarray(scores, dtype=np.float32)
    if scores.size <= 0:
        return np.asarray(scores, dtype=np.float32)

    if reduce_mode == "minmax":
        lo = float(np.min(scores))
        hi = float(np.max(scores))
        span = hi - lo
        if span < 1e-9:
            return np.full(scores.shape[0], 0.5, dtype=np.float32)
        return np.asarray((scores - lo) / span, dtype=np.float32)

    if reduce_mode == "zscore_sigmoid":
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if std < 1e-6:
            return np.full(scores.shape[0], 0.5, dtype=np.float32)
        z = (scores - mean) / std
        out = 1.0 / (1.0 + np.exp(-z))
        return np.asarray(out, dtype=np.float32)

    if reduce_mode == "rank_exp":
        order = np.argsort(scores)[::-1]
        tau = max(1.0, float(scores.size) / 3.0)
        out = np.zeros(scores.shape[0], dtype=np.float32)
        for rank, idx in enumerate(order):
            out[int(idx)] = float(math.exp(-float(rank) / tau))
        return out

    return np.asarray(scores, dtype=np.float32)


def _index_source_name(path: Path, index: int) -> str:
    name = path.stem.strip().lower().replace(" ", "_")
    if not name:
        name = f"index_{index + 1}"
    return name


def _format_retrieval_match_id(source: str, raw_id: object) -> str:
    if raw_id is None:
        return f"retrieval:{source}"
    text = str(raw_id).strip()
    if not text:
        return f"retrieval:{source}"
    if text.startswith("retrieval:"):
        return text
    return f"retrieval:{source}:{text}"


def _top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if top_k >= scores.size:
        return np.argsort(scores)[::-1]
    unsorted_idx = np.argpartition(scores, -top_k)[-top_k:]
    return unsorted_idx[np.argsort(scores[unsorted_idx])[::-1]]


def _collect_weighted_ranked_candidates(
    *,
    loaded_indices: Sequence[LoadedRetrievalIndex],
    index_weights: Sequence[float],
    query_mats_by_model: dict[str, np.ndarray],
    global_top_k: int,
    per_index_top_k: int,
    reduce_mode: str,
    index_score_norm: str,
) -> Tuple[List[GeoCandidate], int]:
    merged: List[GeoCandidate] = []
    fallback_model_id = next(iter(query_mats_by_model.keys()), "")
    effective_norm = _normalize_index_score_norm(index_score_norm)
    if effective_norm == "auto":
        effective_norm = "zscore_sigmoid" if len(loaded_indices) > 1 else "none"
    for idx, loaded in enumerate(loaded_indices):
        index = loaded.index
        model_id = getattr(loaded, "model_id", "") or fallback_model_id
        query_mat = query_mats_by_model.get(model_id)
        if query_mat is None:
            raise ValueError(f"query_embedding_missing_for_model:{model_id}")
        if query_mat.shape[1] != index.embeddings.shape[1]:
            raise ValueError(f"query_embedding_dim_mismatch:{model_id}")
        scores_aug = index.embeddings @ query_mat.T
        scores_raw = _aggregate_tta_scores(scores_aug, mode=reduce_mode)
        scores = _normalize_scores_for_source(scores_raw, mode=effective_norm)
        if scores.size <= 0:
            continue
        per_source_top = min(
            max(1, int(per_index_top_k) if per_index_top_k > 0 else int(global_top_k)),
            int(scores.size),
        )
        weight = float(index_weights[idx]) if idx < len(index_weights) else 1.0
        for item_idx in _top_indices(scores, per_source_top):
            lat = float(index.latitudes[item_idx])
            lon = float(index.longitudes[item_idx])
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            raw_score = max(-1.0, min(1.0, float(scores[item_idx])))
            weighted = max(-1.0, min(1.0, raw_score * weight))
            merged.append(
                GeoCandidate(
                    latitude=lat,
                    longitude=lon,
                    retrieval_score=weighted,
                    match_id=_format_retrieval_match_id(loaded.source, index.ids[item_idx]),
                )
            )

    if not merged:
        return [], 0
    merged.sort(key=lambda item: item.retrieval_score, reverse=True)
    deduped: List[GeoCandidate] = []
    seen = set()
    for cand in merged:
        key = (round(cand.latitude, 6), round(cand.longitude, 6), cand.match_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    top_k = min(max(1, int(global_top_k)), len(deduped))
    return deduped, top_k


def _candidate_source_key(match_id: Optional[str]) -> str:
    text = (match_id or "").strip().lower()
    if text.startswith("retrieval:"):
        parts = text.split(":")
        if len(parts) >= 3 and parts[1].strip():
            return f"retrieval:{parts[1].strip()}"
        return "retrieval"
    if text.startswith("geoclip"):
        return "geoclip"
    if text.startswith("exif:"):
        return "exif"
    return "unknown"


def _select_source_balanced_candidates(
    ranked: Sequence[GeoCandidate],
    *,
    top_k: int,
    balance_beta: float,
) -> List[GeoCandidate]:
    if not ranked:
        return []
    k = min(max(1, int(top_k)), len(ranked))
    ordered = sorted(ranked, key=lambda cand: cand.retrieval_score, reverse=True)
    beta = max(0.0, float(balance_beta))
    if beta <= 1e-9:
        return ordered[:k]

    selected: List[GeoCandidate] = []
    remaining = list(ordered)
    per_source_counts = {}
    while remaining and len(selected) < k:
        best_idx = 0
        best_value = float("-inf")
        for idx, cand in enumerate(remaining):
            source = _candidate_source_key(cand.match_id)
            count = float(per_source_counts.get(source, 0))
            adjusted = float(cand.retrieval_score) / (1.0 + beta * count)
            if adjusted > best_value:
                best_value = adjusted
                best_idx = idx
        pick = remaining.pop(best_idx)
        selected.append(pick)
        source = _candidate_source_key(pick.match_id)
        per_source_counts[source] = int(per_source_counts.get(source, 0)) + 1

    selected.sort(key=lambda cand: cand.retrieval_score, reverse=True)
    return selected[:k]


class GeoRetrievalProvider:
    def __init__(
        self,
        index_path: Optional[str],
        index_paths: Optional[Sequence[str]] = None,
        index_weights: Optional[Sequence[float]] = None,
        index_model_ids: Optional[Sequence[str]] = None,
        model_id: str = "openai/clip-vit-large-patch14",
        top_k: int = 10,
        per_index_top_k: int = 0,
        index_score_norm: str = "auto",
        source_balance_beta: float = 0.0,
        min_score: float = 0.2,
        min_keep_topk: int = 0,
        diversity_radius_km: float = 0.0,
        diversity_lambda: float = 1.0,
        diversity_min_keep: int = 1,
        locality_radius_km: float = 0.0,
        locality_weight: float = 0.0,
        query_tta_degrees: Optional[Sequence[float]] = None,
        query_tta_reduce: str = "mean",
    ) -> None:
        self.index_paths = _normalize_index_paths(index_path=index_path, index_paths=index_paths)
        self.index_weights = _normalize_index_weights(index_weights, len(self.index_paths))
        self.model_id = model_id
        self.index_model_ids = _normalize_index_model_ids(
            index_model_ids,
            n_indices=len(self.index_paths),
            default_model_id=self.model_id,
        )
        self.top_k = top_k
        self.per_index_top_k = max(0, int(per_index_top_k))
        self.index_score_norm = _normalize_index_score_norm(index_score_norm)
        self.source_balance_beta = max(0.0, float(source_balance_beta))
        self.min_score = min_score
        self.min_keep_topk = max(0, int(min_keep_topk))
        self.diversity_radius_km = max(0.0, float(diversity_radius_km))
        self.diversity_lambda = min(1.0, max(0.0, float(diversity_lambda)))
        self.diversity_min_keep = max(0, int(diversity_min_keep))
        self.locality_radius_km = max(0.0, float(locality_radius_km))
        self.locality_weight = max(0.0, float(locality_weight))
        self.query_tta_degrees = _normalize_tta_degrees(query_tta_degrees)
        reduce_mode = str(query_tta_reduce).lower()
        if reduce_mode not in {"mean", "max", "rrf"}:
            reduce_mode = "mean"
        self.query_tta_reduce = reduce_mode
        self._indices: Optional[List[LoadedRetrievalIndex]] = None
        self._embedder: Optional[ClipEmbedder] = None
        self._embedders_by_model: dict[str, ClipEmbedder] = {}
        self.last_error: Optional[str] = None

    def candidates(self, image_path: str) -> List[GeoCandidate]:
        if not self.index_paths:
            self.last_error = "index_not_configured"
            return []
        existing_paths = [path for path in self.index_paths if path.exists()]
        if not existing_paths:
            self.last_error = "index_not_found"
            return []
        try:
            loaded_indices = self._ensure_indices()
            embedder = self._ensure_embedder()
        except Exception as exc:
            self.last_error = str(exc)
            return []

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            query_mats_by_model: dict[str, np.ndarray] = {}
            model_ids = list(
                dict.fromkeys((getattr(loaded, "model_id", "") or self.model_id) for loaded in loaded_indices)
            )
            for model_id in model_ids:
                if model_id == self.model_id:
                    query_mats_by_model[model_id] = _query_embeddings(embedder, image, self.query_tta_degrees)
                    continue
                other = self._ensure_embedder_for_model(model_id)
                query_mats_by_model[model_id] = _query_embeddings(other, image, self.query_tta_degrees)
            ranked, top_k = _collect_weighted_ranked_candidates(
                loaded_indices=loaded_indices,
                index_weights=self.index_weights,
                query_mats_by_model=query_mats_by_model,
                global_top_k=self.top_k,
                per_index_top_k=self.per_index_top_k,
                reduce_mode=self.query_tta_reduce,
                index_score_norm=self.index_score_norm,
            )
            if not ranked:
                self.last_error = "index_empty"
                return []
            filtered = [item for item in ranked if item.retrieval_score >= self.min_score]
            min_keep = min(max(0, int(self.min_keep_topk)), top_k)
            if len(filtered) < min_keep:
                existing = {
                    (cand.latitude, cand.longitude, cand.retrieval_score, cand.match_id)
                    for cand in filtered
                }
                for cand in ranked:
                    key = (cand.latitude, cand.longitude, cand.retrieval_score, cand.match_id)
                    if key in existing:
                        continue
                    filtered.append(cand)
                    existing.add(key)
                    if len(filtered) >= min_keep:
                        break
            ranked = list(filtered)
            ranked = _apply_locality_rerank(
                ranked,
                radius_km=self.locality_radius_km,
                weight=self.locality_weight,
            )
            ranked = _select_source_balanced_candidates(
                ranked,
                top_k=top_k,
                balance_beta=self.source_balance_beta,
            )
            results = _select_diverse_geo_candidates(
                ranked,
                top_k=top_k,
                radius_km=self.diversity_radius_km,
                diversity_lambda=self.diversity_lambda,
                min_keep=self.diversity_min_keep,
            )
            self.last_error = None
            return results
        except Exception as exc:
            self.last_error = str(exc)
            return []

    def _ensure_indices(self) -> List[LoadedRetrievalIndex]:
        if self._indices is not None:
            return self._indices
        loaded: List[LoadedRetrievalIndex] = []
        for idx, path in enumerate(self.index_paths):
            if not path.exists():
                continue
            index = load_index(path)
            source = _index_source_name(path, idx)
            configured_model = self.index_model_ids[idx] if idx < len(self.index_model_ids) else ""
            model_id = configured_model or index.model_id or self.model_id
            loaded.append(LoadedRetrievalIndex(source=source, path=path, model_id=model_id, index=index))
        if not loaded:
            raise FileNotFoundError("index_not_found")
        self._indices = loaded
        return self._indices

    def _ensure_embedder(self) -> ClipEmbedder:
        if self._embedder is None:
            device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
            self._embedder = ClipEmbedder(self.model_id, device)
        return self._embedder

    def _ensure_embedder_for_model(self, model_id: str) -> ClipEmbedder:
        if model_id == self.model_id:
            return self._ensure_embedder()
        embedder = self._embedders_by_model.get(model_id)
        if embedder is not None:
            return embedder
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        embedder = ClipEmbedder(model_id, device)
        self._embedders_by_model[model_id] = embedder
        return embedder


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    term = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    arc = 2.0 * math.atan2(math.sqrt(term), math.sqrt(max(1.0 - term, 0.0)))
    return radius_km * arc


def _diversity_penalty(candidate: GeoCandidate, selected: Sequence[GeoCandidate], radius_km: float) -> float:
    if not selected:
        return 0.0
    nearest = min(
        _haversine_km(candidate.latitude, candidate.longitude, pick.latitude, pick.longitude)
        for pick in selected
    )
    scale = max(1e-3, float(radius_km))
    return math.exp(-nearest / scale)


def _select_diverse_geo_candidates(
    ranked: Sequence[GeoCandidate],
    top_k: int,
    radius_km: float,
    diversity_lambda: float,
    min_keep: int,
) -> List[GeoCandidate]:
    if not ranked:
        return []
    k = min(max(1, int(top_k)), len(ranked))
    ordered = sorted(ranked, key=lambda item: item.retrieval_score, reverse=True)
    if radius_km <= 0.0 or diversity_lambda >= 0.999:
        return list(ordered[:k])

    keep = min(k, max(0, int(min_keep)))
    selected: List[GeoCandidate] = list(ordered[:keep])
    remaining = list(ordered[keep:])
    if not selected and remaining:
        selected.append(remaining.pop(0))

    while remaining and len(selected) < k:
        best_idx = 0
        best_value = float("-inf")
        for idx, cand in enumerate(remaining):
            penalty = _diversity_penalty(cand, selected, radius_km=radius_km)
            value = diversity_lambda * cand.retrieval_score - (1.0 - diversity_lambda) * penalty
            if value > best_value:
                best_value = value
                best_idx = idx
        selected.append(remaining.pop(best_idx))

    selected.sort(key=lambda item: item.retrieval_score, reverse=True)
    return selected[:k]


def _score_to_unit_interval(score: float) -> float:
    if not math.isfinite(score):
        return 0.5
    if 0.0 <= score <= 1.0:
        return score
    if -1.0 <= score <= 1.0:
        return 0.5 * (score + 1.0)
    return 1.0 / (1.0 + math.exp(-score))


def _locality_support_likelihoods(candidates: Sequence[GeoCandidate], radius_km: float) -> List[float]:
    if not candidates:
        return []
    if len(candidates) < 3:
        return [1.0 for _ in candidates]
    radius = max(1e-3, float(radius_km))
    raw: List[float] = []
    for idx_i, cand_i in enumerate(candidates):
        support = 0.0
        for idx_j, cand_j in enumerate(candidates):
            if idx_i == idx_j:
                continue
            dist = _haversine_km(cand_i.latitude, cand_i.longitude, cand_j.latitude, cand_j.longitude)
            support += _score_to_unit_interval(cand_j.retrieval_score) * math.exp(-dist / radius)
        raw.append(max(1e-9, support))
    peak = max(raw) if raw else 0.0
    if peak <= 0.0:
        return [1.0 for _ in candidates]
    return [max(1e-3, min(1.0, val / peak)) for val in raw]


def _apply_locality_rerank(
    ranked: Sequence[GeoCandidate],
    radius_km: float,
    weight: float,
) -> List[GeoCandidate]:
    if not ranked:
        return []
    if radius_km <= 0.0 or weight <= 0.0:
        return list(ranked)

    support = _locality_support_likelihoods(ranked, radius_km=radius_km)
    alpha = max(0.0, min(1.0, float(weight) / (1.0 + float(weight))))
    rescored: List[GeoCandidate] = []
    for cand, sup in zip(ranked, support):
        base = _score_to_unit_interval(cand.retrieval_score)
        adjusted = base * ((1.0 - alpha) + alpha * sup)
        rescored.append(
            GeoCandidate(
                latitude=cand.latitude,
                longitude=cand.longitude,
                retrieval_score=max(1e-6, min(1.0, adjusted)),
                match_id=cand.match_id,
            )
        )
    rescored.sort(key=lambda item: item.retrieval_score, reverse=True)
    return rescored


def _normalize_tta_degrees(values: Optional[Sequence[float]]) -> List[float]:
    if not values:
        return [0.0]
    out: List[float] = []
    seen = set()
    for raw in values:
        if not isinstance(raw, (int, float)):
            continue
        val = float(raw)
        if not math.isfinite(val):
            continue
        norm = ((val + 180.0) % 360.0) - 180.0
        key = round(norm, 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    if not out:
        return [0.0]
    out.sort(key=lambda item: abs(item))
    return out[:12]


def _query_embeddings(embedder: ClipEmbedder, image: Image.Image, tta_degrees: Sequence[float]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for deg in tta_degrees:
        if abs(float(deg)) < 1e-9:
            variant = image
        else:
            resampling = getattr(Image, "Resampling", Image)
            variant = image.rotate(float(deg), resample=resampling.BICUBIC, expand=False)
        emb = embedder.embed(variant)
        if emb.ndim == 2:
            if emb.shape[0] <= 0:
                continue
            vec = emb[0]
        else:
            vec = emb
        vectors.append(np.asarray(vec, dtype=np.float32))

    if not vectors:
        raise ValueError("query_embedding_empty")

    mat = np.stack(vectors, axis=0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-12, None)
    return mat.astype(np.float32, copy=False)


def _aggregate_tta_scores(scores_aug: np.ndarray, mode: str = "mean") -> np.ndarray:
    if scores_aug.ndim == 1:
        return scores_aug.astype(np.float32, copy=False)
    if scores_aug.ndim != 2:
        raise ValueError("tta_scores_invalid_rank")
    if scores_aug.shape[1] <= 0:
        raise ValueError("tta_scores_empty")
    reduce_mode = str(mode).lower()
    if reduce_mode == "max":
        out = np.max(scores_aug, axis=1)
    elif reduce_mode == "rrf":
        out = _rrf_aggregate(scores_aug)
    else:
        out = np.mean(scores_aug, axis=1)
    return np.asarray(out, dtype=np.float32)


def _rrf_aggregate(scores_aug: np.ndarray, k: int = 60) -> np.ndarray:
    if scores_aug.ndim != 2:
        raise ValueError("rrf_scores_invalid_rank")
    n_items, n_aug = scores_aug.shape
    if n_items <= 0 or n_aug <= 0:
        raise ValueError("rrf_scores_empty")
    rrf = np.zeros(n_items, dtype=np.float64)
    for col in range(n_aug):
        order = np.argsort(scores_aug[:, col])[::-1]
        for rank, idx in enumerate(order, start=1):
            rrf[idx] += 1.0 / (k + rank)
    lo = float(np.min(rrf))
    hi = float(np.max(rrf))
    if hi - lo < 1e-12:
        return np.full(n_items, 0.5, dtype=np.float32)
    return ((rrf - lo) / (hi - lo)).astype(np.float32)
