"""
Embedding-based retrieval for geo candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

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

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


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


def _normalize_source_fusion_mode(mode: object) -> str:
    text = str(mode).strip().lower()
    if text not in {"weighted_score", "rrf"}:
        return "weighted_score"
    return text


def _normalize_query_expansion_top_n(value: object) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _normalize_query_expansion_beta(value: object) -> float:
    try:
        beta = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(beta):
        return 0.0
    return max(0.0, min(1.0, beta))


def _normalize_query_expansion_alpha(value: object) -> float:
    try:
        alpha = float(value)
    except Exception:
        return 0.5
    if not math.isfinite(alpha):
        return 0.5
    return max(0.0, min(1.0, alpha))


def _normalize_tta_agreement_top_n(value: object) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _normalize_tta_agreement_weight(value: object) -> float:
    try:
        weight = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(weight):
        return 0.0
    return max(0.0, min(1.0, weight))


def _normalize_local_match_top_n(value: object) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _normalize_local_match_weight(value: object) -> float:
    try:
        weight = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(weight):
        return 0.0
    return max(0.0, min(1.0, weight))


def _normalize_local_match_ratio(value: object) -> float:
    try:
        ratio = float(value)
    except Exception:
        return 0.8
    if not math.isfinite(ratio):
        return 0.8
    return max(0.5, min(0.95, ratio))


def _normalize_local_match_max_features(value: object) -> int:
    try:
        count = int(value)
    except Exception:
        return 1200
    return max(128, min(5000, count))


def _normalize_graph_rerank_top_n(value: object) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _normalize_graph_rerank_sigma_km(value: object) -> float:
    try:
        sigma = float(value)
    except Exception:
        return 3.0
    if not math.isfinite(sigma):
        return 3.0
    return max(0.1, min(50.0, sigma))


def _normalize_graph_rerank_score_alpha(value: object) -> float:
    try:
        alpha = float(value)
    except Exception:
        return 0.4
    if not math.isfinite(alpha):
        return 0.4
    return max(0.0, min(3.0, alpha))


def _normalize_graph_rerank_support_beta(value: object) -> float:
    try:
        beta = float(value)
    except Exception:
        return 1.0
    if not math.isfinite(beta):
        return 1.0
    return max(0.0, min(5.0, beta))


def _normalize_graph_rerank_center_radius_km(value: object) -> float:
    try:
        radius = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(radius):
        return 0.0
    return max(0.0, min(50.0, radius))


def _normalize_kde_refine_top_n(value: object) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _normalize_kde_refine_sigma_km(value: object) -> float:
    try:
        sigma = float(value)
    except Exception:
        return 2.0
    if not math.isfinite(sigma):
        return 2.0
    return max(0.1, min(50.0, sigma))


def _normalize_kde_refine_score_power(value: object) -> float:
    try:
        power = float(value)
    except Exception:
        return 1.0
    if not math.isfinite(power):
        return 1.0
    return max(0.0, min(5.0, power))


def _normalize_kde_refine_margin_threshold(value: object) -> float:
    try:
        margin = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(margin):
        return 0.0
    return max(0.0, min(1.0, margin))


def _normalize_kde_refine_switch_radius_km(value: object) -> float:
    try:
        radius = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(radius):
        return 0.0
    return max(0.0, min(50.0, radius))


def _normalize_kde_refine_max_iters(value: object) -> int:
    try:
        iters = int(value)
    except Exception:
        return 8
    return max(1, min(32, iters))


def _normalize_kde_refine_adaptive_mass(value: object) -> float:
    try:
        mass = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(mass):
        return 0.0
    return max(0.0, min(1.0, mass))


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


def _aggregate_query_vector(query_mat: np.ndarray) -> np.ndarray:
    if query_mat.ndim == 1:
        vec = np.asarray(query_mat, dtype=np.float32)
    elif query_mat.ndim == 2:
        if query_mat.shape[0] <= 0:
            raise ValueError("query_embedding_empty")
        vec = np.asarray(np.mean(query_mat, axis=0), dtype=np.float32)
    else:
        raise ValueError("query_embedding_invalid_rank")
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("query_embedding_zero_norm")
    return np.asarray(vec / norm, dtype=np.float32)


def _apply_query_expansion(
    *,
    index_embeddings: np.ndarray,
    query_mat: np.ndarray,
    scores_raw: np.ndarray,
    top_n: int,
    beta: float,
    alpha: float,
) -> np.ndarray:
    n_items = int(scores_raw.size)
    keep_n = min(max(0, int(top_n)), n_items)
    blend_beta = _normalize_query_expansion_beta(beta)
    blend_alpha = _normalize_query_expansion_alpha(alpha)
    if keep_n < 2 or blend_beta <= 1e-9:
        return np.asarray(scores_raw, dtype=np.float32)

    query_vec = _aggregate_query_vector(query_mat)
    if index_embeddings.ndim != 2 or query_vec.shape[0] != index_embeddings.shape[1]:
        raise ValueError("query_expansion_dim_mismatch")

    top_idx = _top_indices(scores_raw, keep_n)
    if top_idx.size < 2:
        return np.asarray(scores_raw, dtype=np.float32)

    top_scores = np.asarray(scores_raw[top_idx], dtype=np.float32)
    logits = top_scores - float(np.max(top_scores))
    weights = np.exp(logits)
    denom = float(np.sum(weights))
    if not math.isfinite(denom) or denom <= 1e-12:
        return np.asarray(scores_raw, dtype=np.float32)
    weights = np.asarray(weights / denom, dtype=np.float32)

    feedback = np.sum(index_embeddings[top_idx] * weights[:, None], axis=0)
    feedback = np.asarray(feedback, dtype=np.float32)
    feedback_norm = float(np.linalg.norm(feedback))
    if not math.isfinite(feedback_norm) or feedback_norm < 1e-12:
        return np.asarray(scores_raw, dtype=np.float32)
    feedback = np.asarray(feedback / feedback_norm, dtype=np.float32)

    fused_query = (1.0 - blend_beta) * query_vec + blend_beta * feedback
    fused_norm = float(np.linalg.norm(fused_query))
    if not math.isfinite(fused_norm) or fused_norm < 1e-12:
        return np.asarray(scores_raw, dtype=np.float32)
    fused_query = np.asarray(fused_query / fused_norm, dtype=np.float32)

    expanded_scores = np.asarray(index_embeddings @ fused_query, dtype=np.float32)
    base_scores = np.asarray(scores_raw, dtype=np.float32)
    out = blend_alpha * base_scores + (1.0 - blend_alpha) * expanded_scores
    return np.asarray(out, dtype=np.float32)


def _collect_weighted_ranked_candidates(
    *,
    loaded_indices: Sequence[LoadedRetrievalIndex],
    index_weights: Sequence[float],
    query_mats_by_model: dict[str, np.ndarray],
    global_top_k: int,
    per_index_top_k: int,
    reduce_mode: str,
    index_score_norm: str,
    query_expansion_top_n: int,
    query_expansion_beta: float,
    query_expansion_alpha: float,
    tta_agreement_top_n: int,
    tta_agreement_weight: float,
) -> Tuple[List[GeoCandidate], int]:
    return _collect_ranked_candidates(
        loaded_indices=loaded_indices,
        index_weights=index_weights,
        query_mats_by_model=query_mats_by_model,
        global_top_k=global_top_k,
        per_index_top_k=per_index_top_k,
        reduce_mode=reduce_mode,
        index_score_norm=index_score_norm,
        query_expansion_top_n=query_expansion_top_n,
        query_expansion_beta=query_expansion_beta,
        query_expansion_alpha=query_expansion_alpha,
        tta_agreement_top_n=tta_agreement_top_n,
        tta_agreement_weight=tta_agreement_weight,
        source_fusion_mode="weighted_score",
    )


def _collect_ranked_candidates(
    *,
    loaded_indices: Sequence[LoadedRetrievalIndex],
    index_weights: Sequence[float],
    query_mats_by_model: dict[str, np.ndarray],
    global_top_k: int,
    per_index_top_k: int,
    reduce_mode: str,
    index_score_norm: str,
    query_expansion_top_n: int,
    query_expansion_beta: float,
    query_expansion_alpha: float,
    tta_agreement_top_n: int,
    tta_agreement_weight: float,
    source_fusion_mode: str,
) -> Tuple[List[GeoCandidate], int]:
    merged: List[GeoCandidate] = []
    fallback_model_id = next(iter(query_mats_by_model.keys()), "")
    fusion_mode = _normalize_source_fusion_mode(source_fusion_mode)
    effective_norm = _normalize_index_score_norm(index_score_norm)
    if effective_norm == "auto":
        effective_norm = "zscore_sigmoid" if len(loaded_indices) > 1 else "none"
    # Rank-fusion mode aggregates per-source ranks and is intentionally score-scale agnostic.
    rrf_k = 60.0
    rrf_accum: dict[tuple[float, float], dict[str, object]] = {}
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
        scores_raw = _apply_query_expansion(
            index_embeddings=index.embeddings,
            query_mat=query_mat,
            scores_raw=scores_raw,
            top_n=query_expansion_top_n,
            beta=query_expansion_beta,
            alpha=query_expansion_alpha,
        )
        scores = _normalize_scores_for_source(scores_raw, mode=effective_norm)
        scores = _apply_tta_agreement_rerank(
            scores_aug=scores_aug,
            aggregated_scores=scores,
            top_n=tta_agreement_top_n,
            weight=tta_agreement_weight,
        )
        if scores.size <= 0:
            continue
        per_source_top = min(
            max(1, int(per_index_top_k) if per_index_top_k > 0 else int(global_top_k)),
            int(scores.size),
        )
        weight = float(index_weights[idx]) if idx < len(index_weights) else 1.0
        top_order = list(_top_indices(scores, per_source_top))
        for rank, item_idx in enumerate(top_order, start=1):
            lat = float(index.latitudes[item_idx])
            lon = float(index.longitudes[item_idx])
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            image_path = str(index.paths[item_idx])
            raw_score = max(-1.0, min(1.0, float(scores[item_idx])))
            match_id = _format_retrieval_match_id(loaded.source, index.ids[item_idx])
            if fusion_mode == "rrf":
                key = (round(lat, 5), round(lon, 5))
                state = rrf_accum.get(key)
                contrib = max(0.0, weight) / (rrf_k + float(rank))
                if state is None:
                    rrf_accum[key] = {
                        "latitude": lat,
                        "longitude": lon,
                        "score": float(contrib),
                        "match_id": match_id,
                        "best_raw": raw_score,
                        "image_path": image_path,
                    }
                else:
                    state["score"] = float(state["score"]) + float(contrib)
                    if raw_score > float(state.get("best_raw", -1.0)):
                        state["best_raw"] = raw_score
                        state["match_id"] = match_id
                        state["image_path"] = image_path
                continue
            weighted = max(-1.0, min(1.0, raw_score * weight))
            merged.append(
                GeoCandidate(
                    latitude=lat,
                    longitude=lon,
                    retrieval_score=weighted,
                    match_id=match_id,
                    image_path=image_path,
                )
            )

    if fusion_mode == "rrf" and rrf_accum:
        totals = np.asarray([float(item["score"]) for item in rrf_accum.values()], dtype=np.float32)
        lo = float(np.min(totals))
        hi = float(np.max(totals))
        span = hi - lo
        for item in rrf_accum.values():
            total = float(item["score"])
            if span < 1e-9:
                fused = 0.5
            else:
                fused = (total - lo) / span
            merged.append(
                GeoCandidate(
                    latitude=float(item["latitude"]),
                    longitude=float(item["longitude"]),
                    retrieval_score=max(1e-6, min(1.0, float(fused))),
                    match_id=str(item["match_id"]),
                    image_path=(str(item["image_path"]) if item.get("image_path") else None),
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
        consensus_top_n: int = 0,
        consensus_radius_km: float = 0.0,
        consensus_score_power: float = 1.0,
        query_tta_degrees: Optional[Sequence[float]] = None,
        query_tta_modes: Optional[Sequence[str]] = None,
        query_tta_scales: Optional[Sequence[float]] = None,
        query_tta_auto_modality: bool = False,
        query_tta_reduce: str = "mean",
        query_expansion_top_n: int = 0,
        query_expansion_beta: float = 0.0,
        query_expansion_alpha: float = 0.5,
        tta_agreement_top_n: int = 0,
        tta_agreement_weight: float = 0.0,
        local_match_top_n: int = 0,
        local_match_weight: float = 0.0,
        local_match_ratio: float = 0.8,
        local_match_max_features: int = 1200,
        graph_rerank_top_n: int = 0,
        graph_rerank_sigma_km: float = 3.0,
        graph_rerank_score_alpha: float = 0.4,
        graph_rerank_support_beta: float = 1.0,
        graph_rerank_center_radius_km: float = 0.0,
        kde_refine_top_n: int = 0,
        kde_refine_sigma_km: float = 2.0,
        kde_refine_score_power: float = 1.0,
        kde_refine_margin_threshold: float = 0.0,
        kde_refine_switch_radius_km: float = 0.0,
        kde_refine_max_iters: int = 8,
        kde_refine_adaptive_mass: float = 0.0,
        source_fusion_mode: str = "weighted_score",
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
        self.source_fusion_mode = _normalize_source_fusion_mode(source_fusion_mode)
        self.source_balance_beta = max(0.0, float(source_balance_beta))
        self.min_score = min_score
        self.min_keep_topk = max(0, int(min_keep_topk))
        self.diversity_radius_km = max(0.0, float(diversity_radius_km))
        self.diversity_lambda = min(1.0, max(0.0, float(diversity_lambda)))
        self.diversity_min_keep = max(0, int(diversity_min_keep))
        self.locality_radius_km = max(0.0, float(locality_radius_km))
        self.locality_weight = max(0.0, float(locality_weight))
        self.consensus_top_n = max(0, int(consensus_top_n))
        self.consensus_radius_km = max(0.0, float(consensus_radius_km))
        self.consensus_score_power = max(0.0, float(consensus_score_power))
        self.query_tta_degrees = _normalize_tta_degrees(query_tta_degrees)
        self.query_tta_modes = _normalize_tta_modes(query_tta_modes)
        self.query_tta_scales = _normalize_tta_scales(query_tta_scales)
        self.query_tta_auto_modality = bool(query_tta_auto_modality)
        reduce_mode = str(query_tta_reduce).lower()
        if reduce_mode not in {"mean", "median", "max", "rrf"}:
            reduce_mode = "mean"
        self.query_tta_reduce = reduce_mode
        self.query_expansion_top_n = _normalize_query_expansion_top_n(query_expansion_top_n)
        self.query_expansion_beta = _normalize_query_expansion_beta(query_expansion_beta)
        self.query_expansion_alpha = _normalize_query_expansion_alpha(query_expansion_alpha)
        self.tta_agreement_top_n = _normalize_tta_agreement_top_n(tta_agreement_top_n)
        self.tta_agreement_weight = _normalize_tta_agreement_weight(tta_agreement_weight)
        self.local_match_top_n = _normalize_local_match_top_n(local_match_top_n)
        self.local_match_weight = _normalize_local_match_weight(local_match_weight)
        self.local_match_ratio = _normalize_local_match_ratio(local_match_ratio)
        self.local_match_max_features = _normalize_local_match_max_features(local_match_max_features)
        self.graph_rerank_top_n = _normalize_graph_rerank_top_n(graph_rerank_top_n)
        self.graph_rerank_sigma_km = _normalize_graph_rerank_sigma_km(graph_rerank_sigma_km)
        self.graph_rerank_score_alpha = _normalize_graph_rerank_score_alpha(graph_rerank_score_alpha)
        self.graph_rerank_support_beta = _normalize_graph_rerank_support_beta(graph_rerank_support_beta)
        self.graph_rerank_center_radius_km = _normalize_graph_rerank_center_radius_km(
            graph_rerank_center_radius_km
        )
        self.kde_refine_top_n = _normalize_kde_refine_top_n(kde_refine_top_n)
        self.kde_refine_sigma_km = _normalize_kde_refine_sigma_km(kde_refine_sigma_km)
        self.kde_refine_score_power = _normalize_kde_refine_score_power(kde_refine_score_power)
        self.kde_refine_margin_threshold = _normalize_kde_refine_margin_threshold(kde_refine_margin_threshold)
        self.kde_refine_switch_radius_km = _normalize_kde_refine_switch_radius_km(
            kde_refine_switch_radius_km
        )
        self.kde_refine_max_iters = _normalize_kde_refine_max_iters(kde_refine_max_iters)
        self.kde_refine_adaptive_mass = _normalize_kde_refine_adaptive_mass(kde_refine_adaptive_mass)
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
            query_gray = None
            if self.local_match_top_n > 0 and self.local_match_weight > 0.0 and cv2 is not None:
                query_gray = np.asarray(image.convert("L"), dtype=np.uint8)
            effective_tta_modes = _select_query_tta_modes(
                image_path=image_path,
                image=image,
                base_modes=self.query_tta_modes,
                auto_modality=self.query_tta_auto_modality,
            )
            query_mats_by_model: dict[str, np.ndarray] = {}
            model_ids = list(
                dict.fromkeys((getattr(loaded, "model_id", "") or self.model_id) for loaded in loaded_indices)
            )
            for model_id in model_ids:
                if model_id == self.model_id:
                    query_mats_by_model[model_id] = _query_embeddings(
                        embedder,
                        image,
                        self.query_tta_degrees,
                        effective_tta_modes,
                        self.query_tta_scales,
                    )
                    continue
                other = self._ensure_embedder_for_model(model_id)
                query_mats_by_model[model_id] = _query_embeddings(
                    other,
                    image,
                    self.query_tta_degrees,
                    effective_tta_modes,
                    self.query_tta_scales,
                )
            ranked, top_k = _collect_ranked_candidates(
                loaded_indices=loaded_indices,
                index_weights=self.index_weights,
                query_mats_by_model=query_mats_by_model,
                global_top_k=self.top_k,
                per_index_top_k=self.per_index_top_k,
                reduce_mode=self.query_tta_reduce,
                index_score_norm=self.index_score_norm,
                query_expansion_top_n=self.query_expansion_top_n,
                query_expansion_beta=self.query_expansion_beta,
                query_expansion_alpha=self.query_expansion_alpha,
                tta_agreement_top_n=self.tta_agreement_top_n,
                tta_agreement_weight=self.tta_agreement_weight,
                source_fusion_mode=self.source_fusion_mode,
            )
            if not ranked:
                self.last_error = "index_empty"
                return []
            ranked = _apply_local_match_rerank(
                ranked,
                query_gray=query_gray,
                top_n=self.local_match_top_n,
                weight=self.local_match_weight,
                ratio_test=self.local_match_ratio,
                max_features=self.local_match_max_features,
            )
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
            ranked = _apply_graph_support_rerank(
                ranked,
                top_n=self.graph_rerank_top_n,
                sigma_km=self.graph_rerank_sigma_km,
                score_alpha=self.graph_rerank_score_alpha,
                support_beta=self.graph_rerank_support_beta,
                center_radius_km=self.graph_rerank_center_radius_km,
            )
            results = _select_diverse_geo_candidates(
                ranked,
                top_k=top_k,
                radius_km=self.diversity_radius_km,
                diversity_lambda=self.diversity_lambda,
                min_keep=self.diversity_min_keep,
            )
            results = _apply_consensus_refinement(
                results,
                top_n=self.consensus_top_n,
                radius_km=self.consensus_radius_km,
                score_power=self.consensus_score_power,
            )
            results = _apply_kde_mode_refinement(
                results,
                top_n=self.kde_refine_top_n,
                sigma_km=self.kde_refine_sigma_km,
                score_power=self.kde_refine_score_power,
                margin_threshold=self.kde_refine_margin_threshold,
                switch_radius_km=self.kde_refine_switch_radius_km,
                max_iters=self.kde_refine_max_iters,
                adaptive_mass=self.kde_refine_adaptive_mass,
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
                image_path=cand.image_path,
            )
        )
    rescored.sort(key=lambda item: item.retrieval_score, reverse=True)
    return rescored


def _apply_graph_support_rerank(
    ranked: Sequence[GeoCandidate],
    *,
    top_n: int,
    sigma_km: float,
    score_alpha: float,
    support_beta: float,
    center_radius_km: float,
) -> List[GeoCandidate]:
    if not ranked:
        return []
    ordered = sorted(ranked, key=lambda item: item.retrieval_score, reverse=True)
    n = min(max(0, int(top_n)), len(ordered))
    sigma = float(sigma_km)
    alpha = max(0.0, float(score_alpha))
    beta = max(0.0, float(support_beta))
    center_radius = max(0.0, float(center_radius_km))
    if n < 2 or sigma <= 0.0 or beta <= 0.0:
        return ordered

    subset = list(ordered[:n])
    base = [max(1e-9, _score_to_unit_interval(cand.retrieval_score)) for cand in subset]
    inv_two_sigma_sq = 0.5 / max(1e-6, sigma * sigma)

    support_vals: List[float] = []
    for idx_i, cand_i in enumerate(subset):
        support = 0.0
        for idx_j, cand_j in enumerate(subset):
            dist = _haversine_km(cand_i.latitude, cand_i.longitude, cand_j.latitude, cand_j.longitude)
            support += base[idx_j] * math.exp(-(dist * dist) * inv_two_sigma_sq)
        support_vals.append(max(1e-9, support))

    raw_vals = [(base[idx] ** alpha) * (support_vals[idx] ** beta) for idx in range(len(subset))]
    peak = max(raw_vals) if raw_vals else 0.0
    if peak <= 1e-12:
        return ordered
    scores = [max(1e-6, min(1.0, float(val / peak))) for val in raw_vals]

    updated_subset: List[GeoCandidate] = []
    for cand, score in zip(subset, scores):
        updated_subset.append(
            GeoCandidate(
                latitude=cand.latitude,
                longitude=cand.longitude,
                retrieval_score=score,
                match_id=cand.match_id,
                image_path=cand.image_path,
            )
        )

    if center_radius > 0.0 and len(updated_subset) >= 2:
        anchor_idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
        anchor = updated_subset[anchor_idx]
        members: List[int] = []
        for idx, cand in enumerate(updated_subset):
            dist = _haversine_km(anchor.latitude, anchor.longitude, cand.latitude, cand.longitude)
            if dist <= center_radius:
                members.append(idx)
        if len(members) >= 2:
            member_weights = [max(1e-9, float(scores[idx])) for idx in members]
            total = sum(member_weights)
            if total > 0.0:
                lat = sum(updated_subset[idx].latitude * w for idx, w in zip(members, member_weights)) / total
                lon = _circular_weighted_mean_longitude(
                    [updated_subset[idx].longitude for idx in members],
                    member_weights,
                )
                updated_subset[anchor_idx] = GeoCandidate(
                    latitude=float(lat),
                    longitude=float(lon),
                    retrieval_score=float(updated_subset[anchor_idx].retrieval_score),
                    match_id="retrieval:graph_support_consensus",
                    image_path=updated_subset[anchor_idx].image_path,
                )

    merged = [*updated_subset, *ordered[n:]]
    merged.sort(key=lambda item: item.retrieval_score, reverse=True)
    return merged


def _resolve_candidate_image_path(raw_path: Optional[str]) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.exists():
        return candidate
    return None


def _build_local_match_engines(max_features: int):
    if cv2 is None:
        return []
    engines = []
    if hasattr(cv2, "SIFT_create"):
        try:
            sift = cv2.SIFT_create(nfeatures=max_features)
            matcher = cv2.BFMatcher(getattr(cv2, "NORM_L2", 4), crossCheck=False)
            engines.append(("sift", sift, matcher, 6))
        except Exception:
            pass
    try:
        orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=15,
        )
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        engines.append(("orb", orb, matcher, 6))
    except Exception:
        pass
    return engines


def _score_local_feature_match(
    query_kp,
    query_desc: np.ndarray,
    cand_path: Path,
    *,
    orb,
    matcher,
    ratio_test: float,
) -> Optional[float]:
    if cv2 is None:
        return None
    cand_img = cv2.imread(str(cand_path), cv2.IMREAD_GRAYSCALE)
    if cand_img is None:
        return None
    cand_kp, cand_desc = orb.detectAndCompute(cand_img, None)
    if cand_desc is None or len(cand_kp) < 4:
        return 0.0
    try:
        pairs = matcher.knnMatch(query_desc, cand_desc, k=2)
    except Exception:
        return None
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair[0], pair[1]
        if first.distance < ratio_test * second.distance:
            good.append(first)
    if not good:
        return 0.0

    inliers = len(good)
    if len(good) >= 4:
        src_pts = np.float32([query_kp[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([cand_kp[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None and mask.size > 0:
            inliers = int(np.sum(mask))

    denom_inliers = max(24.0, min(float(len(query_kp)), float(len(cand_kp))))
    inlier_score = min(1.0, float(inliers) / denom_inliers)
    denom_matches = max(40.0, float(min(query_desc.shape[0], cand_desc.shape[0])))
    match_score = min(1.0, float(len(good)) / denom_matches)
    return max(0.0, min(1.0, (0.6 * inlier_score) + (0.4 * match_score)))


def _apply_local_match_rerank(
    ranked: Sequence[GeoCandidate],
    *,
    query_gray: Optional[np.ndarray],
    top_n: int,
    weight: float,
    ratio_test: float,
    max_features: int,
) -> List[GeoCandidate]:
    if not ranked:
        return []
    if cv2 is None:
        return list(ranked)
    k = min(max(0, int(top_n)), len(ranked))
    blend = _normalize_local_match_weight(weight)
    if k <= 0 or blend <= 1e-9:
        return list(ranked)
    if query_gray is None or query_gray.ndim != 2:
        return list(ranked)

    engines = _build_local_match_engines(max_features=max_features)
    if not engines:
        return list(ranked)

    query_features = []
    for name, detector, matcher, min_kp in engines:
        try:
            query_kp, query_desc = detector.detectAndCompute(query_gray, None)
        except Exception:
            continue
        if query_desc is None or len(query_kp) < max(4, int(min_kp)):
            continue
        query_features.append((name, query_kp, query_desc, detector, matcher))
    if not query_features:
        return list(ranked)

    ordered = sorted(ranked, key=lambda item: item.retrieval_score, reverse=True)
    local_scores: List[Optional[float]] = []
    for idx, cand in enumerate(ordered[:k]):
        cand_path = _resolve_candidate_image_path(cand.image_path)
        if cand_path is None:
            local_scores.append(None)
            continue
        best_local = None
        for name, query_kp, query_desc, detector, matcher in query_features:
            effective_ratio = float(ratio_test)
            if isinstance(query_desc, np.ndarray) and np.issubdtype(query_desc.dtype, np.floating):
                effective_ratio = min(float(ratio_test), 0.78)
            local_score = _score_local_feature_match(
                query_kp=query_kp,
                query_desc=query_desc,
                cand_path=cand_path,
                orb=detector,
                matcher=matcher,
                ratio_test=effective_ratio,
            )
            if local_score is None:
                continue
            value = max(0.0, min(1.0, float(local_score)))
            if best_local is None or value > best_local:
                best_local = value
        local_scores.append(best_local)

    valid_locals = [val for val in local_scores if val is not None]
    if not valid_locals:
        return ordered

    # Evidence gate: only trust local reranking when at least one candidate
    # has meaningful geometric agreement with the query.
    if max(valid_locals) < 0.18:
        return ordered

    updated: List[GeoCandidate] = []
    changed = False
    for idx, cand in enumerate(ordered):
        if idx >= k:
            updated.append(cand)
            continue
        local_score = local_scores[idx] if idx < len(local_scores) else None
        if local_score is None:
            updated.append(cand)
            continue
        base = _score_to_unit_interval(cand.retrieval_score)
        local_strength = max(0.0, min(1.0, (float(local_score) - 0.10) / 0.90))
        effective_blend = blend * local_strength
        fused = ((1.0 - effective_blend) * base) + (effective_blend * float(local_score))
        updated.append(
            GeoCandidate(
                latitude=cand.latitude,
                longitude=cand.longitude,
                retrieval_score=max(1e-6, min(1.0, float(fused))),
                match_id=cand.match_id,
                image_path=cand.image_path,
            )
        )
        changed = True
    if not changed:
        return ordered
    updated.sort(key=lambda item: item.retrieval_score, reverse=True)
    return updated


def _normalize_longitude(lon: float) -> float:
    wrapped = ((float(lon) + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and lon > 0.0:
        return 180.0
    return wrapped


def _circular_weighted_mean_longitude(values: Sequence[float], weights: Sequence[float]) -> float:
    sin_sum = 0.0
    cos_sum = 0.0
    for lon, weight in zip(values, weights):
        radians = math.radians(float(lon))
        w = float(weight)
        sin_sum += w * math.sin(radians)
        cos_sum += w * math.cos(radians)
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        total = sum(float(w) for w in weights)
        if total <= 0.0:
            return _normalize_longitude(float(values[0]) if values else 0.0)
        return _normalize_longitude(
            sum(float(v) * float(w) for v, w in zip(values, weights)) / total
        )
    return _normalize_longitude(math.degrees(math.atan2(sin_sum, cos_sum)))


def _latlon_to_local_xy_km(
    lat: float,
    lon: float,
    *,
    ref_lat: float,
    ref_lon: float,
) -> tuple[float, float]:
    lat_scale = 110.574
    cos_lat = max(1e-6, math.cos(math.radians(ref_lat)))
    lon_scale = 111.320 * cos_lat
    delta_lon = _normalize_longitude(float(lon) - float(ref_lon))
    x_km = delta_lon * lon_scale
    y_km = (float(lat) - float(ref_lat)) * lat_scale
    return x_km, y_km


def _local_xy_km_to_latlon(
    x_km: float,
    y_km: float,
    *,
    ref_lat: float,
    ref_lon: float,
) -> tuple[float, float]:
    lat_scale = 110.574
    cos_lat = max(1e-6, math.cos(math.radians(ref_lat)))
    lon_scale = 111.320 * cos_lat
    lat = float(ref_lat) + (float(y_km) / lat_scale)
    lon = _normalize_longitude(float(ref_lon) + (float(x_km) / lon_scale))
    return lat, lon


def _weighted_geo_median_latlon(
    lats: Sequence[float],
    lons: Sequence[float],
    weights: Sequence[float],
    *,
    max_iters: int = 24,
    tol_km: float = 1e-3,
) -> tuple[float, float]:
    if not lats or not lons or not weights:
        return 0.0, 0.0
    n = min(len(lats), len(lons), len(weights))
    if n == 1:
        return float(lats[0]), _normalize_longitude(float(lons[0]))

    safe_weights = [max(1e-9, float(weights[idx])) for idx in range(n)]
    total_w = sum(safe_weights)
    if total_w <= 0.0:
        return float(lats[0]), _normalize_longitude(float(lons[0]))

    ref_lat = sum(float(lats[idx]) * safe_weights[idx] for idx in range(n)) / total_w
    ref_lon = _circular_weighted_mean_longitude([float(lons[idx]) for idx in range(n)], safe_weights)
    points_xy = [
        _latlon_to_local_xy_km(float(lats[idx]), float(lons[idx]), ref_lat=ref_lat, ref_lon=ref_lon)
        for idx in range(n)
    ]

    x = sum(px * w for (px, _), w in zip(points_xy, safe_weights)) / total_w
    y = sum(py * w for (_, py), w in zip(points_xy, safe_weights)) / total_w

    for _ in range(max(1, int(max_iters))):
        num_x = 0.0
        num_y = 0.0
        denom = 0.0
        snapped = None
        for (px, py), w in zip(points_xy, safe_weights):
            dist = math.hypot(x - px, y - py)
            if dist < 1e-9:
                snapped = (px, py)
                break
            factor = w / dist
            num_x += factor * px
            num_y += factor * py
            denom += factor

        if snapped is not None:
            x_new, y_new = snapped
        elif denom <= 0.0:
            break
        else:
            x_new = num_x / denom
            y_new = num_y / denom

        step = math.hypot(x_new - x, y_new - y)
        x, y = x_new, y_new
        if step <= max(1e-6, float(tol_km)):
            break

    return _local_xy_km_to_latlon(x, y, ref_lat=ref_lat, ref_lon=ref_lon)


def _apply_consensus_refinement(
    ranked: Sequence[GeoCandidate],
    *,
    top_n: int,
    radius_km: float,
    score_power: float,
) -> List[GeoCandidate]:
    if not ranked:
        return []
    ordered = sorted(ranked, key=lambda item: item.retrieval_score, reverse=True)
    if len(ordered) < 2:
        return ordered
    n = min(len(ordered), max(0, int(top_n)))
    radius = float(radius_km)
    if n < 2 or radius <= 0.0:
        return ordered

    subset = list(ordered[:n])
    base_weights = [max(1e-9, _score_to_unit_interval(item.retrieval_score)) for item in subset]
    sigma = max(0.5, radius)
    anchor_idx = 0
    best_support = float("-inf")
    for idx_i, cand_i in enumerate(subset):
        support = 0.0
        for cand_j, weight_j in zip(subset, base_weights):
            dist = _haversine_km(cand_i.latitude, cand_i.longitude, cand_j.latitude, cand_j.longitude)
            support += weight_j * math.exp(-0.5 * (dist / sigma) ** 2)
        if support > best_support:
            best_support = support
            anchor_idx = idx_i

    anchor = subset[anchor_idx]
    cluster: List[GeoCandidate] = []
    for cand in subset:
        dist = _haversine_km(anchor.latitude, anchor.longitude, cand.latitude, cand.longitude)
        if dist <= radius:
            cluster.append(cand)
    if len(cluster) < 2:
        return ordered

    power = max(0.0, float(score_power))
    cluster_weights = [max(1e-9, _score_to_unit_interval(cand.retrieval_score) ** power) for cand in cluster]
    total_weight = sum(cluster_weights)
    if total_weight <= 0.0:
        return ordered

    mean_lat = sum(cand.latitude * weight for cand, weight in zip(cluster, cluster_weights)) / total_weight
    mean_lon = _circular_weighted_mean_longitude([cand.longitude for cand in cluster], cluster_weights)
    median_lat, median_lon = _weighted_geo_median_latlon(
        [cand.latitude for cand in cluster],
        [cand.longitude for cand in cluster],
        cluster_weights,
    )

    sigma = max(0.5, radius * 0.5)

    def _cluster_support(lat: float, lon: float) -> float:
        support = 0.0
        for cand, weight in zip(cluster, cluster_weights):
            dist = _haversine_km(lat, lon, cand.latitude, cand.longitude)
            support += weight * math.exp(-0.5 * (dist / sigma) ** 2)
        return support

    mean_support = _cluster_support(float(mean_lat), float(mean_lon))
    median_support = _cluster_support(float(median_lat), float(median_lon))
    center_gap_km = _haversine_km(float(mean_lat), float(mean_lon), float(median_lat), float(median_lon))
    use_median = (
        median_support > (mean_support * 1.05)
        and center_gap_km >= max(0.15, radius * 0.05)
    )
    if use_median:
        center_lat, center_lon = float(median_lat), float(median_lon)
    else:
        center_lat, center_lon = float(mean_lat), float(mean_lon)
    refined = GeoCandidate(
        latitude=float(center_lat),
        longitude=float(center_lon),
        retrieval_score=float(ordered[0].retrieval_score),
        match_id="retrieval:consensus",
        image_path=ordered[0].image_path,
    )
    if (
        abs(refined.latitude - ordered[0].latitude) < 1e-9
        and abs(refined.longitude - ordered[0].longitude) < 1e-9
    ):
        return ordered
    return [refined, *ordered[1:]]


def _apply_kde_mode_refinement(
    ranked: Sequence[GeoCandidate],
    *,
    top_n: int,
    sigma_km: float,
    score_power: float,
    margin_threshold: float,
    switch_radius_km: float,
    max_iters: int,
    adaptive_mass: float = 0.0,
) -> List[GeoCandidate]:
    if not ranked:
        return []
    ordered = sorted(ranked, key=lambda item: item.retrieval_score, reverse=True)
    if len(ordered) < 2:
        return ordered

    n = min(len(ordered), max(0, int(top_n)))
    sigma = float(sigma_km)
    power = max(0.0, float(score_power))
    if n < 2 or sigma <= 0.0:
        return ordered

    subset_full = list(ordered[:n])
    weights_full = [max(1e-9, _score_to_unit_interval(item.retrieval_score) ** power) for item in subset_full]
    mass = _normalize_kde_refine_adaptive_mass(adaptive_mass)
    if mass > 0.0 and len(subset_full) >= 3:
        target = mass * sum(weights_full)
        running = 0.0
        keep = 0
        for weight in weights_full:
            running += float(weight)
            keep += 1
            if keep >= 2 and running >= target:
                break
        keep = max(2, min(len(subset_full), keep))
        subset = subset_full[:keep]
        weights = weights_full[:keep]
    else:
        subset = subset_full
        weights = weights_full
    total_w = sum(weights)
    if total_w <= 1e-12:
        return ordered

    ref_lat = sum(item.latitude * w for item, w in zip(subset, weights)) / total_w
    ref_lon = _circular_weighted_mean_longitude([item.longitude for item in subset], weights)
    points_xy = [
        _latlon_to_local_xy_km(item.latitude, item.longitude, ref_lat=ref_lat, ref_lon=ref_lon)
        for item in subset
    ]

    inv_two_sigma_sq = 0.5 / max(1e-6, sigma * sigma)
    seed_count = max(1, min(8, len(points_xy)))
    iters = max(1, int(max_iters))

    best_x = points_xy[0][0]
    best_y = points_xy[0][1]
    best_density = float("-inf")

    for seed_x, seed_y in points_xy[:seed_count]:
        x = float(seed_x)
        y = float(seed_y)
        for _ in range(iters):
            num_x = 0.0
            num_y = 0.0
            denom = 0.0
            for (px, py), w in zip(points_xy, weights):
                dist_sq = (x - px) ** 2 + (y - py) ** 2
                factor = w * math.exp(-dist_sq * inv_two_sigma_sq)
                num_x += factor * px
                num_y += factor * py
                denom += factor
            if denom <= 1e-12:
                break
            next_x = num_x / denom
            next_y = num_y / denom
            step = math.hypot(next_x - x, next_y - y)
            x, y = next_x, next_y
            if step <= 1e-3:
                break

        density = 0.0
        for (px, py), w in zip(points_xy, weights):
            dist_sq = (x - px) ** 2 + (y - py) ** 2
            density += w * math.exp(-dist_sq * inv_two_sigma_sq)
        if density > best_density:
            best_density = density
            best_x = x
            best_y = y

    mode_lat, mode_lon = _local_xy_km_to_latlon(best_x, best_y, ref_lat=ref_lat, ref_lon=ref_lon)
    top1 = ordered[0]

    guard_radius = max(0.0, float(switch_radius_km))
    if guard_radius > 0.0 and len(ordered) >= 2:
        top2 = ordered[1]
        score_gap = float(top1.retrieval_score) - float(top2.retrieval_score)
        mode_shift_km = _haversine_km(mode_lat, mode_lon, top1.latitude, top1.longitude)
        if score_gap >= max(0.0, float(margin_threshold)) and mode_shift_km >= guard_radius:
            return ordered

    if (
        abs(mode_lat - top1.latitude) < 1e-9
        and abs(mode_lon - top1.longitude) < 1e-9
    ):
        return ordered

    refined = GeoCandidate(
        latitude=float(mode_lat),
        longitude=float(mode_lon),
        retrieval_score=float(top1.retrieval_score),
        match_id="retrieval:kde_mode",
        image_path=top1.image_path,
    )
    return [refined, *ordered[1:]]


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


def _normalize_tta_modes(values: Optional[Sequence[str]]) -> List[str]:
    allowed = {"rgb", "gray", "equalize", "edge"}
    if not values:
        return ["rgb"]
    out: List[str] = []
    seen = set()
    for raw in values:
        mode = str(raw).strip().lower() if raw is not None else ""
        if mode not in allowed or mode in seen:
            continue
        seen.add(mode)
        out.append(mode)
    return out or ["rgb"]


def _normalize_tta_scales(values: Optional[Sequence[float]]) -> List[float]:
    if not values:
        return [1.0]
    out: List[float] = []
    seen = set()
    for raw in values:
        if not isinstance(raw, (int, float)):
            continue
        val = float(raw)
        if not math.isfinite(val):
            continue
        val = min(1.0, max(0.45, val))
        key = round(val, 3)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(key))
    if not out:
        return [1.0]
    if 1.0 not in seen:
        out.insert(0, 1.0)
    out = sorted(out, reverse=True)
    return out[:6]


def _apply_tta_mode(image: Image.Image, mode: str) -> Image.Image:
    if mode == "gray":
        return ImageOps.grayscale(image).convert("RGB")
    if mode == "equalize":
        gray = ImageOps.grayscale(image)
        return ImageOps.equalize(gray).convert("RGB")
    if mode == "edge":
        gray = ImageOps.grayscale(image)
        return gray.filter(ImageFilter.FIND_EDGES).convert("RGB")
    return image


def _iter_tta_scaled_views(image: Image.Image, scales: Sequence[float]):
    width, height = image.size
    yielded = set()
    for scale in scales:
        ratio = min(1.0, max(0.45, float(scale)))
        if ratio >= 0.999:
            if "full" not in yielded:
                yielded.add("full")
                yield image
            continue

        crop_w = max(8, int(round(width * ratio)))
        crop_h = max(8, int(round(height * ratio)))
        crop_w = min(width, crop_w)
        crop_h = min(height, crop_h)
        if crop_w >= width and crop_h >= height:
            if "full" not in yielded:
                yielded.add("full")
                yield image
            continue

        anchors = [
            ((width - crop_w) // 2, (height - crop_h) // 2),
            (0, 0),
            (max(0, width - crop_w), 0),
            (0, max(0, height - crop_h)),
            (max(0, width - crop_w), max(0, height - crop_h)),
        ]
        for left, top in anchors:
            box = (int(left), int(top), int(left + crop_w), int(top + crop_h))
            if box in yielded:
                continue
            yielded.add(box)
            yield image.crop(box)

    if not yielded:
        yield image


def _detect_query_modality(image_path: str, image: Image.Image) -> str:
    name = Path(str(image_path)).name.upper()
    if name.startswith("PAN_"):
        return "pan"
    if name.startswith("RGB_") or name.startswith("RGB-PANSHARPEN_"):
        return "rgb"
    if name.startswith("MUL_") or name.startswith("MUL-PANSHARPEN_"):
        return "mul"

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim >= 3 and arr.shape[2] >= 3:
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]
        chroma = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / (3.0 * 255.0))
        if chroma < 0.015:
            return "pan"
        if chroma > 0.06:
            return "rgb"
        return "mul"
    return "unknown"


def _select_query_tta_modes(
    *,
    image_path: str,
    image: Image.Image,
    base_modes: Sequence[str],
    auto_modality: bool,
) -> List[str]:
    modes = _normalize_tta_modes(base_modes)
    if not auto_modality:
        return modes
    mode_set = set(modes)
    modality = _detect_query_modality(image_path, image)
    if modality == "pan":
        preferred = ["gray", "edge"]
    elif modality == "rgb":
        preferred = ["rgb"]
    elif modality == "mul":
        preferred = ["rgb", "gray"]
    else:
        preferred = ["rgb", "gray"]
    chosen = [mode for mode in preferred if mode in mode_set]
    return chosen or modes


def _query_embeddings(
    embedder: ClipEmbedder,
    image: Image.Image,
    tta_degrees: Sequence[float],
    tta_modes: Sequence[str],
    tta_scales: Sequence[float],
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for deg in tta_degrees:
        if abs(float(deg)) < 1e-9:
            variant = image
        else:
            resampling = getattr(Image, "Resampling", Image)
            variant = image.rotate(float(deg), resample=resampling.BICUBIC, expand=False)
        for mode in tta_modes:
            base_view = _apply_tta_mode(variant, mode)
            for view in _iter_tta_scaled_views(base_view, tta_scales):
                emb = embedder.embed(view)
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
    elif reduce_mode == "median":
        out = np.median(scores_aug, axis=1)
    elif reduce_mode == "rrf":
        out = _rrf_aggregate(scores_aug)
    else:
        out = np.mean(scores_aug, axis=1)
    return np.asarray(out, dtype=np.float32)


def _apply_tta_agreement_rerank(
    *,
    scores_aug: np.ndarray,
    aggregated_scores: np.ndarray,
    top_n: int,
    weight: float,
) -> np.ndarray:
    if scores_aug.ndim != 2:
        return np.asarray(aggregated_scores, dtype=np.float32)
    n_items, n_aug = scores_aug.shape
    if n_items <= 1 or n_aug <= 1:
        return np.asarray(aggregated_scores, dtype=np.float32)
    keep_n = min(max(0, int(top_n)), n_items)
    blend = _normalize_tta_agreement_weight(weight)
    if keep_n < 2 or blend <= 1e-9:
        return np.asarray(aggregated_scores, dtype=np.float32)

    support = np.zeros(n_items, dtype=np.float64)
    rank_norm = float(sum(1.0 / float(rank + 1) for rank in range(keep_n)))
    if rank_norm <= 1e-12:
        return np.asarray(aggregated_scores, dtype=np.float32)

    for col in range(n_aug):
        order = _top_indices(np.asarray(scores_aug[:, col], dtype=np.float32), keep_n)
        for rank, item_idx in enumerate(order):
            support[int(item_idx)] += 1.0 / float(rank + 1)

    support /= float(n_aug) * rank_norm
    support = np.clip(support, 0.0, 1.0)

    base = np.asarray(aggregated_scores, dtype=np.float32)
    lo = float(np.min(base))
    hi = float(np.max(base))
    span = hi - lo
    if span <= 1e-12:
        return base
    base_unit = (base - lo) / span
    merged_unit = ((1.0 - blend) * base_unit) + (blend * np.asarray(support, dtype=np.float32))
    out = lo + (np.clip(merged_unit, 0.0, 1.0) * span)
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
