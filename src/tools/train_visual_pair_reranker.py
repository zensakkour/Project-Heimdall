"""Train a visual query-candidate reranker for geo retrieval shortlists."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from src.core.geo.retrieval_provider import ClipEmbedder
from src.core.logic.config import load_config
from src.core.logic.types import GeoCandidate
from src.tools.run_geo_eval import build_retrieval_provider, haversine_km, load_metadata_records, resolve_image_path
from src.tools.train_crossview_projection import _normalize_device

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


@dataclass(frozen=True)
class CandidateGroup:
    features: np.ndarray
    distances_km: np.ndarray
    retrieval_scores: np.ndarray


class VisualPairReranker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _resolve_candidate_image(path_text: str) -> Optional[Path]:
    raw = str(path_text or "").strip()
    if not raw:
        return None
    path = Path(raw)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                path,
                Path("data/spacenet_paris") / raw,
                Path("data/paris_realistic_v1_combined") / raw,
                Path("data/paris_realistic_v1_combined/aerial/images") / path.name,
                Path("data/spacenet_paris/chips") / path.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _source_feature(cand: GeoCandidate) -> float:
    text = f"{cand.match_id or ''} {cand.image_path or ''}".lower()
    if "aerial_clip_index" in text or "paris_realistic_v1_combined" in text:
        return 1.0
    return 0.0


def _candidate_features(
    *,
    query_vec: np.ndarray,
    candidate_vec: np.ndarray,
    candidate: GeoCandidate,
    rank: int,
    group_size: int,
) -> np.ndarray:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    c = np.asarray(candidate_vec, dtype=np.float32).reshape(-1)
    product = q * c
    abs_diff = np.abs(q - c)
    raw_score = float(candidate.retrieval_score)
    if not math.isfinite(raw_score):
        raw_score = 0.0
    score_unit = raw_score if 0.0 <= raw_score <= 1.0 else 1.0 / (1.0 + math.exp(-raw_score))
    rank_unit = 1.0 - (float(rank - 1) / max(1.0, float(group_size - 1)))
    extras = np.asarray([score_unit, rank_unit, _source_feature(candidate)], dtype=np.float32)
    return np.concatenate([product, abs_diff, extras]).astype(np.float32)


def _embed_paths(
    *,
    embedder: ClipEmbedder,
    paths: Sequence[Path],
    batch_size: int,
) -> dict[str, np.ndarray]:
    unique: list[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)

    out: dict[str, np.ndarray] = {}
    for start in range(0, len(unique), max(1, int(batch_size))):
        batch_paths = unique[start : start + max(1, int(batch_size))]
        images = []
        used = []
        for path in batch_paths:
            try:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))
                    used.append(path)
            except Exception:
                continue
        if not images:
            continue
        vectors = embedder.embed_many(images)
        for path, vec in zip(used, vectors):
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            out[str(path)] = arr
        print(f"Embedded {min(start + len(batch_paths), len(unique))}/{len(unique)} images...")
    return out


def _collect_candidate_groups(
    *,
    cfg_path: Path,
    images_dir: Path,
    metadata_path: Path,
    query_metadata_path: Optional[Path],
    limit: int,
    seed: int,
    model_id: str,
    batch_size: int,
    device: str,
) -> tuple[list[CandidateGroup], dict]:
    cfg = load_config(str(cfg_path))
    provider_cfg = replace(
        cfg,
        geolocator=replace(
            cfg.geolocator,
            retrieval_top_k=max(25, cfg.geolocator.retrieval_top_k),
            retrieval_min_keep_topk=max(10, cfg.geolocator.retrieval_min_keep_topk),
        ),
    )
    provider = build_retrieval_provider(provider_cfg)
    if provider is None:
        raise ValueError("config_has_no_retrieval_provider")

    records = load_metadata_records(metadata_path, query_metadata_path=query_metadata_path, images_dir=images_dir)
    random.Random(seed).shuffle(records)
    if limit > 0:
        records = records[:limit]

    raw_groups = []
    query_paths: list[Path] = []
    candidate_paths: list[Path] = []
    missing_files = 0
    null_candidates = 0
    skipped_no_candidate_image = 0
    oracle_distances = []

    for item in records:
        rel_path = str(item.get("path") or "")
        query_path = Path(rel_path) if Path(rel_path).is_absolute() else resolve_image_path(images_dir, rel_path)
        if not query_path.exists():
            missing_files += 1
            continue
        candidates = provider.candidates(str(query_path))
        if not candidates:
            null_candidates += 1
            continue
        gt_lat = float(item.get("latitude", item.get("lat")))
        gt_lon = float(item.get("longitude", item.get("lon", item.get("lng"))))
        candidate_items = []
        for rank, cand in enumerate(candidates, start=1):
            candidate_path = _resolve_candidate_image(str(cand.image_path or ""))
            if candidate_path is None:
                skipped_no_candidate_image += 1
                continue
            distance_km = haversine_km(gt_lat, gt_lon, cand.latitude, cand.longitude)
            candidate_items.append((rank, cand, candidate_path, distance_km))
            candidate_paths.append(candidate_path)
        if not candidate_items:
            null_candidates += 1
            continue
        distances = [distance_km for _, _, _, distance_km in candidate_items]
        oracle_distances.append(min(distances))
        raw_groups.append((query_path, candidate_items, np.asarray(distances, dtype=np.float32)))
        query_paths.append(query_path)

    embedder = ClipEmbedder(model_id=model_id, device=_normalize_device(device), projection_path=None)
    query_embeddings = _embed_paths(embedder=embedder, paths=query_paths, batch_size=batch_size)
    candidate_embeddings = _embed_paths(embedder=embedder, paths=candidate_paths, batch_size=batch_size)

    groups: list[CandidateGroup] = []
    dropped_embedding = 0
    for query_path, candidate_items, distances in raw_groups:
        q_vec = query_embeddings.get(str(query_path))
        if q_vec is None:
            dropped_embedding += 1
            continue
        rows = []
        kept_distances = []
        kept_scores = []
        group_size = len(candidate_items)
        for item_idx, (rank, cand, candidate_path, _distance_km) in enumerate(candidate_items):
            c_vec = candidate_embeddings.get(str(candidate_path))
            if c_vec is None:
                continue
            rows.append(
                _candidate_features(
                    query_vec=q_vec,
                    candidate_vec=c_vec,
                    candidate=cand,
                    rank=rank,
                    group_size=group_size,
                )
            )
            kept_distances.append(float(distances[item_idx]))
            kept_scores.append(float(cand.retrieval_score))
        if len(rows) < 2:
            dropped_embedding += 1
            continue
        groups.append(
            CandidateGroup(
                features=np.asarray(rows, dtype=np.float32),
                distances_km=np.asarray(kept_distances, dtype=np.float32),
                retrieval_scores=np.asarray(kept_scores, dtype=np.float32),
            )
        )

    stats = {
        "metadata": str(metadata_path),
        "images_dir": str(images_dir),
        "records_seen": len(records),
        "missing_files": missing_files,
        "null_candidates": null_candidates,
        "candidate_groups": len(groups),
        "candidate_rows": int(sum(group.features.shape[0] for group in groups)),
        "skipped_no_candidate_image": skipped_no_candidate_image,
        "dropped_embedding_groups": dropped_embedding,
        "oracle_mean_km": float(sum(oracle_distances) / len(oracle_distances)) if oracle_distances else None,
    }
    return groups, stats


def _softmax(values: torch.Tensor) -> torch.Tensor:
    return torch.softmax(values, dim=0)


def _target_distribution(distances: np.ndarray, sigma_km: float) -> np.ndarray:
    sigma = max(0.1, float(sigma_km))
    logits = -0.5 * (np.asarray(distances, dtype=np.float32) / sigma) ** 2
    logits = logits - float(np.max(logits))
    exp_values = np.exp(np.clip(logits, -60.0, 60.0))
    total = float(np.sum(exp_values))
    if total <= 0.0:
        return np.full_like(exp_values, 1.0 / max(1, exp_values.size), dtype=np.float32)
    return np.asarray(exp_values / total, dtype=np.float32)


def _standardize(groups: Sequence[CandidateGroup]) -> tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate([group.features for group in groups], axis=0)
    means = flat.mean(axis=0).astype(np.float32)
    scales = flat.std(axis=0).astype(np.float32)
    scales = np.where(scales < 1e-6, 1.0, scales).astype(np.float32)
    return means, scales


def _train_model(
    *,
    groups: Sequence[CandidateGroup],
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    target_sigma_km: float,
    seed: int,
) -> tuple[VisualPairReranker, np.ndarray, np.ndarray, dict]:
    if torch is None:
        raise RuntimeError("torch_not_available")
    if not groups:
        raise ValueError("no_training_groups")
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    means, scales = _standardize(groups)
    input_dim = int(groups[0].features.shape[1])
    model = VisualPairReranker(input_dim=input_dim, hidden_dim=max(0, int(hidden_dim)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=max(1e-6, float(learning_rate)), weight_decay=max(0.0, float(weight_decay)))
    order = list(range(len(groups)))
    history = []
    start = time.perf_counter()
    for epoch in range(1, max(1, int(epochs)) + 1):
        random.shuffle(order)
        losses = []
        for idx in order:
            group = groups[idx]
            x = torch.from_numpy((group.features - means) / scales).float()
            target = torch.from_numpy(_target_distribution(group.distances_km, target_sigma_km)).float()
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=0)
            loss = -(target * log_probs).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == int(epochs) or epoch % max(1, int(epochs) // 5) == 0:
            history.append({"epoch": epoch, "loss_mean": float(sum(losses) / len(losses))})
    report = {
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "target_sigma_km": float(target_sigma_km),
        "elapsed_sec": float(time.perf_counter() - start),
        "history": history,
    }
    return model, means, scales, report


def _retrieval_logit(score: float, temperature: float) -> float:
    value = float(score)
    if not math.isfinite(value):
        value = 0.5
    if not 0.0 <= value <= 1.0:
        value = 1.0 / (1.0 + math.exp(-value))
    value = max(1e-6, min(1.0 - 1e-6, value))
    return math.log(value / (1.0 - value)) / max(1e-3, float(temperature))


def _metrics(values: Sequence[float]) -> dict:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"mean_km": None, "median_km": None, "p90_km": None, "within_1km_pct": 0.0, "within_2km_pct": 0.0, "within_5km_pct": 0.0, "within_10km_pct": 0.0}
    n = len(vals)
    return {
        "mean_km": float(sum(vals) / n),
        "median_km": float(vals[n // 2]),
        "p90_km": float(vals[max(0, min(n - 1, int(round(0.9 * (n - 1)))))]),
        "within_1km_pct": 100.0 * sum(1 for v in vals if v <= 1.0) / n,
        "within_2km_pct": 100.0 * sum(1 for v in vals if v <= 2.0) / n,
        "within_5km_pct": 100.0 * sum(1 for v in vals if v <= 5.0) / n,
        "within_10km_pct": 100.0 * sum(1 for v in vals if v <= 10.0) / n,
    }


def _evaluate(
    *,
    groups: Sequence[CandidateGroup],
    model: VisualPairReranker,
    means: np.ndarray,
    scales: np.ndarray,
    weights: Sequence[float],
    retrieval_temperature: float,
) -> dict:
    model.eval()
    base_distances = []
    oracle_distances = []
    model_scores_by_group = []
    with torch.no_grad():
        for group in groups:
            x = torch.from_numpy((group.features - means) / scales).float()
            scores = model(x).cpu().numpy().astype(np.float32)
            model_scores_by_group.append(scores)
            base_distances.append(float(group.distances_km[0]))
            oracle_distances.append(float(np.min(group.distances_km)))

    variants = {}
    for weight in weights:
        selected = []
        for group, model_scores in zip(groups, model_scores_by_group):
            fused = []
            for score, retrieval_score in zip(model_scores, group.retrieval_scores):
                fused.append(float(weight) * float(score) + _retrieval_logit(float(retrieval_score), retrieval_temperature))
            selected.append(float(group.distances_km[int(np.argmax(np.asarray(fused, dtype=np.float32)))]))
        variants[str(float(weight))] = _metrics(selected)
    best_key = min(variants, key=lambda key: (variants[key]["mean_km"] or 1e9, -(variants[key]["within_5km_pct"] or 0.0)))
    return {
        "evaluated": len(groups),
        "base": _metrics(base_distances),
        "oracle": _metrics(oracle_distances),
        "reranked_by_weight": variants,
        "best_weight": float(best_key),
        "best": variants[best_key],
    }


def _save_model(path: Path, model: VisualPairReranker, means: np.ndarray, scales: np.ndarray, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch is None:
        raise RuntimeError("torch_not_available")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "means": means,
            "scales": scales,
            "report": report,
        },
        path,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a visual CLIP pair reranker for geo candidates.")
    parser.add_argument("--config", default="src/config/paris.json")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--query-metadata", default="")
    parser.add_argument("--eval-metadata", default="")
    parser.add_argument("--eval-query-metadata", default="")
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--eval-limit", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-sigma-km", type=float, default=1.5)
    parser.add_argument("--fusion-weights", default="0,0.25,0.5,1,1.5,2,3")
    parser.add_argument("--retrieval-temperature", type=float, default=0.22)
    parser.add_argument("--output", default="runs/visual_pair_reranker.pt")
    parser.add_argument("--report-output", default="")
    args = parser.parse_args(argv)

    query_metadata = Path(args.query_metadata) if str(args.query_metadata).strip() else None
    eval_metadata = Path(args.eval_metadata) if str(args.eval_metadata).strip() else Path(args.metadata)
    eval_query_metadata = Path(args.eval_query_metadata) if str(args.eval_query_metadata).strip() else None
    weights = [float(item.strip()) for item in str(args.fusion_weights).split(",") if item.strip()]

    train_groups, train_stats = _collect_candidate_groups(
        cfg_path=Path(args.config),
        images_dir=Path(args.images_dir),
        metadata_path=Path(args.metadata),
        query_metadata_path=query_metadata,
        limit=int(args.limit),
        seed=int(args.seed),
        model_id=str(args.embedding_model),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    model, means, scales, train_report = _train_model(
        groups=train_groups,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        target_sigma_km=float(args.target_sigma_km),
        seed=int(args.seed),
    )
    eval_groups, eval_stats = _collect_candidate_groups(
        cfg_path=Path(args.config),
        images_dir=Path(args.images_dir),
        metadata_path=eval_metadata,
        query_metadata_path=eval_query_metadata,
        limit=int(args.eval_limit),
        seed=int(args.seed),
        model_id=str(args.embedding_model),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    eval_report = _evaluate(
        groups=eval_groups,
        model=model,
        means=means,
        scales=scales,
        weights=weights,
        retrieval_temperature=float(args.retrieval_temperature),
    )
    report = {
        "train": train_stats,
        "eval_data": eval_stats,
        "training": train_report,
        "eval": eval_report,
    }
    _save_model(Path(args.output), model, means, scales, report)
    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
