"""Train a street-query to aerial-reference projection head."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from src.core.geo.retrieval_provider import ClipEmbedder
from src.tools import train_retrieval_projection as baseproj

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


@dataclass(frozen=True)
class TripletRow:
    query_idx: int
    positive_indices: Tuple[int, ...]
    negative_indices: Tuple[int, ...]
    sample_weight: float


class QueryProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(input_dim), int(output_dim), bias=True)
        with torch.no_grad():
            eye = torch.eye(int(output_dim), int(input_dim), dtype=self.linear.weight.dtype)
            self.linear.weight.copy_(eye)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(x), dim=-1, eps=1e-12)


def _load_initial_projection(
    path: Optional[Path],
    *,
    input_dim: int,
    output_dim: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if path is None or not str(path).strip():
        return None
    with np.load(path, allow_pickle=False) as payload:
        matrix = np.asarray(payload["matrix"], dtype=np.float32)
        bias = np.asarray(payload["bias"], dtype=np.float32)
    expected_matrix = (int(output_dim), int(input_dim))
    expected_bias = (int(output_dim),)
    if matrix.shape != expected_matrix or bias.shape != expected_bias:
        raise ValueError(
            "initial projection shape mismatch: "
            f"matrix={matrix.shape} expected={expected_matrix}, "
            f"bias={bias.shape} expected={expected_bias}"
        )
    return matrix, bias


def _safe_float(value: object) -> Optional[float]:
    try:
        num = float(value)
    except Exception:
        return None
    if not math.isfinite(num):
        return None
    return num


def _resolve_query_image(images_dir: Path, rel_path: str) -> Optional[Path]:
    rel = str(rel_path).replace("\\", "/")
    direct = images_dir / rel
    if direct.exists():
        return direct
    parent = images_dir.parent / rel
    if parent.exists():
        return parent
    by_name = images_dir / Path(rel).name
    if by_name.exists():
        return by_name
    return None


def _normalize_device(device: str) -> str:
    raw = str(device).strip().lower()
    if raw == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if raw.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        return "cpu"
    return raw or "cpu"


def _load_query_embeddings(
    *,
    query_paths: Sequence[str],
    street_images_dir: Path,
    model_id: str,
    device: str,
    batch_size: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    unique: List[str] = []
    seen = set()
    for item in query_paths:
        key = Path(str(item)).as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)

    embedder = ClipEmbedder(model_id=model_id, device=_normalize_device(device), projection_path=None)
    by_exact: Dict[str, np.ndarray] = {}
    by_name: Dict[str, np.ndarray] = {}
    for begin in range(0, len(unique), max(1, int(batch_size))):
        batch_keys = unique[begin : begin + max(1, int(batch_size))]
        batch_images: List[Image.Image] = []
        batch_norm_paths: List[str] = []
        for key in batch_keys:
            resolved = _resolve_query_image(street_images_dir, key)
            if resolved is None or not resolved.exists():
                continue
            with Image.open(resolved) as img:
                batch_images.append(img.convert("RGB"))
            batch_norm_paths.append(key)
        if not batch_images:
            continue
        vectors = embedder.embed_many(batch_images)
        for norm_path, vec in zip(batch_norm_paths, vectors):
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            by_exact[norm_path] = arr
            by_name.setdefault(Path(norm_path).name, arr)
        if (begin // max(1, int(batch_size))) % 10 == 0:
            print(f"Embedded {begin + len(batch_keys)}/{len(unique)} query images...")
    return by_exact, by_name


def _lookup_embedding(key: str, by_exact: Dict[str, np.ndarray], by_name: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    as_key = Path(str(key)).as_posix()
    hit = by_exact.get(as_key)
    if hit is not None:
        return hit
    return by_name.get(Path(as_key).name)


def _build_training_records(
    *,
    triplets: Sequence[dict],
    query_embeddings_exact: Dict[str, np.ndarray],
    query_embeddings_name: Dict[str, np.ndarray],
    aerial_exact: Dict[str, np.ndarray],
    aerial_name: Dict[str, np.ndarray],
    sample_weight_mode: str,
    sample_weight_power: float,
    sample_weight_max: float,
) -> tuple[np.ndarray, np.ndarray, List[TripletRow], dict]:
    query_vectors: List[np.ndarray] = []
    query_map: Dict[str, int] = {}
    ref_vectors: List[np.ndarray] = []
    ref_map: Dict[str, int] = {}
    rows: List[TripletRow] = []
    dropped_missing = 0
    explicit_weights = 0
    sample_weights: List[float] = []

    def ensure_query(path: str) -> Optional[int]:
        key = Path(str(path)).as_posix()
        existing = query_map.get(key)
        if existing is not None:
            return existing
        vec = _lookup_embedding(key, query_embeddings_exact, query_embeddings_name)
        if vec is None:
            return None
        idx = len(query_vectors)
        query_vectors.append(np.asarray(vec, dtype=np.float32))
        query_map[key] = idx
        return idx

    def ensure_ref(path: str) -> Optional[int]:
        key = Path(str(path)).as_posix()
        existing = ref_map.get(key)
        if existing is not None:
            return existing
        vec = _lookup_embedding(key, aerial_exact, aerial_name)
        if vec is None:
            return None
        idx = len(ref_vectors)
        ref_vectors.append(np.asarray(vec, dtype=np.float32))
        ref_map[key] = idx
        return idx

    for row in triplets:
        q_path = str(row.get("query_path") or "").strip()
        q_idx = ensure_query(q_path)
        if q_idx is None:
            dropped_missing += 1
            continue
        pos_indices = []
        for item in row.get("positives", []):
            p_idx = ensure_ref(str((item or {}).get("path") or "").strip())
            if p_idx is not None:
                pos_indices.append(p_idx)
        neg_indices = []
        for item in row.get("hard_negatives", []):
            n_idx = ensure_ref(str((item or {}).get("path") or "").strip())
            if n_idx is not None:
                neg_indices.append(n_idx)
        if not pos_indices or not neg_indices:
            dropped_missing += 1
            continue
        weight, explicit = baseproj._resolve_triplet_weight(
            row,
            sample_weight_mode=sample_weight_mode,
            sample_weight_power=sample_weight_power,
            sample_weight_max=sample_weight_max,
        )
        if explicit:
            explicit_weights += 1
        sample_weights.append(float(weight))
        rows.append(
            TripletRow(
                query_idx=int(q_idx),
                positive_indices=tuple(pos_indices),
                negative_indices=tuple(neg_indices),
                sample_weight=float(weight),
            )
        )

    if not query_vectors or not ref_vectors or not rows:
        raise ValueError("no_valid_crossview_training_records")
    q_mat = np.asarray(query_vectors, dtype=np.float32)
    r_mat = np.asarray(ref_vectors, dtype=np.float32)
    q_mat = q_mat / np.linalg.norm(q_mat, axis=1, keepdims=True).clip(min=1e-12)
    r_mat = r_mat / np.linalg.norm(r_mat, axis=1, keepdims=True).clip(min=1e-12)
    stats = {
        "triplets_used": len(rows),
        "queries_embedded": int(q_mat.shape[0]),
        "references_used": int(r_mat.shape[0]),
        "embedding_dim": int(q_mat.shape[1]),
        "dropped_missing": int(dropped_missing),
        "triplets_with_explicit_weight": int(explicit_weights),
        "sample_weight_mean": float(sum(sample_weights) / len(sample_weights)) if sample_weights else None,
        "sample_weight_max": max(sample_weights) if sample_weights else None,
    }
    return q_mat, r_mat, rows, stats


def _evaluate_rows(model, q_base: torch.Tensor, r_base: torch.Tensor, rows: Sequence[TripletRow], margin: float) -> dict:
    gaps: List[float] = []
    losses: List[float] = []
    weights: List[float] = []
    with torch.no_grad():
        for row in rows:
            q = model(q_base[row.query_idx : row.query_idx + 1])
            pos = r_base[list(row.positive_indices)]
            neg = r_base[list(row.negative_indices)]
            sim_pos = torch.matmul(q, pos.transpose(0, 1)).squeeze(0)
            sim_neg = torch.matmul(q, neg.transpose(0, 1)).squeeze(0)
            hard_pos = torch.min(sim_pos)
            hard_neg = torch.max(sim_neg)
            gap = float(hard_pos.item() - hard_neg.item())
            gaps.append(gap)
            losses.append(float(torch.relu(torch.tensor(float(margin), device=q.device) - (hard_pos - hard_neg)).item()))
            weights.append(float(max(1e-6, row.sample_weight)))
    if not gaps:
        return {"count": 0, "triplet_satisfied_pct": 0.0, "weighted_triplet_satisfied_pct": 0.0, "weighted_hard_triplet_loss": None}
    total_weight = float(sum(weights))
    satisfied = [gap >= float(margin) for gap in gaps]
    return {
        "count": len(gaps),
        "mean_gap": float(sum(gaps) / len(gaps)),
        "triplet_satisfied_pct": 100.0 * float(sum(1 for item in satisfied if item)) / float(len(satisfied)),
        "weighted_triplet_satisfied_pct": (
            100.0 * float(sum(weight for ok, weight in zip(satisfied, weights) if ok)) / float(total_weight)
            if total_weight > 0.0
            else 0.0
        ),
        "weighted_hard_triplet_loss": (
            float(sum(weight * loss for weight, loss in zip(weights, losses)) / total_weight) if total_weight > 0.0 else None
        ),
    }


def train_crossview_projection(
    *,
    query_embeddings: np.ndarray,
    aerial_embeddings: np.ndarray,
    rows: Sequence[TripletRow],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    margin: float,
    temperature: float,
    ce_weight: float,
    seed: int,
    device: str,
    initial_projection: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if torch is None or nn is None or F is None:
        raise RuntimeError("torch_not_available")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    use_device = str(device).strip().lower()
    if use_device == "auto":
        use_device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_device.startswith("cuda") and not torch.cuda.is_available():
        use_device = "cpu"
    torch_device = torch.device(use_device)

    q_base = torch.from_numpy(np.asarray(query_embeddings, dtype=np.float32)).to(torch_device)
    r_base = torch.from_numpy(np.asarray(aerial_embeddings, dtype=np.float32)).to(torch_device)
    q_base = F.normalize(q_base, dim=-1, eps=1e-12)
    r_base = F.normalize(r_base, dim=-1, eps=1e-12)

    model = QueryProjectionHead(input_dim=int(q_base.shape[1]), output_dim=int(r_base.shape[1])).to(torch_device)
    if initial_projection is not None:
        init_weight, init_bias = initial_projection
        with torch.no_grad():
            model.linear.weight.copy_(torch.as_tensor(init_weight, dtype=model.linear.weight.dtype, device=torch_device))
            model.linear.bias.copy_(torch.as_tensor(init_bias, dtype=model.linear.bias.dtype, device=torch_device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(max(0.0, weight_decay)))

    history: List[dict] = []
    start = time.perf_counter()
    order_rng = random.Random(int(seed))
    temp = max(1e-4, float(temperature))

    for epoch in range(1, int(epochs) + 1):
        order = list(range(len(rows)))
        order_rng.shuffle(order)
        model.train()
        train_losses: List[float] = []
        for begin in range(0, len(order), max(1, int(batch_size))):
            batch_ids = order[begin : begin + max(1, int(batch_size))]
            if not batch_ids:
                continue
            optimizer.zero_grad(set_to_none=True)
            losses = []
            weights = []
            for idx in batch_ids:
                row = rows[idx]
                q = model(q_base[row.query_idx : row.query_idx + 1])
                pos = r_base[list(row.positive_indices)]
                neg = r_base[list(row.negative_indices)]
                sim_pos = torch.matmul(q, pos.transpose(0, 1)).squeeze(0)
                sim_neg = torch.matmul(q, neg.transpose(0, 1)).squeeze(0)
                hard_pos = torch.min(sim_pos)
                hard_neg = torch.max(sim_neg)
                triplet_loss = F.relu(torch.tensor(float(margin), device=torch_device) + hard_neg - hard_pos)
                pos_logit = torch.max(sim_pos)
                logits = torch.cat([pos_logit.view(1), sim_neg], dim=0).view(1, -1) / temp
                ce = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=torch_device))
                losses.append(triplet_loss + (float(max(0.0, ce_weight)) * ce))
                weights.append(float(max(1e-6, row.sample_weight)))
            loss_tensor = torch.stack(losses)
            weight_tensor = torch.tensor(weights, dtype=loss_tensor.dtype, device=loss_tensor.device)
            batch_loss = (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-12)
            batch_loss.backward()
            optimizer.step()
            train_losses.append(float(batch_loss.detach().cpu().item()))

        model.eval()
        eval_stats = _evaluate_rows(model, q_base, r_base, rows, margin=float(margin))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss_mean": float(sum(train_losses) / len(train_losses)) if train_losses else None,
                "eval": eval_stats,
            }
        )

    elapsed = float(time.perf_counter() - start)
    with torch.no_grad():
        weight = model.linear.weight.detach().cpu().numpy().astype(np.float32)
        bias = model.linear.bias.detach().cpu().numpy().astype(np.float32)
    report = {
        "device": str(torch_device),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "margin": float(margin),
        "temperature": float(temp),
        "ce_weight": float(max(0.0, ce_weight)),
        "initialized_from_projection": initial_projection is not None,
        "elapsed_sec": elapsed,
        "history": history,
    }
    return weight, bias, report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a query-only street-to-aerial projection.")
    parser.add_argument("--triplets", required=True)
    parser.add_argument("--aerial-index", required=True)
    parser.add_argument("--street-images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", default="")
    parser.add_argument("--init-projection", default="", help="Optional existing query projection to fine-tune from.")
    parser.add_argument("--embedding-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--max-triplets", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--sample-weight-mode", default="triplet_weight", choices=["auto", "uniform", "triplet_weight"])
    parser.add_argument("--sample-weight-power", type=float, default=1.0)
    parser.add_argument("--sample-weight-max", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    triplets = baseproj._load_triplets(Path(args.triplets), int(args.max_triplets))
    if not triplets:
        raise ValueError("triplets_empty")
    aerial_exact, aerial_name, _ = baseproj._load_index_embeddings(Path(args.aerial_index))
    query_paths = [str(row.get("query_path") or "").strip() for row in triplets if str(row.get("query_path") or "").strip()]
    query_exact, query_name = _load_query_embeddings(
        query_paths=query_paths,
        street_images_dir=Path(args.street_images_dir),
        model_id=str(args.embedding_model),
        device=str(args.device),
        batch_size=max(8, int(args.batch_size)),
    )
    q_mat, r_mat, rows, dataset_stats = _build_training_records(
        triplets=triplets,
        query_embeddings_exact=query_exact,
        query_embeddings_name=query_name,
        aerial_exact=aerial_exact,
        aerial_name=aerial_name,
        sample_weight_mode=str(args.sample_weight_mode),
        sample_weight_power=float(args.sample_weight_power),
        sample_weight_max=float(args.sample_weight_max),
    )
    initial_projection = _load_initial_projection(
        Path(args.init_projection) if str(args.init_projection).strip() else None,
        input_dim=int(q_mat.shape[1]),
        output_dim=int(r_mat.shape[1]),
    )
    weight, bias, train_report = train_crossview_projection(
        query_embeddings=q_mat,
        aerial_embeddings=r_mat,
        rows=rows,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        margin=float(args.margin),
        temperature=float(args.temperature),
        ce_weight=float(args.ce_weight),
        seed=int(args.seed),
        device=str(args.device),
        initial_projection=initial_projection,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        matrix=weight,
        bias=bias,
        model_id=np.asarray(str(args.embedding_model), dtype=np.str_),
        projection_role=np.asarray("query_only_street_to_aerial", dtype=np.str_),
    )
    report = {
        "triplets_loaded": len(triplets),
        "triplets_used": len(rows),
        "street_images_dir": str(Path(args.street_images_dir)),
        "aerial_index": str(Path(args.aerial_index)),
        "embedding_model": str(args.embedding_model),
        "init_projection": str(args.init_projection) if str(args.init_projection).strip() else None,
        "dataset_stats": dataset_stats,
        "training": train_report,
    }
    report_path = Path(args.report_output) if str(args.report_output).strip() else output_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "report": str(report_path), "triplets_used": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
