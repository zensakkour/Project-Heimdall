"""
Train a lightweight retrieval projection head from hard-negative triplets.

The projection can be applied during index build and query embedding:
- build index with: `src.tools.build_geo_index --projection-path ...`
- evaluate/query with config: `geolocator.retrieval_projection_path`
"""
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

from src.core.geo.retrieval_provider import ClipEmbedder, load_index

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None


@dataclass(frozen=True)
class TripletRecord:
    query_idx: int
    positive_indices: Tuple[int, ...]
    negative_indices: Tuple[int, ...]


def _as_posix(path: str) -> str:
    return Path(str(path)).as_posix()


def _resolve_image_path(images_dir: Path, rel_path: str) -> Optional[Path]:
    rel = str(rel_path).replace("\\", "/")
    direct = images_dir / rel
    if direct.exists():
        return direct
    parent = images_dir.parent / rel
    if parent.exists():
        return parent
    if rel.startswith("chips/"):
        trimmed = rel.split("chips/", 1)[1]
        candidate = images_dir / trimmed
        if candidate.exists():
            return candidate
    by_name = images_dir / Path(rel).name
    if by_name.exists():
        return by_name
    return None


def _load_triplets(path: Path, max_triplets: int) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        q = row.get("query_path")
        positives = row.get("positives")
        negatives = row.get("hard_negatives")
        if not q or not isinstance(positives, list) or not isinstance(negatives, list):
            continue
        rows.append(row)
    if max_triplets > 0:
        rows = rows[: max_triplets]
    return rows


def _collect_requested_paths(triplets: Sequence[dict]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for row in triplets:
        q = str(row.get("query_path") or "").strip()
        if q:
            key = _as_posix(q)
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        for item in row.get("positives", []):
            p = str((item or {}).get("path") or "").strip()
            if p:
                key = _as_posix(p)
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        for item in row.get("hard_negatives", []):
            n = str((item or {}).get("path") or "").strip()
            if n:
                key = _as_posix(n)
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
    return ordered


def _load_index_embeddings(index_path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    idx = load_index(index_path)
    by_exact: Dict[str, np.ndarray] = {}
    by_name: Dict[str, np.ndarray] = {}
    for i in range(idx.embeddings.shape[0]):
        vec = np.asarray(idx.embeddings[i], dtype=np.float32)
        key = _as_posix(str(idx.paths[i]))
        by_exact.setdefault(key, vec)
        by_name.setdefault(Path(key).name, vec)
    dim = int(idx.embeddings.shape[1])
    return by_exact, by_name, dim


def _lookup_embedding(
    key: str,
    *,
    by_exact: Dict[str, np.ndarray],
    by_name: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    as_key = _as_posix(key)
    hit = by_exact.get(as_key)
    if hit is not None:
        return hit
    by_base = by_name.get(Path(as_key).name)
    if by_base is not None:
        by_exact[as_key] = by_base
    return by_base


def _embed_missing(
    *,
    requested_paths: Sequence[str],
    by_exact: Dict[str, np.ndarray],
    by_name: Dict[str, np.ndarray],
    model_id: str,
    images_dir: Path,
    device: str,
) -> tuple[int, int]:
    missing = [path for path in requested_paths if _lookup_embedding(path, by_exact=by_exact, by_name=by_name) is None]
    if not missing:
        return 0, 0
    embedder = ClipEmbedder(model_id=model_id, device=device, projection_path=None)
    written = 0
    unresolved = 0
    for rel in missing:
        resolved = _resolve_image_path(images_dir, rel)
        if resolved is None or not resolved.exists():
            unresolved += 1
            continue
        with Image.open(resolved) as img:
            vec = embedder.embed(img.convert("RGB")).reshape(-1).astype(np.float32)
        by_exact[_as_posix(rel)] = vec
        by_exact[_as_posix(str(resolved))] = vec
        by_name[resolved.name] = vec
        written += 1
    return written, unresolved


def _build_training_records(
    triplets: Sequence[dict],
    *,
    by_exact: Dict[str, np.ndarray],
    by_name: Dict[str, np.ndarray],
) -> tuple[np.ndarray, List[TripletRecord], dict]:
    path_to_idx: Dict[str, int] = {}
    vectors: List[np.ndarray] = []

    def ensure_idx(path: str) -> Optional[int]:
        key = _as_posix(path)
        existing = path_to_idx.get(key)
        if existing is not None:
            return existing
        vec = _lookup_embedding(key, by_exact=by_exact, by_name=by_name)
        if vec is None:
            return None
        idx = len(vectors)
        path_to_idx[key] = idx
        vectors.append(np.asarray(vec, dtype=np.float32))
        return idx

    used_rows: List[TripletRecord] = []
    dropped_missing = 0
    dropped_structure = 0
    for row in triplets:
        q_path = str(row.get("query_path") or "").strip()
        if not q_path:
            dropped_structure += 1
            continue
        q_idx = ensure_idx(q_path)
        if q_idx is None:
            dropped_missing += 1
            continue

        pos_ids: List[int] = []
        for item in row.get("positives", []):
            p = str((item or {}).get("path") or "").strip()
            if not p:
                continue
            p_idx = ensure_idx(p)
            if p_idx is None or p_idx == q_idx:
                continue
            if p_idx not in pos_ids:
                pos_ids.append(p_idx)

        neg_ids: List[int] = []
        for item in row.get("hard_negatives", []):
            n = str((item or {}).get("path") or "").strip()
            if not n:
                continue
            n_idx = ensure_idx(n)
            if n_idx is None or n_idx == q_idx:
                continue
            if n_idx not in neg_ids:
                neg_ids.append(n_idx)

        if not pos_ids or not neg_ids:
            dropped_structure += 1
            continue
        used_rows.append(
            TripletRecord(
                query_idx=int(q_idx),
                positive_indices=tuple(int(v) for v in pos_ids),
                negative_indices=tuple(int(v) for v in neg_ids),
            )
        )

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), [], {
            "dropped_missing": dropped_missing,
            "dropped_structure": dropped_structure,
        }

    mat = np.vstack(vectors).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-12, None)
    stats = {
        "dropped_missing": dropped_missing,
        "dropped_structure": dropped_structure,
        "unique_embeddings": int(mat.shape[0]),
        "embedding_dim": int(mat.shape[1]),
    }
    return mat, used_rows, stats


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        with torch.no_grad():
            self.linear.weight.zero_()
            eye = min(input_dim, output_dim)
            self.linear.weight[:eye, :eye] = torch.eye(eye, dtype=torch.float32)
            self.linear.bias.zero_()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        y = self.linear(x)
        return F.normalize(y, dim=-1, eps=1e-12)


def _evaluate_margin_stats(
    *,
    model: ProjectionHead,
    base_embeddings: "torch.Tensor",
    rows: Sequence[TripletRecord],
    margin: float,
) -> dict:
    if not rows:
        return {
            "count": 0,
            "mean_gap": None,
            "median_gap": None,
            "p90_gap": None,
            "triplet_satisfied_pct": 0.0,
            "hard_triplet_loss": None,
        }
    gaps: List[float] = []
    losses: List[float] = []
    with torch.no_grad():
        for row in rows:
            q = model(base_embeddings[row.query_idx : row.query_idx + 1])
            pos = model(base_embeddings[list(row.positive_indices)])
            neg = model(base_embeddings[list(row.negative_indices)])
            sim_pos = torch.matmul(q, pos.transpose(0, 1)).squeeze(0)
            sim_neg = torch.matmul(q, neg.transpose(0, 1)).squeeze(0)
            hard_pos = torch.min(sim_pos)
            hard_neg = torch.max(sim_neg)
            gap = float(hard_pos.item() - hard_neg.item())
            gaps.append(gap)
            losses.append(float(F.relu(torch.tensor(float(margin), device=q.device) - (hard_pos - hard_neg)).item()))
    gaps_sorted = sorted(gaps)
    n = len(gaps_sorted)

    def pct(v: float) -> float:
        return 100.0 * float(sum(1 for x in gaps if x >= v)) / float(n)

    def percentile(p: float) -> float:
        idx = max(0, min(n - 1, int(round((p / 100.0) * (n - 1)))))
        return float(gaps_sorted[idx])

    return {
        "count": n,
        "mean_gap": float(sum(gaps) / n),
        "median_gap": percentile(50.0),
        "p90_gap": percentile(90.0),
        "triplet_satisfied_pct": pct(float(margin)),
        "hard_triplet_loss": float(sum(losses) / n),
    }


def train_projection(
    *,
    embeddings: np.ndarray,
    rows: Sequence[TripletRecord],
    output_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    margin: float,
    temperature: float,
    ce_weight: float,
    orth_weight: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if torch is None or nn is None or F is None:
        raise RuntimeError("torch_not_available")
    if embeddings.ndim != 2 or embeddings.shape[0] <= 0:
        raise ValueError("embeddings_empty")
    if not rows:
        raise ValueError("triplets_empty")

    input_dim = int(embeddings.shape[1])
    out_dim = int(output_dim) if int(output_dim) > 0 else input_dim
    if out_dim <= 0:
        raise ValueError("output_dim_invalid")

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    use_device = str(device).strip().lower()
    if use_device == "auto":
        use_device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_device.startswith("cuda") and not torch.cuda.is_available():
        use_device = "cpu"
    torch_device = torch.device(use_device)

    base = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(torch_device)
    base = F.normalize(base, dim=-1, eps=1e-12)
    model = ProjectionHead(input_dim=input_dim, output_dim=out_dim).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(max(0.0, weight_decay)),
    )

    history: List[dict] = []
    start = time.perf_counter()
    n_rows = len(rows)
    batch_n = max(1, int(batch_size))
    temp = max(1e-4, float(temperature))
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = list(range(n_rows))
        random.shuffle(order)
        losses: List[float] = []
        for begin in range(0, n_rows, batch_n):
            batch_ids = order[begin : begin + batch_n]
            if not batch_ids:
                continue
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.tensor(0.0, dtype=torch.float32, device=torch_device)
            valid_count = 0
            for idx in batch_ids:
                row = rows[idx]
                q = model(base[row.query_idx : row.query_idx + 1])
                pos = model(base[list(row.positive_indices)])
                neg = model(base[list(row.negative_indices)])
                sim_pos = torch.matmul(q, pos.transpose(0, 1)).squeeze(0)
                sim_neg = torch.matmul(q, neg.transpose(0, 1)).squeeze(0)
                hard_pos = torch.min(sim_pos)
                hard_neg = torch.max(sim_neg)
                tri = F.relu(torch.tensor(float(margin), device=torch_device) + hard_neg - hard_pos)
                pos_logit = torch.max(sim_pos)
                logits = torch.cat([pos_logit.view(1), sim_neg], dim=0).view(1, -1) / temp
                ce = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=torch_device))
                loss = tri + float(max(0.0, ce_weight)) * ce
                batch_loss = batch_loss + loss
                valid_count += 1

            if valid_count <= 0:
                continue
            batch_loss = batch_loss / float(valid_count)
            if orth_weight > 0.0 and input_dim == out_dim:
                w = model.linear.weight
                eye = torch.eye(out_dim, dtype=w.dtype, device=w.device)
                orth = torch.mean((torch.matmul(w, w.transpose(0, 1)) - eye) ** 2)
                batch_loss = batch_loss + float(orth_weight) * orth
            batch_loss.backward()
            optimizer.step()
            losses.append(float(batch_loss.detach().cpu().item()))

        model.eval()
        eval_stats = _evaluate_margin_stats(model=model, base_embeddings=base, rows=rows, margin=float(margin))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss_mean": (float(sum(losses) / len(losses)) if losses else None),
                "eval": eval_stats,
            }
        )

    elapsed = float(time.perf_counter() - start)
    with torch.no_grad():
        weight = model.linear.weight.detach().cpu().numpy().astype(np.float32)
        bias = model.linear.bias.detach().cpu().numpy().astype(np.float32)
    report = {
        "device": str(torch_device),
        "input_dim": input_dim,
        "output_dim": out_dim,
        "epochs": int(epochs),
        "rows": int(n_rows),
        "batch_size": int(batch_n),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "margin": float(margin),
        "temperature": float(temp),
        "ce_weight": float(ce_weight),
        "orth_weight": float(orth_weight),
        "elapsed_sec": elapsed,
        "history": history,
    }
    return weight, bias, report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train retrieval projection from hard-negative triplets.")
    parser.add_argument("--triplets", required=True, help="JSONL from mine_hard_negative_triplets.")
    parser.add_argument("--output", default="runs/retrieval_projection.npz", help="Output projection .npz")
    parser.add_argument("--report-output", default="", help="Optional report JSON output path.")
    parser.add_argument("--embedding-index", default="", help="Optional index .npz to bootstrap embeddings.")
    parser.add_argument("--images-dir", default="data/spacenet_paris_test/chips", help="Image root for missing embeds.")
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14", help="HF model ID for missing embeds.")
    parser.add_argument("--max-triplets", type=int, default=0, help="Limit loaded triplets (0=all).")
    parser.add_argument("--output-dim", type=int, default=0, help="Projection output dim (0=input dim).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--orth-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = parser.parse_args(argv)

    triplet_path = Path(args.triplets)
    if not triplet_path.exists():
        raise FileNotFoundError(f"triplets_not_found:{triplet_path}")
    triplets = _load_triplets(triplet_path, max_triplets=int(args.max_triplets))
    if not triplets:
        raise ValueError("triplets_empty_or_invalid")
    requested_paths = _collect_requested_paths(triplets)

    by_exact: Dict[str, np.ndarray] = {}
    by_name: Dict[str, np.ndarray] = {}
    input_dim = 0
    if str(args.embedding_index).strip():
        idx_path = Path(args.embedding_index)
        if not idx_path.exists():
            raise FileNotFoundError(f"embedding_index_not_found:{idx_path}")
        by_exact, by_name, input_dim = _load_index_embeddings(idx_path)

    embed_written = 0
    embed_missing = 0
    if requested_paths:
        if torch is None:
            # Without torch we can only continue if index covered everything.
            still_missing = sum(
                1
                for path in requested_paths
                if _lookup_embedding(path, by_exact=by_exact, by_name=by_name) is None
            )
            if still_missing > 0:
                raise RuntimeError("torch_not_available_for_missing_embeddings")
        else:
            use_device = str(args.device).strip().lower()
            if use_device == "auto":
                use_device = "cuda" if torch.cuda.is_available() else "cpu"
            images_dir = Path(args.images_dir)
            if not images_dir.exists():
                raise FileNotFoundError(f"images_dir_not_found:{images_dir}")
            embed_written, embed_missing = _embed_missing(
                requested_paths=requested_paths,
                by_exact=by_exact,
                by_name=by_name,
                model_id=str(args.model_id),
                images_dir=images_dir,
                device=use_device,
            )

    matrix, rows, ds_stats = _build_training_records(triplets, by_exact=by_exact, by_name=by_name)
    if matrix.size <= 0 or not rows:
        raise ValueError("no_valid_training_records")
    input_dim = int(matrix.shape[1])
    output_dim = int(args.output_dim) if int(args.output_dim) > 0 else input_dim

    weight, bias, train_report = train_projection(
        embeddings=matrix,
        rows=rows,
        output_dim=output_dim,
        epochs=int(max(1, args.epochs)),
        batch_size=int(max(1, args.batch_size)),
        learning_rate=float(max(1e-6, args.learning_rate)),
        weight_decay=float(max(0.0, args.weight_decay)),
        margin=float(max(0.0, args.margin)),
        temperature=float(max(1e-4, args.temperature)),
        ce_weight=float(max(0.0, args.ce_weight)),
        orth_weight=float(max(0.0, args.orth_weight)),
        seed=int(args.seed),
        device=str(args.device),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        matrix=weight,
        bias=bias,
        input_dim=np.asarray(int(input_dim), dtype=np.int32),
        output_dim=np.asarray(int(output_dim), dtype=np.int32),
        model_id=np.asarray(str(args.model_id), dtype=np.str_),
        trained_at_unix=np.asarray(time.time(), dtype=np.float64),
    )

    report = {
        "triplets_path": str(triplet_path),
        "triplets_loaded": len(triplets),
        "triplets_used": len(rows),
        "requested_unique_paths": len(requested_paths),
        "embedded_from_images": int(embed_written),
        "missing_after_embed": int(embed_missing),
        "dataset_stats": ds_stats,
        "training": train_report,
        "projection_path": str(output_path),
    }
    report_path = Path(args.report_output) if str(args.report_output).strip() else output_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    final_eval = train_report.get("history", [])[-1].get("eval", {}) if train_report.get("history") else {}
    print(
        "trained projection "
        + f"(triplets_used={len(rows)}, input_dim={input_dim}, output_dim={output_dim}, "
        + f"triplet_satisfied_pct={final_eval.get('triplet_satisfied_pct')})"
    )
    print(f"projection -> {output_path}")
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
