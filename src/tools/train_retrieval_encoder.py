"""
Fine-tune an image retrieval encoder directly from hard-negative triplets.

This trains a CLIP-family image encoder on resolved query/positive/negative
images and saves a local `save_pretrained()` model directory that can be used
by `build_geo_index` or `GeoRetrievalProvider` as a normal `model_id`.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from src.core.geo import retrieval_provider as rp
from src.tools import train_retrieval_projection as projection_tools

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


@dataclass(frozen=True)
class ResolvedTriplet:
    query_path: str
    positive_paths: Tuple[str, ...]
    negative_paths: Tuple[str, ...]
    sample_weight: float


def _safe_float(value: object) -> Optional[float]:
    try:
        num = float(value)
    except Exception:
        return None
    if not num == num or num in {float("inf"), float("-inf")}:
        return None
    return num


def _normalize_device(device: str) -> str:
    if rp.torch is None:
        raise RuntimeError("torch_not_available")
    raw = str(device).strip().lower()
    if raw == "auto":
        return "cuda" if rp.torch.cuda.is_available() else "cpu"
    if raw.startswith("cuda") and not rp.torch.cuda.is_available():
        return "cpu"
    return raw or "cpu"


def _load_pretrained(factory, model_id: str):
    target = str(model_id).strip()
    try:
        return factory.from_pretrained(target)
    except Exception:
        if Path(target).exists():
            raise
        return factory.from_pretrained(target, local_files_only=True)


def _resolve_image_path(path: str, *, primary_dir: Path, fallback_dir: Optional[Path]) -> Optional[Path]:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    resolved = projection_tools._resolve_image_path(primary_dir, path)
    if resolved is not None and resolved.exists():
        return resolved
    if fallback_dir is not None:
        resolved = projection_tools._resolve_image_path(fallback_dir, path)
        if resolved is not None and resolved.exists():
            return resolved
    return None


def _resolve_triplets(
    *,
    triplets: Sequence[dict],
    query_images_dir: Path,
    reference_images_dir: Path,
    max_triplets: int,
    max_positives_per_row: int,
    max_negatives_per_row: int,
    sample_weight_mode: str,
    sample_weight_power: float,
    sample_weight_max: float,
) -> tuple[List[ResolvedTriplet], dict]:
    resolved: List[ResolvedTriplet] = []
    dropped_missing_query = 0
    dropped_missing_refs = 0

    rows = list(triplets)
    if max_triplets > 0:
        rows = rows[: max_triplets]

    for row in rows:
        q_path = str(row.get("query_path") or "").strip()
        if not q_path:
            dropped_missing_query += 1
            continue
        query_resolved = _resolve_image_path(
            q_path,
            primary_dir=query_images_dir,
            fallback_dir=reference_images_dir,
        )
        if query_resolved is None:
            dropped_missing_query += 1
            continue

        positives: List[str] = []
        for item in row.get("positives", []):
            p_path = str((item or {}).get("path") or "").strip()
            if not p_path:
                continue
            resolved_path = _resolve_image_path(
                p_path,
                primary_dir=reference_images_dir,
                fallback_dir=query_images_dir,
            )
            if resolved_path is None:
                continue
            positives.append(str(resolved_path))
            if len(positives) >= max(1, int(max_positives_per_row)):
                break

        negatives: List[str] = []
        for item in row.get("hard_negatives", []):
            n_path = str((item or {}).get("path") or "").strip()
            if not n_path:
                continue
            resolved_path = _resolve_image_path(
                n_path,
                primary_dir=reference_images_dir,
                fallback_dir=query_images_dir,
            )
            if resolved_path is None:
                continue
            negatives.append(str(resolved_path))
            if len(negatives) >= max(1, int(max_negatives_per_row)):
                break

        if not positives or not negatives:
            dropped_missing_refs += 1
            continue

        weight, _ = projection_tools._resolve_triplet_weight(
            row,
            sample_weight_mode=sample_weight_mode,
            sample_weight_power=sample_weight_power,
            sample_weight_max=sample_weight_max,
        )
        resolved.append(
            ResolvedTriplet(
                query_path=str(query_resolved),
                positive_paths=tuple(positives),
                negative_paths=tuple(negatives),
                sample_weight=float(weight),
            )
        )

    stats = {
        "triplets_loaded": len(rows),
        "triplets_resolved": len(resolved),
        "dropped_missing_query": int(dropped_missing_query),
        "dropped_missing_refs": int(dropped_missing_refs),
    }
    return resolved, stats


def _sample_batch_rows(
    rows: Sequence[ResolvedTriplet],
    *,
    batch_ids: Sequence[int],
    rng: random.Random,
) -> tuple[List[str], List[str], List[str], List[float]]:
    query_paths: List[str] = []
    positive_paths: List[str] = []
    negative_paths: List[str] = []
    weights: List[float] = []
    for idx in batch_ids:
        row = rows[int(idx)]
        query_paths.append(row.query_path)
        positive_paths.append(rng.choice(list(row.positive_paths)))
        negative_paths.append(rng.choice(list(row.negative_paths)))
        weights.append(float(max(1e-6, row.sample_weight)))
    return query_paths, positive_paths, negative_paths, weights


def _load_rgb_images(paths: Iterable[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def _image_features(model, processor, image_paths: Sequence[str], device: str):
    if rp.torch is None:
        raise RuntimeError("torch_not_available")
    images = _load_rgb_images(image_paths)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    if hasattr(model, "get_image_features"):
        feats = model.get_image_features(**inputs)
    else:
        outputs = model(**inputs)
        feats = rp._extract_tensor(outputs)
    feats = rp._extract_tensor(feats)
    return rp.torch.nn.functional.normalize(feats, dim=-1, eps=1e-12)


def _vision_pooled_features(model, processor, image_paths: Sequence[str], device: str):
    if rp.torch is None:
        raise RuntimeError("torch_not_available")
    if not hasattr(model, "vision_model"):
        raise ValueError("model_missing_vision_model")
    images = _load_rgb_images(image_paths)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
    pooled = getattr(vision_outputs, "pooler_output", None)
    if pooled is None and isinstance(vision_outputs, (tuple, list)) and len(vision_outputs) > 1:
        pooled = vision_outputs[1]
    pooled = rp._extract_tensor(pooled)
    if pooled is None:
        raise RuntimeError("vision_pooler_output_unavailable")
    return pooled


def _collect_unique_paths(rows: Sequence[ResolvedTriplet]) -> List[str]:
    seen: Dict[str, None] = {}
    ordered: List[str] = []
    for row in rows:
        for path in (row.query_path, *row.positive_paths, *row.negative_paths):
            norm = str(path).strip()
            if not norm or norm in seen:
                continue
            seen[norm] = None
            ordered.append(norm)
    return ordered


def _build_projection_feature_cache(
    *,
    model,
    processor,
    rows: Sequence[ResolvedTriplet],
    device: str,
    batch_size: int,
) -> Dict[str, object]:
    if rp.torch is None:
        raise RuntimeError("torch_not_available")
    unique_paths = _collect_unique_paths(rows)
    if not unique_paths:
        return {}
    cache: Dict[str, object] = {}
    model.eval()
    with rp.torch.no_grad():
        for begin in range(0, len(unique_paths), max(1, int(batch_size))):
            batch_paths = unique_paths[begin : begin + max(1, int(batch_size))]
            pooled = _vision_pooled_features(model, processor, batch_paths, device=device)
            pooled_cpu = pooled.detach().cpu()
            for path, vec in zip(batch_paths, pooled_cpu):
                cache[str(path)] = vec
    return cache


def _cached_projected_features(model, cache: Dict[str, object], image_paths: Sequence[str], device: str):
    if rp.torch is None:
        raise RuntimeError("torch_not_available")
    if not hasattr(model, "visual_projection"):
        raise ValueError("model_missing_visual_projection")
    weight = getattr(model.visual_projection, "weight", None)
    dtype = weight.dtype if weight is not None else None
    base = rp.torch.stack([cache[str(path)] for path in image_paths], dim=0)
    base = base.to(device=device, dtype=dtype)
    feats = model.visual_projection(base)
    return rp.torch.nn.functional.normalize(feats, dim=-1, eps=1e-12)


def _freeze_for_scope(model, train_scope: str) -> dict:
    total_params = 0
    trainable_params = 0
    scope = str(train_scope).strip().lower()
    for param in model.parameters():
        param.requires_grad = False
        total_params += int(param.numel())

    if scope == "all":
        for param in model.parameters():
            param.requires_grad = True
            trainable_params += int(param.numel())
    elif scope == "visual_projection":
        if not hasattr(model, "visual_projection"):
            raise ValueError("model_missing_visual_projection")
        for param in model.visual_projection.parameters():
            param.requires_grad = True
            trainable_params += int(param.numel())
    elif scope == "vision_encoder":
        if not hasattr(model, "vision_model"):
            raise ValueError("model_missing_vision_model")
        for param in model.vision_model.parameters():
            param.requires_grad = True
            trainable_params += int(param.numel())
        if hasattr(model, "visual_projection"):
            for param in model.visual_projection.parameters():
                if not param.requires_grad:
                    trainable_params += int(param.numel())
                param.requires_grad = True
    else:
        raise ValueError(f"unsupported_train_scope:{train_scope}")

    return {
        "train_scope": scope,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def _evaluate_rows(
    *,
    model,
    processor,
    rows: Sequence[ResolvedTriplet],
    margin: float,
    device: str,
    feature_cache: Optional[Dict[str, object]] = None,
) -> dict:
    if rp.torch is None:
        raise RuntimeError("torch_not_available")
    gaps: List[float] = []
    losses: List[float] = []
    weights: List[float] = []

    with rp.torch.no_grad():
        for row in rows:
            if feature_cache:
                q = _cached_projected_features(model, feature_cache, [row.query_path], device=device)
                pos = _cached_projected_features(model, feature_cache, list(row.positive_paths), device=device)
                neg = _cached_projected_features(model, feature_cache, list(row.negative_paths), device=device)
            else:
                q = _image_features(model, processor, [row.query_path], device=device)
                pos = _image_features(model, processor, list(row.positive_paths), device=device)
                neg = _image_features(model, processor, list(row.negative_paths), device=device)
            sim_pos = rp.torch.matmul(q, pos.transpose(0, 1)).squeeze(0)
            sim_neg = rp.torch.matmul(q, neg.transpose(0, 1)).squeeze(0)
            hard_pos = rp.torch.min(sim_pos)
            hard_neg = rp.torch.max(sim_neg)
            gap = float(hard_pos.item() - hard_neg.item())
            gaps.append(gap)
            losses.append(float(rp.torch.relu(rp.torch.tensor(float(margin), device=q.device) - (hard_pos - hard_neg)).item()))
            weights.append(float(max(1e-6, row.sample_weight)))

    if not gaps:
        return {
            "count": 0,
            "mean_gap": None,
            "median_gap": None,
            "p90_gap": None,
            "triplet_satisfied_pct": 0.0,
            "weighted_triplet_satisfied_pct": 0.0,
            "hard_triplet_loss": None,
            "weighted_hard_triplet_loss": None,
            "weighted_mean_gap": None,
        }

    gaps_sorted = sorted(gaps)
    total_weight = float(sum(weights))

    def _pct(threshold: float) -> float:
        return 100.0 * float(sum(1 for value in gaps if value >= threshold)) / float(len(gaps))

    def _weighted_pct(threshold: float) -> float:
        kept = sum(weight for value, weight in zip(gaps, weights) if value >= threshold)
        return 100.0 * float(kept) / float(total_weight) if total_weight > 0.0 else 0.0

    def _percentile(pct: float) -> float:
        idx = int(round((pct / 100.0) * (len(gaps_sorted) - 1)))
        idx = max(0, min(len(gaps_sorted) - 1, idx))
        return float(gaps_sorted[idx])

    return {
        "count": len(gaps),
        "mean_gap": float(sum(gaps) / len(gaps)),
        "median_gap": _percentile(50.0),
        "p90_gap": _percentile(90.0),
        "triplet_satisfied_pct": _pct(float(margin)),
        "weighted_triplet_satisfied_pct": _weighted_pct(float(margin)),
        "hard_triplet_loss": float(sum(losses) / len(losses)),
        "weighted_hard_triplet_loss": (
            float(sum(weight * loss for weight, loss in zip(weights, losses)) / total_weight)
            if total_weight > 0.0
            else None
        ),
        "weighted_mean_gap": (
            float(sum(weight * value for weight, value in zip(weights, gaps)) / total_weight)
            if total_weight > 0.0
            else None
        ),
    }


def train_encoder(
    *,
    model_id: str,
    triplet_rows: Sequence[ResolvedTriplet],
    output_dir: Path,
    train_scope: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    margin: float,
    temperature: float,
    ce_weight: float,
    seed: int,
    device: str,
) -> dict:
    if rp.torch is None or rp.CLIPModel is None or rp.CLIPProcessor is None:
        raise RuntimeError("clip_training_dependencies_not_available")

    random.seed(int(seed))
    if np is not None:
        np.random.seed(int(seed))
    rp.torch.manual_seed(int(seed))

    resolved_device = _normalize_device(device)
    processor = _load_pretrained(rp.CLIPProcessor, model_id)
    model = _load_pretrained(rp.CLIPModel, model_id)
    model.to(resolved_device)
    freeze_stats = _freeze_for_scope(model, train_scope)
    feature_cache: Optional[Dict[str, object]] = None
    cache_stats: Optional[dict] = None
    if str(train_scope).strip().lower() == "visual_projection":
        cache_start = time.perf_counter()
        feature_cache = _build_projection_feature_cache(
            model=model,
            processor=processor,
            rows=triplet_rows,
            device=resolved_device,
            batch_size=max(8, int(batch_size)),
        )
        cache_stats = {
            "mode": "vision_pooler",
            "unique_images": len(feature_cache),
            "elapsed_sec": float(time.perf_counter() - cache_start),
        }
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("no_trainable_params")
    optimizer = rp.torch.optim.AdamW(
        trainable_params,
        lr=float(learning_rate),
        weight_decay=float(max(0.0, weight_decay)),
    )

    best_score = float("-inf")
    best_aux = float("inf")
    history: List[dict] = []
    temp = max(1e-4, float(temperature))
    rng = random.Random(int(seed))
    start = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = list(range(len(triplet_rows)))
        rng.shuffle(order)
        train_losses: List[float] = []
        for begin in range(0, len(order), max(1, int(batch_size))):
            batch_ids = order[begin : begin + max(1, int(batch_size))]
            if not batch_ids:
                continue
            q_paths, p_paths, n_paths, weights = _sample_batch_rows(triplet_rows, batch_ids=batch_ids, rng=rng)
            optimizer.zero_grad(set_to_none=True)
            if feature_cache:
                q = _cached_projected_features(model, feature_cache, q_paths, device=resolved_device)
                p = _cached_projected_features(model, feature_cache, p_paths, device=resolved_device)
                n = _cached_projected_features(model, feature_cache, n_paths, device=resolved_device)
            else:
                q = _image_features(model, processor, q_paths, device=resolved_device)
                p = _image_features(model, processor, p_paths, device=resolved_device)
                n = _image_features(model, processor, n_paths, device=resolved_device)
            sim_pos = rp.torch.sum(q * p, dim=-1)
            sim_neg = rp.torch.sum(q * n, dim=-1)
            triplet_loss = rp.torch.relu(float(margin) + sim_neg - sim_pos)
            batch_weights = rp.torch.tensor(weights, dtype=q.dtype, device=q.device)
            weighted_triplet = (triplet_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-12)

            if float(max(0.0, ce_weight)) > 0.0:
                logits = rp.torch.matmul(q, rp.torch.cat([p, n], dim=0).transpose(0, 1)) / temp
                targets = rp.torch.arange(len(batch_ids), dtype=rp.torch.long, device=q.device)
                ce = rp.torch.nn.functional.cross_entropy(logits, targets)
                loss = weighted_triplet + (float(ce_weight) * ce)
            else:
                loss = weighted_triplet

            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        eval_stats = _evaluate_rows(
            model=model,
            processor=processor,
            rows=triplet_rows,
            margin=float(margin),
            device=resolved_device,
            feature_cache=feature_cache,
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss_mean": (float(sum(train_losses) / len(train_losses)) if train_losses else None),
                "eval": eval_stats,
            }
        )
        score = float(eval_stats.get("weighted_triplet_satisfied_pct") or 0.0)
        aux = float(eval_stats.get("weighted_hard_triplet_loss") or float("inf"))
        if score > best_score or (score == best_score and aux < best_aux):
            best_score = score
            best_aux = aux
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

    elapsed = float(time.perf_counter() - start)
    return {
        "device": resolved_device,
        "model_id": str(model_id),
        "output_dir": str(output_dir),
        "rows": len(triplet_rows),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "margin": float(margin),
        "temperature": float(temp),
        "ce_weight": float(max(0.0, ce_weight)),
        "elapsed_sec": elapsed,
        "best_weighted_triplet_satisfied_pct": float(best_score if best_score != float("-inf") else 0.0),
        "best_weighted_hard_triplet_loss": (None if best_aux == float("inf") else float(best_aux)),
        "freeze_stats": freeze_stats,
        "feature_cache": cache_stats,
        "history": history,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune retrieval encoder from hard-negative triplets.")
    parser.add_argument("--triplets", required=True, help="JSONL from mine_hard_negative_triplets.")
    parser.add_argument("--query-images-dir", required=True, help="Image root for query paths.")
    parser.add_argument("--reference-images-dir", required=True, help="Image root for positive/negative paths.")
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14", help="Base HF/local model id.")
    parser.add_argument("--output-dir", default="runs/retrieval_encoder_finetune/model", help="save_pretrained() directory.")
    parser.add_argument("--report-output", default="", help="Optional JSON report path.")
    parser.add_argument("--max-triplets", type=int, default=0)
    parser.add_argument("--max-positives-per-row", type=int, default=3)
    parser.add_argument("--max-negatives-per-row", type=int, default=6)
    parser.add_argument(
        "--train-scope",
        default="vision_encoder",
        choices=["visual_projection", "vision_encoder", "all"],
        help="How much of the CLIP encoder to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument(
        "--sample-weight-mode",
        default="triplet_weight",
        choices=["auto", "uniform", "triplet_weight"],
    )
    parser.add_argument("--sample-weight-power", type=float, default=1.0)
    parser.add_argument("--sample-weight-max", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = parser.parse_args(argv)

    triplet_path = Path(args.triplets)
    if not triplet_path.exists():
        raise FileNotFoundError(f"triplets_not_found:{triplet_path}")

    triplets = projection_tools._load_triplets(triplet_path, max_triplets=max(0, int(args.max_triplets)))
    if not triplets:
        raise ValueError("triplets_empty_or_invalid")

    resolved_rows, resolve_stats = _resolve_triplets(
        triplets=triplets,
        query_images_dir=Path(args.query_images_dir),
        reference_images_dir=Path(args.reference_images_dir),
        max_triplets=int(args.max_triplets),
        max_positives_per_row=int(args.max_positives_per_row),
        max_negatives_per_row=int(args.max_negatives_per_row),
        sample_weight_mode=str(args.sample_weight_mode),
        sample_weight_power=float(max(0.0, args.sample_weight_power)),
        sample_weight_max=float(max(0.0, args.sample_weight_max)),
    )
    if not resolved_rows:
        raise ValueError("no_resolved_triplets")

    output_dir = Path(args.output_dir)
    report = train_encoder(
        model_id=str(args.model_id),
        triplet_rows=resolved_rows,
        output_dir=output_dir,
        train_scope=str(args.train_scope),
        epochs=int(max(1, args.epochs)),
        batch_size=int(max(1, args.batch_size)),
        learning_rate=float(max(1e-7, args.learning_rate)),
        weight_decay=float(max(0.0, args.weight_decay)),
        margin=float(max(0.0, args.margin)),
        temperature=float(max(1e-4, args.temperature)),
        ce_weight=float(max(0.0, args.ce_weight)),
        seed=int(args.seed),
        device=str(args.device),
    )

    payload = {
        "triplets_path": str(triplet_path),
        "query_images_dir": str(Path(args.query_images_dir)),
        "reference_images_dir": str(Path(args.reference_images_dir)),
        "resolve_stats": resolve_stats,
        "training": report,
    }
    report_path = Path(args.report_output) if str(args.report_output).strip() else output_dir.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "trained retrieval encoder "
        + f"(rows={len(resolved_rows)}, best_weighted_triplet_satisfied_pct={report.get('best_weighted_triplet_satisfied_pct')})"
    )
    print(f"model -> {output_dir}")
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
