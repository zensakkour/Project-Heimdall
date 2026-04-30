"""Build a simple embedding index for geo retrieval."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from src.core.geo.retrieval_provider import ClipEmbedder


def _load_metadata(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    if path.suffix.lower() in {".jsonl", ".json"}:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError("JSON metadata must be a list")
    raise ValueError("metadata must be .csv, .json, or .jsonl")


def _load_sidecar(image_path: Path) -> Optional[Dict[str, float]]:
    candidates = [
        Path(str(image_path) + ".geo.json"),
        image_path.with_suffix(".geo.json"),
        Path(str(image_path) + ".geoloc.json"),
        image_path.with_suffix(".geoloc.json"),
    ]
    sidecar = next((p for p in candidates if p.exists()), None)
    if sidecar is None:
        return None
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(raw, dict) and "candidates" in raw:
        items = raw.get("candidates")
        if isinstance(items, list) and items:
            raw = items[0]
    if not isinstance(raw, dict):
        return None
    lat = raw.get("latitude")
    lon = raw.get("longitude")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    return {"latitude": float(lat), "longitude": float(lon)}


def _resolve_image_path(root_dir: Path, images_dir: Path, rel_path: str) -> Optional[Path]:
    rel_path = rel_path.replace("\\", "/")
    candidate = root_dir / rel_path
    if candidate.exists():
        return candidate
    candidate = images_dir / rel_path
    if candidate.exists():
        return candidate
    if rel_path.startswith("chips/"):
        trimmed = rel_path.split("chips/", 1)[1]
        candidate = images_dir / trimmed
        if candidate.exists():
            return candidate
    return None


def build_index(
    images: List[Path],
    meta: List[Dict[str, str]],
    output: Path,
    model_id: str,
    root_dir: Path,
    images_dir: Path,
    projection_path: Optional[str] = None,
) -> int:
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    embedder = ClipEmbedder(model_id, device, projection_path=projection_path)
    batch_size = 32 if device == "cuda" else 16

    embeddings = []
    latitudes = []
    longitudes = []
    ids = []
    paths = []
    records = []

    for item in meta:
        rel_path = item.get("path") or item.get("image") or item.get("file")
        if not rel_path:
            continue
        image_path = Path(rel_path)
        if not image_path.is_absolute():
            resolved = _resolve_image_path(root_dir, images_dir, rel_path)
            if resolved is None:
                continue
            image_path = resolved
        if not image_path.exists():
            continue
        lat = item.get("latitude") or item.get("lat")
        lon = item.get("longitude") or item.get("lon")
        if lat is None or lon is None:
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except ValueError:
            continue
        records.append((image_path, lat_f, lon_f, item.get("id") or image_path.stem))

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        batch_images = []
        batch_meta = []
        for image_path, lat_f, lon_f, item_id in batch:
            with Image.open(image_path) as img:
                batch_images.append(img.convert("RGB"))
            batch_meta.append((image_path, lat_f, lon_f, item_id))
        vectors = embedder.embed_many(batch_images)
        for idx, (image_path, lat_f, lon_f, item_id) in enumerate(batch_meta):
            embeddings.append(vectors[idx])
            latitudes.append(lat_f)
            longitudes.append(lon_f)
            ids.append(item_id)
            paths.append(str(image_path))

    if not embeddings:
        return 0

    arr = np.vstack(embeddings).astype(np.float32)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        embeddings=arr,
        latitudes=np.array(latitudes, dtype=np.float32),
        longitudes=np.array(longitudes, dtype=np.float32),
        ids=np.array(ids, dtype=object),
        paths=np.array(paths, dtype=object),
        model_id=np.asarray(model_id, dtype=np.str_),
    )
    return len(embeddings)


def build_index_from_sidecars(
    images: List[Path],
    output: Path,
    model_id: str,
    projection_path: Optional[str] = None,
) -> int:
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    embedder = ClipEmbedder(model_id, device, projection_path=projection_path)
    batch_size = 32 if device == "cuda" else 16

    embeddings = []
    latitudes = []
    longitudes = []
    ids = []
    paths = []
    records = []

    for image_path in images:
        meta = _load_sidecar(image_path)
        if meta is None:
            continue
        records.append((image_path, meta["latitude"], meta["longitude"], image_path.stem))

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        batch_images = []
        batch_meta = []
        for image_path, lat_f, lon_f, item_id in batch:
            with Image.open(image_path) as img:
                batch_images.append(img.convert("RGB"))
            batch_meta.append((image_path, lat_f, lon_f, item_id))
        vectors = embedder.embed_many(batch_images)
        for idx, (image_path, lat_f, lon_f, item_id) in enumerate(batch_meta):
            embeddings.append(vectors[idx])
            latitudes.append(lat_f)
            longitudes.append(lon_f)
            ids.append(item_id)
            paths.append(str(image_path))

    if not embeddings:
        return 0

    arr = np.vstack(embeddings).astype(np.float32)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        embeddings=arr,
        latitudes=np.array(latitudes, dtype=np.float32),
        longitudes=np.array(longitudes, dtype=np.float32),
        ids=np.array(ids, dtype=object),
        paths=np.array(paths, dtype=object),
        model_id=np.asarray(model_id, dtype=np.str_),
    )
    return len(embeddings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a geo retrieval embedding index.")
    parser.add_argument("--images-dir", default="data/university-1652/images/train", help="Image folder")
    parser.add_argument("--metadata", help="CSV/JSON/JSONL with path, latitude, longitude")
    parser.add_argument(
        "--output",
        default="data/geo_index/university1652_clip.npz",
        help="Output .npz index path",
    )
    parser.add_argument(
        "--model-id",
        default="openai/clip-vit-large-patch14",
        help="Embedding model id",
    )
    parser.add_argument(
        "--projection-path",
        default="",
        help="Optional .npz projection (matrix+bias) applied to embeddings before indexing.",
    )
    args = parser.parse_args()

    image_dir = Path(args.images_dir)
    images = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        print("No images found.")
        return 1

    output = Path(args.output)
    projection_path = str(args.projection_path).strip() or None
    if args.metadata:
        meta = _load_metadata(Path(args.metadata))
        count = build_index(
            images,
            meta,
            output,
            args.model_id,
            image_dir.parent,
            image_dir,
            projection_path=projection_path,
        )
    else:
        count = build_index_from_sidecars(images, output, args.model_id, projection_path=projection_path)
    if count == 0:
        print("No embeddings written (missing metadata or sidecar).")
        return 1
    print(f"wrote {count} embeddings to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
