"""Build a realistic aerial retrieval index with explicit dataset-root path resolution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from src.tools.build_geo_index import _load_metadata, build_index


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build an aerial retrieval index for a realistic dataset root.")
    parser.add_argument("--root", required=True, help="Dataset root, e.g. data/paris_realistic_v1_combined")
    parser.add_argument("--metadata", default="aerial/metadata.csv", help="Metadata path relative to root.")
    parser.add_argument("--images-dir", default="aerial/images", help="Image folder path relative to root.")
    parser.add_argument("--output", default="indices/aerial_clip_index.npz", help="Output path relative to root.")
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--projection-path", default="")
    args = parser.parse_args(argv)

    root = Path(args.root)
    metadata_path = root / str(args.metadata)
    images_dir = root / str(args.images_dir)
    output_path = root / str(args.output)
    projection_path = str(args.projection_path).strip() or None

    metadata = _load_metadata(metadata_path)
    images = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    count = build_index(
        images,
        metadata,
        output_path,
        str(args.model_id),
        root,
        images_dir,
        projection_path=projection_path,
    )
    print(json.dumps({"count": count, "output": str(output_path)}, indent=2))
    return 0 if count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
