from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from PIL import Image


def export_split(split: str, output_dir: Path, limit: int, source_dir: Path) -> int:
    ds = load_dataset("layumi/university-1652", split=split, streaming=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, item in enumerate(ds):
        if limit and count >= limit:
            break
        image = item.get("image")
        if image is None:
            continue
        name = f"{split}_{idx:06d}.jpg"
        path = output_dir / name
        if isinstance(image, str):
            image_path = source_dir / image
            if not image_path.exists():
                continue
            with Image.open(image_path) as img:
                img.convert("RGB").save(path, format="JPEG", quality=95)
        else:
            image.convert("RGB").save(path, format="JPEG", quality=95)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Export University-1652 images to a local folder.")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--limit", type=int, default=200, help="Max images to export (0 = all)")
    parser.add_argument(
        "--output",
        default="data/university-1652/images",
        help="Output directory for exported images",
    )
    parser.add_argument(
        "--source-dir",
        default="data/University-1652",
        help="Local University-1652 folder that contains train/ and test/ image trees",
    )
    args = parser.parse_args()

    out = Path(args.output) / args.split
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print("Source dataset not found. The HuggingFace mirror does NOT include images.")
        print("Download the full University-1652 dataset from the official link and pass --source-dir.")
        return 1
    count = export_split(args.split, out, args.limit, source_dir)
    print(f"exported {count} images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
