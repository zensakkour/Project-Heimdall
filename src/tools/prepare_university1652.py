from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


def export_split(split: str, output_dir: Path, limit: int) -> int:
    ds = load_dataset("layumi/university-1652", split=split)
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
        image.save(path, format="JPEG", quality=95)
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
    args = parser.parse_args()

    out = Path(args.output) / args.split
    count = export_split(args.split, out, args.limit)
    print(f"exported {count} images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
