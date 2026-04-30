"""Merge multiple realistic street-image datasets into one combined metadata root."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


FIELDNAMES = [
    "image_id",
    "path",
    "lat",
    "lon",
    "heading_deg",
    "captured_at",
    "camera_type",
    "width",
    "height",
    "quality_score",
    "sequence",
    "source",
    "license_info",
]


def _load_metadata(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def merge_street_datasets(
    *,
    metadata_paths: Sequence[Path],
    out_dir: Path,
) -> dict:
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    merged: List[dict] = []
    seen_keys: set[str] = set()

    for metadata_path in metadata_paths:
        src_root = metadata_path.parent
        rows = _load_metadata(metadata_path)
        for row in rows:
            image_id = str(row.get("image_id") or "").strip()
            rel_path = str(row.get("path") or "").strip()
            source = str(row.get("source") or "unknown").strip() or "unknown"
            if not image_id or not rel_path:
                continue
            key = f"{source}::{image_id}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            src_path = src_root / rel_path
            suffix = src_path.suffix or ".jpg"
            dest_name = f"{source.replace(':', '_')}__{image_id}{suffix}"
            dest_rel = f"images/{dest_name}"
            shutil.copy2(src_path, out_dir / dest_rel)

            merged.append(
                {
                    "image_id": image_id,
                    "path": dest_rel,
                    "lat": row.get("lat", ""),
                    "lon": row.get("lon", ""),
                    "heading_deg": row.get("heading_deg", ""),
                    "captured_at": row.get("captured_at", ""),
                    "camera_type": row.get("camera_type", ""),
                    "width": row.get("width", ""),
                    "height": row.get("height", ""),
                    "quality_score": row.get("quality_score", ""),
                    "sequence": row.get("sequence", ""),
                    "source": source,
                    "license_info": row.get("license_info", ""),
                }
            )

    merged.sort(key=lambda item: (str(item["source"]), str(item["image_id"])))
    metadata_out = out_dir / "metadata.csv"
    with metadata_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in merged:
            writer.writerow(row)

    return {
        "metadata_inputs": [str(path) for path in metadata_paths],
        "output_metadata": str(metadata_out),
        "merged_count": len(merged),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge multiple realistic street-image datasets.")
    parser.add_argument("--metadata", nargs="+", required=True, help="One or more metadata.csv paths to merge.")
    parser.add_argument("--out", required=True, help="Combined output directory.")
    args = parser.parse_args(argv)

    summary = merge_street_datasets(
        metadata_paths=[Path(item) for item in args.metadata],
        out_dir=Path(args.out),
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
