"""
Unzip DOTA v1.0 and generate a dataset YAML for Ultralytics.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def find_split_dirs(root: Path) -> dict[str, Path] | None:
    # Look for structure: <root>/**/(images|imgs)/train and labels/ train
    images = list(root.rglob("*images/train")) + list(root.rglob("*imgs/train"))
    val_images = list(root.rglob("*images/val")) + list(root.rglob("*images/valid")) + list(
        root.rglob("*imgs/val")
    )
    labels = list(root.rglob("*labels/train"))
    val_labels = list(root.rglob("*labels/val")) + list(root.rglob("*labels/valid"))

    if images and val_images and labels and val_labels:
        return {
            "train": images[0],
            "val": val_images[0],
            "train_labels": labels[0],
            "val_labels": val_labels[0],
        }
    return None


def write_yaml(output: Path, data_root: Path, splits: dict[str, Path] | None) -> None:
    # Minimal YAML for Ultralytics; assumes labels in parallel to images.
    names = [
        "plane",
        "ship",
        "storage-tank",
        "baseball-diamond",
        "tennis-court",
        "basketball-court",
        "ground-track-field",
        "harbor",
        "bridge",
        "large-vehicle",
        "small-vehicle",
        "helicopter",
        "roundabout",
        "soccer-ball-field",
        "swimming-pool",
    ]
    if splits is None:
        content = {
            "path": str(data_root),
            "train": "train/images",
            "val": "val/images",
            "names": names,
        }
    else:
        content = {
            "path": str(data_root),
            "train": str(splits["train"].relative_to(data_root)),
            "val": str(splits["val"].relative_to(data_root)),
            "names": names,
        }
    output.write_text(_to_yaml(content), encoding="utf-8")


def _to_yaml(data: dict) -> str:
    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}: {json.dumps(value)}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare DOTA v1.0 dataset.")
    parser.add_argument("--zip", default="data/dota/DOTAv1.zip", help="Path to DOTAv1 zip")
    parser.add_argument("--out", default="data/dota/DOTAv1", help="Extraction directory")
    parser.add_argument("--yaml", default="data/dota/dota.yaml", help="Output dataset YAML path")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Missing zip: {zip_path}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)

    splits = find_split_dirs(out)
    yaml_path = Path(args.yaml)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(yaml_path, out, splits)
    print(f"Prepared dataset at {out}")
    print(f"Wrote YAML to {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


