"""
Chip SpaceNet PS-RGB GeoTIFFs into JPG tiles with lat/lon metadata.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform


def to_lonlat(ds: rasterio.DatasetReader, row: float, col: float) -> tuple[float, float]:
    x, y = ds.transform * (col, row)
    if ds.crs and ds.crs.to_string() != "EPSG:4326":
        lon, lat = transform(ds.crs, "EPSG:4326", [x], [y])
        return float(lat[0]), float(lon[0])
    return float(y), float(x)


def scale_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    max_val = float(np.max(arr)) if arr.size else 1.0
    if max_val <= 255.0:
        return np.clip(arr, 0, 255).astype(np.uint8)
    # Simple linear downscale for 16-bit imagery.
    arr = arr / (max_val / 255.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def chip_tiff(
    tif_path: Path,
    out_dir: Path,
    writer: csv.DictWriter,
    chip_size: int,
    stride: int,
    max_chips: int,
) -> int:
    count = 0
    with rasterio.open(tif_path) as ds:
        width = ds.width
        height = ds.height
        for top in range(0, height - chip_size + 1, stride):
            for left in range(0, width - chip_size + 1, stride):
                if max_chips and count >= max_chips:
                    return count
                window = Window(left, top, chip_size, chip_size)
                data = ds.read(indexes=[1, 2, 3], window=window)
                if data.size == 0:
                    continue
                data = np.transpose(data, (1, 2, 0))
                data = scale_to_uint8(data)
                center_row = top + chip_size / 2
                center_col = left + chip_size / 2
                lat, lon = to_lonlat(ds, center_row, center_col)

                name = f"{tif_path.stem}_r{top}_c{left}.jpg"
                out_path = out_dir / name
                Image.fromarray(data).save(out_path, quality=92)
                writer.writerow(
                    {
                        "path": name,
                        "latitude": f"{lat:.8f}",
                        "longitude": f"{lon:.8f}",
                    }
                )
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Chip SpaceNet PS-RGB GeoTIFFs to JPG + metadata.")
    parser.add_argument("--input-dir", default="data/spacenet_paris/PS-RGB", help="Folder with .TIF files")
    parser.add_argument("--output-dir", default="data/spacenet_paris/chips", help="Output JPG folder")
    parser.add_argument("--metadata", default="data/spacenet_paris/metadata.csv", help="Output metadata CSV")
    parser.add_argument("--chip-size", type=int, default=512, help="Chip size in pixels")
    parser.add_argument("--stride", type=int, default=512, help="Stride in pixels")
    parser.add_argument("--max-chips-per-tiff", type=int, default=200, help="Limit chips per input file (0 = no limit)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.metadata)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    tifs = sorted(input_dir.rglob("*.TIF"))
    if not tifs:
        print(f"No .TIF files found under {input_dir}")
        return 1

    total = 0
    with meta_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "latitude", "longitude"])
        writer.writeheader()
        for tif in tifs:
            count = chip_tiff(
                tif,
                out_dir,
                writer,
                args.chip_size,
                args.stride,
                args.max_chips_per_tiff,
            )
            total += count
            print(f"{tif.name}: {count} chips")
    print(f"Wrote {total} chips to {out_dir}")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
