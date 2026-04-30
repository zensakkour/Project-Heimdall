"""Recover and finalize the 40k combined street->aerial IGN dataset."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.tools.split_realistic_dataset import build_realistic_splits


def _read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _image_name_for_street_id(street_id: str) -> str:
    return f"ign_geopf_ortho_{street_id}.png"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _preseed_existing_images(
    *,
    existing_images_dir: Path,
    chunk_meta_dir: Path,
    chunk_out_dir: Path,
) -> dict:
    counts: Dict[str, int] = {}
    total = 0
    for meta_path in sorted(chunk_meta_dir.glob("street_combined_chunk_*.csv")):
        chunk_name = meta_path.stem.replace("street_combined_", "")
        out_images_dir = chunk_out_dir / chunk_name / "aerial" / "images"
        chunk_count = 0
        for row in _read_csv_rows(meta_path):
            street_id = str(row.get("image_id") or "").strip()
            if not street_id:
                continue
            src = existing_images_dir / _image_name_for_street_id(street_id)
            if not src.exists():
                continue
            dst = out_images_dir / src.name
            if dst.exists():
                continue
            _link_or_copy(src, dst)
            chunk_count += 1
        counts[chunk_name] = chunk_count
        total += chunk_count
    return {
        "chunks": counts,
        "total_preseeded": total,
    }


def _chunk_out_dir(chunk_out_dir: Path, meta_path: Path) -> Path:
    return chunk_out_dir / meta_path.stem.replace("street_combined_", "")


def _run_chunk(
    *,
    meta_path: Path,
    chunk_out_dir: Path,
    provider: str,
    crop_size_m: float,
    crop_px: int,
    allow_missing_aerial: bool,
    seed: int,
) -> dict:
    chunk_dir = _chunk_out_dir(chunk_out_dir, meta_path)
    command = [
        sys.executable,
        "-m",
        "src.tools.build_aerial_pairs",
        "--street-metadata",
        str(meta_path),
        "--out",
        str(chunk_dir),
        "--provider",
        str(provider),
        "--crop-size-m",
        str(float(crop_size_m)),
        "--crop-px",
        str(int(crop_px)),
        "--allow-missing-aerial",
        "true" if allow_missing_aerial else "false",
        "--seed",
        str(int(seed)),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(Path.cwd()),
        check=True,
    )
    stdout = completed.stdout.strip()
    return json.loads(stdout) if stdout else {"chunk": chunk_dir.name}


def _run_chunks(
    *,
    chunk_meta_dir: Path,
    chunk_out_dir: Path,
    provider: str,
    crop_size_m: float,
    crop_px: int,
    allow_missing_aerial: bool,
    seed: int,
    max_workers: int,
) -> List[dict]:
    pending_meta = sorted(chunk_meta_dir.glob("street_combined_chunk_*.csv"))
    active: Dict[Future, Path] = {}
    summaries: List[dict] = []
    cursor = 0
    max_workers = max(1, int(max_workers))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while cursor < len(pending_meta) or active:
            while cursor < len(pending_meta) and len(active) < max_workers:
                meta_path = pending_meta[cursor]
                future = executor.submit(
                    _run_chunk,
                    meta_path=meta_path,
                    chunk_out_dir=chunk_out_dir,
                    provider=provider,
                    crop_size_m=crop_size_m,
                    crop_px=crop_px,
                    allow_missing_aerial=allow_missing_aerial,
                    seed=seed,
                )
                active[future] = meta_path
                cursor += 1
            done, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                meta_path = active.pop(future)
                summary = future.result()
                summary["chunk_metadata"] = str(meta_path)
                summaries.append(summary)
                print(json.dumps(summary, indent=2), flush=True, file=sys.stderr)
    summaries.sort(key=lambda item: str(item.get("chunk_metadata") or ""))
    return summaries


def _merge_chunk_outputs(
    *,
    chunk_out_dir: Path,
    final_out_dir: Path,
) -> dict:
    final_images_dir = final_out_dir / "aerial" / "images"
    aerial_rows: List[dict] = []
    pair_rows: List[dict] = []
    seen_aerial_ids: set[str] = set()
    seen_street_ids: set[str] = set()
    linked_images = 0

    for chunk_dir in sorted(path for path in chunk_out_dir.iterdir() if path.is_dir()):
        aerial_meta_path = chunk_dir / "aerial" / "metadata.csv"
        pairs_path = chunk_dir / "pairs.csv"
        if not aerial_meta_path.exists():
            continue
        chunk_aerial_rows = _read_csv_rows(aerial_meta_path)
        chunk_pair_rows = _read_csv_rows(pairs_path) if pairs_path.exists() else []
        pair_by_street = {
            str(row.get("street_id") or "").strip(): row
            for row in chunk_pair_rows
            if str(row.get("street_id") or "").strip()
        }
        for row in chunk_aerial_rows:
            street_id = str(row.get("paired_street_id") or "").strip()
            aerial_id = str(row.get("aerial_id") or "").strip()
            if not street_id or not aerial_id or street_id in seen_street_ids or aerial_id in seen_aerial_ids:
                continue
            rel_path = str(row.get("path") or "").strip()
            if rel_path:
                src = chunk_dir / rel_path
                dst = final_out_dir / rel_path
                if src.exists():
                    _link_or_copy(src, dst)
                    linked_images += 1
            aerial_rows.append(dict(row))
            seen_street_ids.add(street_id)
            seen_aerial_ids.add(aerial_id)
            pair = pair_by_street.get(street_id)
            if pair is not None:
                pair_rows.append(dict(pair))

    _write_csv(
        final_out_dir / "aerial" / "metadata.csv",
        [
            "aerial_id",
            "path",
            "lat",
            "lon",
            "source",
            "provider",
            "resolution_m",
            "crop_size_m",
            "crop_px",
            "license_info",
            "paired_street_id",
            "status",
        ],
        aerial_rows,
    )
    _write_csv(
        final_out_dir / "pairs.csv",
        ["pair_id", "street_id", "street_path", "aerial_id", "aerial_path", "lat", "lon", "heading_deg"],
        pair_rows,
    )
    return {
        "final_out_dir": str(final_out_dir),
        "aerial_rows": len(aerial_rows),
        "pair_rows": len(pair_rows),
        "linked_images": linked_images,
        "final_images_dir": str(final_images_dir),
    }


def recover_combined_dataset(
    *,
    existing_images_dir: Path,
    chunk_meta_dir: Path,
    chunk_out_dir: Path,
    final_out_dir: Path,
    split_out_dir: Path,
    provider: str,
    crop_size_m: float,
    crop_px: int,
    allow_missing_aerial: bool,
    seed: int,
    max_workers: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    cell_size_m: float,
    buffer_cells: int,
    sort_axis: str,
) -> dict:
    preseed = _preseed_existing_images(
        existing_images_dir=existing_images_dir,
        chunk_meta_dir=chunk_meta_dir,
        chunk_out_dir=chunk_out_dir,
    )
    chunk_summaries = _run_chunks(
        chunk_meta_dir=chunk_meta_dir,
        chunk_out_dir=chunk_out_dir,
        provider=provider,
        crop_size_m=crop_size_m,
        crop_px=crop_px,
        allow_missing_aerial=allow_missing_aerial,
        seed=seed,
        max_workers=max_workers,
    )
    merged = _merge_chunk_outputs(
        chunk_out_dir=chunk_out_dir,
        final_out_dir=final_out_dir,
    )
    split_summary = build_realistic_splits(
        pairs_path=final_out_dir / "pairs.csv",
        out_dir=split_out_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        cell_size_m=cell_size_m,
        seed=seed,
        buffer_cells=buffer_cells,
        sort_axis=sort_axis,
    )
    summary = {
        "preseed": preseed,
        "chunks": chunk_summaries,
        "merged": merged,
        "split": split_summary,
    }
    summary_path = final_out_dir / "recovery_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Recover and finalize the 40k combined IGN aerial dataset.")
    parser.add_argument("--existing-images-dir", required=True)
    parser.add_argument("--chunk-meta-dir", required=True)
    parser.add_argument("--chunk-out-dir", required=True)
    parser.add_argument("--final-out-dir", required=True)
    parser.add_argument("--split-out-dir", required=True)
    parser.add_argument("--provider", default="ign_geopf")
    parser.add_argument("--crop-size-m", type=float, default=256.0)
    parser.add_argument("--crop-px", type=int, default=512)
    parser.add_argument("--allow-missing-aerial", default="false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--cell-size-m", type=float, default=300.0)
    parser.add_argument("--buffer-cells", type=int, default=2)
    parser.add_argument("--sort-axis", default="auto")
    args = parser.parse_args(argv)

    allow_missing = str(args.allow_missing_aerial).strip().lower() in {"1", "true", "yes", "y"}
    summary = recover_combined_dataset(
        existing_images_dir=Path(args.existing_images_dir),
        chunk_meta_dir=Path(args.chunk_meta_dir),
        chunk_out_dir=Path(args.chunk_out_dir),
        final_out_dir=Path(args.final_out_dir),
        split_out_dir=Path(args.split_out_dir),
        provider=str(args.provider),
        crop_size_m=float(args.crop_size_m),
        crop_px=int(args.crop_px),
        allow_missing_aerial=allow_missing,
        seed=int(args.seed),
        max_workers=int(args.max_workers),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        cell_size_m=float(args.cell_size_m),
        buffer_cells=int(args.buffer_cells),
        sort_axis=str(args.sort_axis),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
