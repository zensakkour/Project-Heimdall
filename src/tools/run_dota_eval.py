"""
Run Ultralytics validation on DOTA v1.0 dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_names(data_yaml: str) -> list[str]:
    try:
        for line in Path(data_yaml).read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("names:"):
                _, value = line.split(":", 1)
                value = value.strip()
                if value.startswith("["):
                    return json.loads(value)
    except Exception:
        return []
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate detector on DOTA v1.0.")
    parser.add_argument("--weights", default="yolo11x-obb.pt", help="Model weights")
    parser.add_argument("--data", default="data/dota/dota.yaml", help="Dataset YAML")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--output", default="src/dashboard/data/dota_eval.json", help="Output JSON path")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.val(data=args.data, imgsz=args.imgsz)

    names = _load_names(args.data)
    payload = {"weights": args.weights, "imgsz": args.imgsz, "names": names}
    if hasattr(results, "results_dict"):
        payload.update(results.results_dict)
    else:
        payload["summary"] = str(results)

    # Per-class metrics (best effort)
    per_class = []
    try:
        metrics = results.metrics
        maps = getattr(getattr(metrics, "box", None), "maps", None)
        p = getattr(getattr(metrics, "box", None), "p", None)
        r = getattr(getattr(metrics, "box", None), "r", None)
        map50 = getattr(getattr(metrics, "box", None), "map50", None)
        map = getattr(getattr(metrics, "box", None), "map", None)
        for i, name in enumerate(names):
            per_class.append(
                {
                    "name": name,
                    "map": float(maps[i]) if maps is not None and i < len(maps) else None,
                    "p": float(p[i]) if p is not None and i < len(p) else None,
                    "r": float(r[i]) if r is not None and i < len(r) else None,
                }
            )
        payload["overall"] = {"map": map, "map50": map50}
    except Exception:
        payload["per_class"] = []
    else:
        payload["per_class"] = per_class

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


