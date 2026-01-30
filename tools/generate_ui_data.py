"""
Generate UI summary JSON from outputs.jsonl and test report.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate UI summary JSON.")
    parser.add_argument("--jsonl", default="outputs.jsonl", help="Path to batch JSONL output")
    parser.add_argument(
        "--test-report", default="dashboard/data/test_report.json", help="Path to test report JSON"
    )
    parser.add_argument("--output", default="dashboard/data/summary.json", help="Output JSON path")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.jsonl))
    scores = []
    for row in rows:
        result = row.get("result", {})
        geo = result.get("geo") or {}
        scores.append(
            {
                "image": row.get("image", "-"),
                "score": result.get("score"),
                "geo_tier": geo.get("confidence_tier"),
                "geo_conf": geo.get("confidence"),
                "uncertainty_m": geo.get("uncertainty_m"),
            }
        )

    avg = 0.0
    if scores:
        valid = [s["score"] for s in scores if isinstance(s.get("score"), (int, float))]
        if valid:
            avg = sum(valid) / len(valid)

    high_tier = sum(1 for s in scores if s.get("geo_tier") == "high")

    test_report_path = Path(args.test_report)
    tests = None
    if test_report_path.exists():
        try:
            tests = json.loads(test_report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tests = None

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "avg_score": avg,
        "high_tier_count": high_tier,
        "scores": scores[:200],
        "tests": tests,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
