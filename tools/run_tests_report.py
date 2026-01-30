"""
Run pytest and emit a lightweight JSON report for the UI.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


SUMMARY_RE = re.compile(r"(?P<count>\\d+) (?P<label>passed|failed|skipped|xfailed|xpassed|errors?)")


def parse_summary(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in SUMMARY_RE.finditer(text):
        label = match.group("label").replace("errors", "failed")
        counts[label] = counts.get(label, 0) + int(match.group("count"))
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pytest and save a JSON report.")
    parser.add_argument("--output", default="dashboard/data/test_report.json", help="Output JSON path")
    args = parser.parse_args()

    result = subprocess.run(
        ["python", "-m", "pytest", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )

    output = (result.stdout or "") + "\n" + (result.stderr or "")
    counts = parse_summary(output)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "return_code": result.returncode,
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "xfailed": counts.get("xfailed", 0),
        "xpassed": counts.get("xpassed", 0),
        "raw_output": output.strip(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
