"""
Temperature scaling utility for retrieval scores.
"""
from __future__ import annotations

import argparse
import json

from src.core.logic.fusion import _softmax


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate fusion retrieval temperature.")
    parser.add_argument("scores", help="JSON list of retrieval scores")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    raw_scores = json.loads(args.scores)
    if not isinstance(raw_scores, list):
        raise SystemExit("scores must be a JSON list")
    logits = [float(s) / max(args.temperature, 1e-6) for s in raw_scores]
    weights = _softmax(logits)
    print(json.dumps({"temperature": args.temperature, "weights": weights}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


