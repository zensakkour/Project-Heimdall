"""
Temperature scaling utility for retrieval scores.
"""
from __future__ import annotations

import argparse
import json

from src.core.logic.fusion import _softmax, _temperature_scaled_logprob, _to_unit_interval


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate fusion retrieval temperature.")
    parser.add_argument("scores", help="JSON list of retrieval scores")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    raw_scores = json.loads(args.scores)
    if not isinstance(raw_scores, list):
        raise SystemExit("scores must be a JSON list")
    norm_scores = [_to_unit_interval(float(s)) for s in raw_scores]
    logits = [_temperature_scaled_logprob(score, args.temperature) for score in norm_scores]
    weights = _softmax(logits)
    print(
        json.dumps(
            {
                "temperature": args.temperature,
                "input_scores": raw_scores,
                "normalized_scores": norm_scores,
                "weights": weights,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


