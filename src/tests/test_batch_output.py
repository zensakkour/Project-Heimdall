"""
Basic JSONL output validation for batch_run.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.batch_run import assessment_to_dict
from src.core.logic.types import Assessment


def test_batch_output_jsonl_roundtrip() -> None:
    assessment = Assessment(detections=[], geo=None, verification=None, score=0.0)
    payload = {"image": "example.jpg", "result": assessment_to_dict(assessment)}

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        data = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(data) == 1
        parsed = json.loads(data[0])
        assert parsed["image"] == "example.jpg"
        assert parsed["result"]["score"] == 0.0



