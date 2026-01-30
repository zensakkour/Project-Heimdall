"""
Validate JSONL outputs against the batch schema.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_schema(schema_path: Path) -> dict:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate JSONL output against schema.")
    parser.add_argument("jsonl", help="Path to JSONL file")
    parser.add_argument(
        "--schema",
        default="schemas/batch_result.schema.json",
        help="Path to JSON schema",
    )
    args = parser.parse_args()

    try:
        import jsonschema  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "jsonschema is required. Install it with: pip install jsonschema"
        ) from exc

    jsonl_path = Path(args.jsonl)
    schema_path = Path(args.schema)

    if not jsonl_path.exists():
        raise SystemExit(f"JSONL file not found: {jsonl_path}")
    if not schema_path.exists():
        raise SystemExit(f"Schema file not found: {schema_path}")

    schema = load_schema(schema_path)

    errors = 0
    for idx, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            jsonschema.validate(instance=payload, schema=schema)
        except jsonschema.ValidationError as err:
            errors += 1
            print(f"Line {idx}: {err.message}")

    if errors:
        print(f"Validation failed: {errors} error(s)")
        return 1

    print("Validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
