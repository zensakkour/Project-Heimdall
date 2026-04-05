"""
Validate local markdown links in README-style files.

Usage:
  python -m src.tools.check_readme_links README.md
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _is_external(target: str) -> bool:
    lowered = target.lower()
    return lowered.startswith("http://") or lowered.startswith("https://") or lowered.startswith("mailto:")


def check_markdown_file(path: Path, root: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    for idx, line in enumerate(text.splitlines(), start=1):
        for match in LINK_RE.finditer(line):
            raw_target = match.group(1).strip()
            if not raw_target or _is_external(raw_target) or raw_target.startswith("#"):
                continue
            target = raw_target.split("#", 1)[0].split("?", 1)[0].strip()
            if not target:
                continue
            candidate = (path.parent / target).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                errors.append(f"{path}:{idx}: link escapes repository root: {raw_target}")
                continue
            if not candidate.exists():
                errors.append(f"{path}:{idx}: missing link target: {raw_target}")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local markdown links.")
    parser.add_argument("files", nargs="+", help="Markdown files to validate.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    all_errors: list[str] = []
    for raw in args.files:
        path = Path(raw)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        if not path.exists():
            all_errors.append(f"{path}: file not found")
            continue
        all_errors.extend(check_markdown_file(path, repo_root))

    if all_errors:
        for err in all_errors:
            print(err)
        return 1
    print("README link check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

