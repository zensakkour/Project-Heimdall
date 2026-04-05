from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from src.tools.check_readme_links import check_markdown_file


def _make_repo_root() -> Path:
    root = Path.cwd() / "runs" / f"link-test-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_check_markdown_file_reports_missing_target() -> None:
    repo = _make_repo_root()
    md = repo / "README.md"
    try:
        md.write_text("[bad](missing.md)\n", encoding="utf-8")
        errors = check_markdown_file(md, repo)
        assert len(errors) == 1
        assert "missing link target" in errors[0]
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_check_markdown_file_accepts_existing_target() -> None:
    repo = _make_repo_root()
    docs = repo / "docs"
    try:
        docs.mkdir(parents=True)
        target = docs / "x.md"
        target.write_text("# ok\n", encoding="utf-8")
        md = repo / "README.md"
        md.write_text("[good](docs/x.md)\n", encoding="utf-8")
        errors = check_markdown_file(md, repo)
        assert errors == []
    finally:
        shutil.rmtree(repo, ignore_errors=True)
