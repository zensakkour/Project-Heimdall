from __future__ import annotations

import shutil
import uuid

from src.tools import doctor


def test_doctor_run_checks_shape() -> None:
    report = doctor.run_checks()
    assert report["status"] in {"ok", "degraded"}
    assert "python" in report
    assert "requirements" in report
    assert "paths" in report
    assert "pip_check" in report
    assert "failures" in report


def test_doctor_requirements_file_exists_check() -> None:
    report = doctor.run_checks()
    assert report["paths"]["requirements"]["exists"] is True
    assert report["paths"]["requirements_lock"]["exists"] is True


def test_read_requirements_supports_nested_includes() -> None:
    base = doctor.ROOT / "runs" / f"doctor-test-{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        core = base / "requirements-core.txt"
        ml = base / "requirements-ml.txt"
        top = base / "requirements.txt"
        core.write_text("fastapi\nuvicorn\n", encoding="utf-8")
        ml.write_text("# comment\nultralytics\n", encoding="utf-8")
        top.write_text("-r requirements-core.txt\n--requirement requirements-ml.txt\n", encoding="utf-8")
        packages = doctor._read_requirements(top)
        assert packages == ["fastapi", "uvicorn", "ultralytics"]
    finally:
        shutil.rmtree(base, ignore_errors=True)
