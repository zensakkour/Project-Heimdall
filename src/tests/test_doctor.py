from __future__ import annotations

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

