"""
Environment doctor for reproducible setup and verification.

Examples:
  python -m src.tools.doctor
  python -m src.tools.doctor --json
  python -m src.tools.doctor --rebuild
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[2]
REQS = ROOT / "requirements.txt"
LOCK = ROOT / "requirements.lock.txt"


def _read_requirements(path: Path, visited: Optional[set[Path]] = None) -> list[str]:
    if not path.exists():
        return []
    if visited is None:
        visited = set()
    resolved = path.resolve()
    if resolved in visited:
        return []
    visited.add(resolved)
    packages = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r ") or line.startswith("--requirement "):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                nested_path = (path.parent / parts[1].strip()).resolve()
                packages.extend(_read_requirements(nested_path, visited))
            continue
        if line.startswith("-"):
            continue
        packages.append(line)
    return packages


def _check_python() -> dict:
    version = sys.version_info
    ok = (version.major, version.minor) >= (3, 10)
    return {
        "ok": ok,
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "required_min": "3.10",
    }


def _check_packages(packages: list[str]) -> dict:
    out = {}
    for package in packages:
        name = package.split("==")[0].strip()
        try:
            installed = metadata.version(name)
            pinned = package if "==" in package else None
            matches_pin = pinned is None or installed == pinned.split("==", 1)[1]
            out[name] = {
                "ok": True,
                "installed_version": installed,
                "pinned": pinned,
                "matches_pin": matches_pin,
            }
        except Exception as exc:
            out[name] = {"ok": False, "error": str(exc), "pinned": package if "==" in package else None}
    return out


def _check_paths() -> dict:
    paths = {
        "requirements": REQS,
        "requirements_core": ROOT / "requirements-core.txt",
        "requirements_ml": ROOT / "requirements-ml.txt",
        "requirements_lock": LOCK,
        "defaults_config": ROOT / "src" / "config" / "defaults.json",
        "dashboard_live": ROOT / "src" / "dashboard" / "analysis" / "index.html",
    }
    out = {}
    for key, path in paths.items():
        out[key] = {"path": str(path), "exists": path.exists()}
    return out


def _pip_check() -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        ok = proc.returncode == 0
        return {
            "ok": ok,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def run_checks() -> dict:
    req_pkgs = _read_requirements(REQS)
    lock_pkgs = _read_requirements(LOCK)
    py = _check_python()
    reqs = _check_packages(req_pkgs)
    lock = _check_packages(lock_pkgs) if lock_pkgs else {}
    paths = _check_paths()
    pip_check = _pip_check()

    failures = []
    if not py["ok"]:
        failures.append("python_version")
    if not paths["requirements"]["exists"]:
        failures.append("requirements_missing")
    if not paths["requirements_lock"]["exists"]:
        failures.append("requirements_lock_missing")
    if not pip_check.get("ok", False):
        failures.append("pip_check_failed")
    for name, info in reqs.items():
        if not info.get("ok"):
            failures.append(f"missing_pkg:{name}")
    for key, info in paths.items():
        if not info.get("exists"):
            failures.append(f"missing_path:{key}")

    status = "ok" if not failures else "degraded"
    return {
        "status": status,
        "failures": failures,
        "python": py,
        "requirements": reqs,
        "requirements_lock": lock,
        "paths": paths,
        "pip_check": pip_check,
    }


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def rebuild_and_verify(venv_dir: Path, python_exe: Optional[str] = None) -> int:
    current_prefix = Path(sys.prefix).resolve()
    target_prefix = venv_dir.resolve()
    if current_prefix == target_prefix:
        print("Refusing to rebuild active venv. Run with a non-venv Python interpreter.", file=sys.stderr)
        return 2

    py = python_exe or sys.executable
    if venv_dir.exists():
        shutil.rmtree(venv_dir)

    steps = [
        ([py, "-m", "venv", str(venv_dir)], "create venv"),
        ([str(_venv_python(venv_dir)), "-m", "pip", "install", "--upgrade", "pip"], "upgrade pip"),
        ([str(_venv_python(venv_dir)), "-m", "pip", "install", "-r", str(LOCK)], "install lockfile"),
        ([str(_venv_python(venv_dir)), "-m", "src.tools.doctor", "--json"], "verify"),
    ]
    for cmd, label in steps:
        print(f"[doctor] {label}: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if proc.returncode != 0:
            return proc.returncode
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heimdall environment doctor and rebuild helper.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild .venv from requirements.lock.txt then verify.",
    )
    parser.add_argument(
        "--venv-dir",
        default=str(ROOT / ".venv"),
        help="Virtual environment directory for --rebuild.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter used to create venv during --rebuild (defaults to current).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.rebuild:
        return rebuild_and_verify(Path(args.venv_dir), args.python)

    report = run_checks()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"doctor status: {report['status']}")
        if report["failures"]:
            print("failures:")
            for item in report["failures"]:
                print(f" - {item}")
        else:
            print("all checks passed")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
