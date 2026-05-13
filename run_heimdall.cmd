@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)

echo Starting Project Heimdall...
echo Using %PYTHON_EXE%
echo.

"%PYTHON_EXE%" -m src.tools.dev_app --open-browser --no-reload %*

endlocal
