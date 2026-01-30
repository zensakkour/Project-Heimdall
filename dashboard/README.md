# Heimdall UI

This is a lightweight local UI for test status and score summaries.

## Generate data

From the repo root:

```powershell
python tools/run_tests_report.py
python tools/generate_ui_data.py --jsonl outputs.jsonl
```

## Serve the UI

```powershell
cd dashboard
python -m http.server 8000
```

Open: `http://localhost:8000`
