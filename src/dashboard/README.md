# Heimdall UI

This is a lightweight local UI for test status and score summaries.

## Generate data

From the repo root:

```powershell
python -m src.tools.run_tests_report
python -m src.tools.generate_ui_data --jsonl outputs.jsonl
```

## Serve the UI

```powershell
cd dashboard
python -m http.server 8000
```

Open: `http://localhost:8000`
