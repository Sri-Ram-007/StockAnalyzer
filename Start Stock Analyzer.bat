@echo off
REM Double-click to open the Stock Analyzer in your browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First-time setup: creating the Python environment...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
