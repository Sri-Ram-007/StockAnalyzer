@echo off
REM Starts the Stock Analyzer so your phone can open it too (phone must be on the same Wi-Fi).
REM Open Settings in the app to see a QR code. Windows may ask to allow Python on "Private networks": click Allow.
REM Use this only on your home Wi-Fi, not on public Wi-Fi.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless false --browser.gatherUsageStats false
