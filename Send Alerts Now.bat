@echo off
REM Sends today's portfolio summary and alerts once (Telegram / email). Used by Windows Task Scheduler too.
cd /d "%~dp0"
".venv\Scripts\python.exe" run_alerts.py
