@echo off
REM Uploads your latest holdings, watchlist and alert rules to GitHub so the daily email uses them.
cd /d "%~dp0"
git add data/portfolio.json data/watchlist.json data/alert_rules.json
git commit -m "Update holdings, watchlist and alert rules" || echo Nothing new to upload.
git push
pause
