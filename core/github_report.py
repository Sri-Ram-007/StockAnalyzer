"""Deliver the daily report through GitHub itself (no passwords or extra accounts).

When run inside GitHub Actions, it opens an issue in your repository with today's report. GitHub then
sends you a notification and email automatically. Yesterday's report issue is closed to keep things tidy.
It uses the temporary token GitHub gives every workflow run (GITHUB_TOKEN), so nothing needs setting up.
"""
import os
import re

import pandas as pd
import requests

LABEL = "daily-report"


def html_to_markdown(text: str) -> str:
    text = re.sub(r"</?b>", "**", text)
    text = re.sub(r"</?i>", "_", text)
    return re.sub(r"<[^>]+>", "", text)


def post_issue(text: str) -> str:
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPOSITORY")
    if not (token and repo):
        return "GitHub report: skipped (only runs inside GitHub Actions)"
    api = f"https://api.github.com/repos/{repo}"
    h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    # close earlier daily reports (matched by title, so it works even if labels are not allowed)
    old = requests.get(f"{api}/issues", headers=h, params={"state": "open", "per_page": 100}, timeout=20)
    for issue in old.json() if old.ok else []:
        if issue.get("title", "").startswith("📈 Stock Analyzer") and "pull_request" not in issue:
            requests.patch(f"{api}/issues/{issue['number']}", headers=h, json={"state": "closed"}, timeout=20)
    requests.post(f"{api}/labels", headers=h, json={"name": LABEL, "color": "1a7f5a"}, timeout=20)  # ok if it exists
    body = html_to_markdown(text)
    title = f"📈 Stock Analyzer · {pd.Timestamp.now(tz='Asia/Kolkata'):%d %b %Y}"
    r = requests.post(f"{api}/issues", headers=h, json={"title": title, "body": body, "labels": [LABEL]}, timeout=20)
    return f"GitHub report: posted ({r.json().get('html_url')})" if r.ok else f"GitHub report error: {r.text[:200]}"


def write_job_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(html_to_markdown(text) + "\n")
