"""Send today's portfolio summary and alerts.

  python run_alerts.py            build the message and deliver it
  python run_alerts.py --dry-run  only print the message, send nothing

Delivery: inside GitHub Actions the report is posted as an issue in your repository (GitHub notifies you,
no passwords needed). Telegram / email are also used if you ever set them up.
"""
import sys

from core.alerts import build_message, load_config, send
from core.github_report import post_issue, write_job_summary

if __name__ == "__main__":
    text = build_message()
    print(text)
    if "--dry-run" not in sys.argv:
        write_job_summary(text)
        print(post_issue(text))
        if any(load_config().get(k) for k in ("TELEGRAM_BOT_TOKEN", "EMAIL_ADDRESS")):
            for result in send(text):
                print(result)
