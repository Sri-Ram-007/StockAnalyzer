"""Send today's portfolio summary and alerts.

  python run_alerts.py            build the message and send it (Telegram and/or email)
  python run_alerts.py --dry-run  only print the message, send nothing
"""
import sys

from core.alerts import build_message, send

if __name__ == "__main__":
    text = build_message()
    print(text)
    if "--dry-run" not in sys.argv:
        for result in send(text):
            print(result)
