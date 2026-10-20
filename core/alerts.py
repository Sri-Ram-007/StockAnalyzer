"""Phase 5: daily alerts on Telegram and/or email.

Checks your portfolio and watchlist once a day and reports only *new* events, by comparing today with
the previous trading day (so the same fall is not reported every day):
  - crossed below X% from its 1-year high
  - moved more than Y% in one day
  - a holding's total loss crossed Z%
  - a price target you set was crossed (above or below)

Credentials are read from environment variables (used by GitHub Actions secrets) or, on your own
computer, from data/alerts_config.json. They are never stored anywhere else.
"""
import json
import os
import smtplib
from email.mime.text import MIMEText

import pandas as pd
import requests

from core.data import DATA_DIR, get_ohlc, load_watchlist
from core.funds import get_nav
from core.portfolio import load_holdings, summary, value_portfolio

RULES_FILE = DATA_DIR / "alert_rules.json"
CONFIG_FILE = DATA_DIR / "alerts_config.json"
DEFAULT_RULES = {"fall_from_high": [20, 30, 40], "day_move": 5, "holding_loss": 15, "targets": [],
                 "daily_summary": True, "include_ipos": True}
YEAR = 252


def load_rules() -> dict:
    try:
        return {**DEFAULT_RULES, **json.loads(RULES_FILE.read_text())}
    except Exception:
        return dict(DEFAULT_RULES)


def save_rules(rules: dict) -> None:
    RULES_FILE.write_text(json.dumps(rules, indent=1))


def load_config() -> dict:
    """Telegram / email settings: environment variables first, then the local config file."""
    try:
        cfg = json.loads(CONFIG_FILE.read_text())
    except Exception:
        cfg = {}
    for key in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "EMAIL_ADDRESS", "EMAIL_APP_PASSWORD", "EMAIL_TO"]:
        if os.environ.get(key):
            cfg[key] = os.environ[key]
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps({k: v for k, v in cfg.items() if v}, indent=1))


def _series(kind: str, ident: str) -> pd.Series:
    if kind == "fund":
        return get_nav(int(ident))
    o = get_ohlc(ident)
    return o["Close"].dropna() if not o.empty else pd.Series(dtype=float)


def _fresh(s: pd.Series, max_age_days: int = 5) -> bool:
    return len(s) > YEAR // 2 and (pd.Timestamp.now().normalize() - s.index[-1].normalize()).days <= max_age_days


def evaluate(rules: dict | None = None) -> list[str]:
    """Return a list of alert lines for events that happened on the latest trading day."""
    rules = rules or load_rules()
    holdings = load_holdings()
    items = {(h["kind"], h["ident"]): h["name"] for h in holdings}
    for t in load_watchlist():
        items.setdefault(("stock", t), t)
    alerts = []

    for (kind, ident), name in items.items():
        s = _series(kind, ident)
        if not _fresh(s):
            continue
        today, yday = float(s.iloc[-1]), float(s.iloc[-2])
        hi_today, hi_yday = float(s.iloc[-YEAR:].max()), float(s.iloc[-YEAR - 1:-1].max())
        dd_today, dd_yday = today / hi_today - 1, yday / hi_yday - 1
        unit = "NAV" if kind == "fund" else "price"

        for thr in sorted(rules.get("fall_from_high", []), reverse=True):
            if dd_today <= -thr / 100 < dd_yday:
                alerts.append(f"📉 <b>{name}</b> is now {abs(dd_today):.0%} below its 1-year high "
                              f"({unit} {today:,.2f} vs high {hi_today:,.2f}).")
                break
        move = today / yday - 1
        if rules.get("day_move") and abs(move) >= rules["day_move"] / 100:
            alerts.append(f"{'🚀' if move > 0 else '⚠️'} <b>{name}</b> moved {move:+.1%} today ({unit} {today:,.2f}).")
        for tg in rules.get("targets", []):
            if tg.get("ident") != ident:
                continue
            if tg.get("above") and yday < tg["above"] <= today:
                alerts.append(f"🎯 <b>{name}</b> rose above your target {tg['above']:,.2f} (now {today:,.2f}).")
            if tg.get("below") and yday > tg["below"] >= today:
                alerts.append(f"🎯 <b>{name}</b> fell below your level {tg['below']:,.2f} (now {today:,.2f}).")

    loss = rules.get("holding_loss")
    if loss:
        for h in holdings:
            s = _series(h["kind"], h["ident"])
            if not _fresh(s):
                continue
            p_today, p_yday = float(s.iloc[-1]) / h["buy_price"] - 1, float(s.iloc[-2]) / h["buy_price"] - 1
            if p_today <= -loss / 100 < p_yday:
                alerts.append(f"🔻 Your holding <b>{h['name']}</b> is now down {abs(p_today):.0%} from your buy price.")
    return alerts


def ipo_lines() -> list[str]:
    """Indian IPOs open now, opening within a week, just closed, and listed in the last week."""
    from core.ipo import india_open_and_upcoming, india_past_ipos
    today = pd.Timestamp.now().normalize()
    try:
        df = india_open_and_upcoming()
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        return ["", "<b>🆕 IPOs</b>", "IPO data was not available today (NSE did not respond)."]
    band = lambda r: str(r["Price band"]).replace("Rs.", "₹")
    subs = lambda r: "" if pd.isna(r["Subscribed (times)"]) else f", {r['Subscribed (times)']:.1f}× subscribed"
    open_now = df[(df["Opens"] <= today) & (df["Closes"] >= today)]
    soon = df[(df["Opens"] > today) & (df["Opens"] <= today + pd.Timedelta(days=7))]
    closed = df[(df["Closes"] < today) & (df["Closes"] >= today - pd.Timedelta(days=4))]
    out = []
    out += [f"• Open now: <b>{r['Company']}</b> ({r['Type']}), closes {r['Closes']:%d %b}, {band(r)}{subs(r)}" for _, r in open_now.iterrows()]
    out += [f"• Opens {r['Opens']:%d %b}: <b>{r['Company']}</b> ({r['Type']}), {band(r)}" for _, r in soon.iterrows()]
    out += [f"• Just closed: <b>{r['Company']}</b>{subs(r)}" for _, r in closed.iterrows()]
    try:
        past = india_past_ipos()
        recent = past[past["listing_date"] >= today - pd.Timedelta(days=7)]
        for _, r in recent.iterrows():
            o = get_ohlc(r["Ticker"])
            if len(o):
                gain = float(o["Close"].iloc[0]) / r["issue_price"] - 1
                now = float(o["Close"].iloc[-1]) / r["issue_price"] - 1
                out.append(f"• Listed {r['listing_date']:%d %b}: <b>{r['Company']}</b> ({r['Type']}) "
                           f"{gain:+.0%} on listing day, now {now:+.0%} vs issue price ₹{r['issue_price']:,.0f}")
    except Exception:
        pass
    return ["", "<b>🆕 IPOs</b>"] + (out or ["No IPOs opening, closing or listing this week."])


def inr(x: float, signed: bool = False) -> str:
    x = 0.0 if abs(x) < 0.5 else x
    sign = "-" if x < 0 else ("+" if signed and x > 0 else "")
    return f"{sign}₹{abs(x):,.0f}"


def build_message(rules: dict | None = None) -> str:
    rules = rules or load_rules()
    lines = [f"<b>📈 Stock Analyzer · {pd.Timestamp.now():%d %b %Y}</b>"]
    holdings = load_holdings()
    if rules.get("daily_summary") and holdings:
        df = value_portfolio(holdings)
        sm = summary(df)
        if sm:
            lines += ["", "<b>Your portfolio</b>",
                      f"Value: {inr(sm['value'])}  ({inr(sm['day_pnl'], True)} today, {sm['day_pct']:+.2%})",
                      f"Total profit/loss: {inr(sm['pnl'], True)} ({sm['pnl_pct']:+.1%})"]
            movers = df.dropna(subset=["day_pct"]).sort_values("day_pct")
            if len(movers):
                best, worst = movers.iloc[-1], movers.iloc[0]
                lines.append(f"Best today: {best['name']} {best['day_pct']:+.2%} · Worst: {worst['name']} {worst['day_pct']:+.2%}")
    alerts = evaluate(rules)
    lines += ["", "<b>🔔 Alerts</b>"] + [f"• {a}" for a in alerts] if alerts else ["", "No new alerts today."]
    if rules.get("include_ipos", True):
        lines += ipo_lines()
    try:  # tell the user if the monthly insurance-data update could not run
        st = json.loads((DATA_DIR / "insurance" / "update_status.json").read_text())
        if st.get("result", "").startswith("failed"):
            lines += ["", f"⚠️ Insurance data update: {st['result']}"]
    except Exception:
        pass
    lines += ["", "<i>Educational tool, not investment advice.</i>"]
    return "\n".join(lines)


def send_telegram(text: str, cfg: dict) -> str:
    token, chat = cfg.get("TELEGRAM_BOT_TOKEN"), cfg.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return "Telegram not set up"
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True},
                      timeout=20)
    return "Telegram: sent" if r.ok else f"Telegram error: {r.text[:150]}"


def send_email(text: str, cfg: dict) -> str:
    sender, pwd = cfg.get("EMAIL_ADDRESS"), cfg.get("EMAIL_APP_PASSWORD")
    to = cfg.get("EMAIL_TO") or sender
    if not (sender and pwd):
        return "Email not set up"
    html = text.replace("\n", "<br>")
    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"], msg["From"], msg["To"] = f"Stock Analyzer · {pd.Timestamp.now():%d %b %Y}", sender, to
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
            smtp.login(sender, pwd)
            smtp.send_message(msg)
        return "Email: sent"
    except Exception as e:
        return f"Email error: {str(e)[:150]}"


def send(text: str) -> list[str]:
    cfg = load_config()
    return [send_telegram(text, cfg), send_email(text, cfg)]
