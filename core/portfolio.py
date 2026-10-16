"""Phase 5: portfolio tracker.

Holdings are saved in data/portfolio.json on your computer. Each holding is a stock/ETF (Yahoo ticker)
or an Indian mutual fund (AMFI scheme code). Values are shown in rupees; foreign holdings are converted
at the latest exchange rate from Yahoo Finance.
"""
import json
import uuid

import pandas as pd
import yfinance as yf

from core.data import DATA_DIR, get_ohlc
from core.funds import get_nav

PORTFOLIO_FILE = DATA_DIR / "portfolio.json"


def load_holdings() -> list[dict]:
    try:
        return json.loads(PORTFOLIO_FILE.read_text())
    except Exception:
        return []


def save_holdings(holdings: list[dict]) -> None:
    PORTFOLIO_FILE.write_text(json.dumps(holdings, indent=1))


def add_holding(kind: str, ident: str, name: str, qty: float, buy_price: float, buy_date: str, currency: str = "INR") -> None:
    h = load_holdings()
    h.append({"id": uuid.uuid4().hex[:8], "kind": kind, "ident": str(ident), "name": name, "qty": float(qty),
              "buy_price": float(buy_price), "buy_date": buy_date, "currency": currency})
    save_holdings(h)


def remove_holding(hid: str) -> None:
    save_holdings([h for h in load_holdings() if h["id"] != hid])


def fx_to_inr(currency: str) -> float:
    """Rupees per 1 unit of `currency` (1.0 for INR). London pence (GBp) are converted to pounds first."""
    if not currency or currency == "INR":
        return 1.0
    factor = 0.01 if currency == "GBp" else 1.0
    cur = "GBP" if currency == "GBp" else currency
    try:
        closes = yf.Ticker(f"{cur}INR=X").history(period="10d")["Close"].dropna()
        return float(closes.iloc[-1]) * factor if len(closes) else float("nan")
    except Exception:
        return float("nan")


def fx_on(currency: str, date: str) -> float:
    """Rupees per unit of `currency` on a past date (the buy date), so profit includes currency moves."""
    if not currency or currency == "INR":
        return 1.0
    factor = 0.01 if currency == "GBp" else 1.0
    cur = "GBP" if currency == "GBp" else currency
    try:
        d = pd.Timestamp(date)
        h = yf.Ticker(f"{cur}INR=X").history(start=d - pd.Timedelta(days=10), end=d + pd.Timedelta(days=1))["Close"].dropna()
        return float(h.iloc[-1]) * factor if len(h) else fx_to_inr(currency)
    except Exception:
        return fx_to_inr(currency)


def price_series(h: dict) -> pd.Series:
    """Daily close (stocks/ETFs) or NAV (mutual funds) for one holding."""
    if h["kind"] == "fund":
        return get_nav(int(h["ident"]))
    o = get_ohlc(h["ident"])
    return o["Close"].dropna() if not o.empty else pd.Series(dtype=float)


def value_portfolio(holdings: list[dict]) -> pd.DataFrame:
    """One row per holding with invested amount, current value, profit/loss and today's change, all in rupees."""
    rows, fx_cache = [], {}
    for h in holdings:
        s = price_series(h)
        cur = h.get("currency", "INR")
        if cur not in fx_cache:
            fx_cache[cur] = fx_to_inr(cur)
        fx = fx_cache[cur]
        if len(s) < 2:
            price = prev = float("nan")
            asof = None
        else:
            price, prev, asof = float(s.iloc[-1]), float(s.iloc[-2]), s.index[-1]
        buy_fx = fx if cur == "INR" else fx_on(cur, h.get("buy_date") or pd.Timestamp.now().date().isoformat())
        invested = h["qty"] * h["buy_price"] * buy_fx
        value = h["qty"] * price * fx
        high_1y = float(s.iloc[-252:].max()) if len(s) else float("nan")
        rows.append({
            "id": h["id"], "kind": h["kind"], "ident": h["ident"], "name": h["name"], "currency": cur,
            "qty": h["qty"], "buy_price": h["buy_price"], "price": price, "asof": asof,
            "invested_inr": invested, "value_inr": value,
            "pnl_inr": value - invested, "pnl_pct": value / invested - 1 if invested else float("nan"),
            "day_pnl_inr": h["qty"] * (price - prev) * fx, "day_pct": price / prev - 1 if prev else float("nan"),
            "below_1y_high": price / high_1y - 1 if high_1y else float("nan"),
            "buy_date": h.get("buy_date"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        total = df["value_inr"].sum()
        df["weight"] = df["value_inr"] / total if total else float("nan")
    return df


def summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    invested, value, day = df["invested_inr"].sum(), df["value_inr"].sum(), df["day_pnl_inr"].sum()
    return {"invested": invested, "value": value, "pnl": value - invested,
            "pnl_pct": value / invested - 1 if invested else float("nan"),
            "day_pnl": day, "day_pct": day / (value - day) if value - day else float("nan")}
