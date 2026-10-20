"""IPOs: open, upcoming and past Indian IPOs (NSE), US IPOs (Nasdaq), and how listed IPOs performed.

Sources (free, no keys):
  - NSE public APIs: current issues (with subscription), upcoming issues, all past issues
  - Nasdaq IPO calendar: priced / upcoming / filed US IPOs
  - Yahoo Finance prices for after-listing performance
"""
import json
import re
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from core.data import DATA_DIR

IPO_DIR = DATA_DIR / "ipo"
IPO_DIR.mkdir(exist_ok=True)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
EQUITY_TYPES = {"EQ": "Mainboard", "BE": "Mainboard", "SME": "SME"}


def _age_hours(path):
    return (time.time() - path.stat().st_mtime) / 3600 if path.exists() else float("inf")


def _nse_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*",
                      "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/"})
    s.get("https://www.nseindia.com", timeout=20)  # sets the cookies NSE's API requires
    return s


def _cached_json(name, max_hours, fetch):
    cache = IPO_DIR / f"{name}.json"
    if _age_hours(cache) > max_hours:
        try:
            data = fetch()
            if data is not None:
                cache.write_text(json.dumps(data))
        except Exception:
            pass
    return json.loads(cache.read_text()) if cache.exists() else None


def _price_band(text):
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(text))]
    return (min(nums), max(nums)) if nums else (np.nan, np.nan)


# ---------------- India ----------------
def india_open_and_upcoming() -> pd.DataFrame:
    """IPOs open now, just closed, or forthcoming on NSE, with total subscription where available."""
    def fetch():
        s = _nse_session()
        cur = s.get("https://www.nseindia.com/api/ipo-current-issue", timeout=20).json()
        up = s.get("https://www.nseindia.com/api/all-upcoming-issues?category=ipo", timeout=20).json()
        return {"current": cur, "upcoming": up}
    d = _cached_json("india_current", 1, fetch)
    if not d:
        return pd.DataFrame()
    subs = {r["symbol"]: r.get("noOfTime") for r in d["current"] if r.get("category", "Total") == "Total"}
    rows = {}
    for r in d["upcoming"] + d["current"]:
        lo, hi = _price_band(r.get("issuePrice"))
        rows[r["symbol"]] = {
            "Company": r["companyName"], "Symbol": r["symbol"],
            "Type": "SME" if r.get("series") == "SME" else "Mainboard",
            "Status": r.get("status", ""), "Opens": pd.to_datetime(r.get("issueStartDate"), format="%d-%b-%Y", errors="coerce"),
            "Closes": pd.to_datetime(r.get("issueEndDate"), format="%d-%b-%Y", errors="coerce"),
            "Price band": r.get("issuePrice", ""), "Upper price": hi,
            "Subscribed (times)": pd.to_numeric(subs.get(r["symbol"]), errors="coerce"),
        }
    df = pd.DataFrame(rows.values())
    order = {"Open": 0, "Active": 0, "Forthcoming": 1, "Closed": 2}
    return df.sort_values(["Status", "Opens"], key=lambda c: c.map(order).fillna(3) if c.name == "Status" else c,
                          ascending=[True, False]).reset_index(drop=True)


def india_past_ipos(since_year: int = 2016) -> pd.DataFrame:
    """Every listed equity IPO on NSE since `since_year` (mainboard and SME), with issue price and listing date."""
    def fetch():
        return _nse_session().get("https://www.nseindia.com/api/public-past-issues", timeout=30).json()
    d = _cached_json("india_past", 24, fetch)
    if not d:
        return pd.DataFrame()
    df = pd.DataFrame(d)
    df = df[df["securityType"].isin(EQUITY_TYPES)].copy()
    df["Type"] = df["securityType"].map(EQUITY_TYPES)
    df["issue_price"] = pd.to_numeric(df["issuePrice"], errors="coerce")
    df["listing_date"] = pd.to_datetime(df["listingDate"].str.title(), format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_price", "listing_date"])
    df = df[df["listing_date"].dt.year >= since_year]
    df["Ticker"] = df["symbol"].str.strip() + ".NS"
    return df.rename(columns={"company": "Company"})[["Company", "Ticker", "Type", "issue_price", "listing_date"]] \
             .drop_duplicates("Ticker").sort_values("listing_date", ascending=False).reset_index(drop=True)


def listing_prices(tickers: list[str], key: str) -> pd.DataFrame:
    """Daily Open and Close for many recently listed stocks (cached 12 h). Returns a two-level column frame."""
    cache = IPO_DIR / f"prices_{key}.parquet"
    if _age_hours(cache) > 12:
        import logging
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # renamed/delisted tickers are expected
        frames = []
        for i in range(0, len(tickers), 150):
            raw = yf.download(tickers[i:i + 150], period="max", auto_adjust=False, progress=False, threads=True,
                              group_by="column")
            if not raw.empty and isinstance(raw.columns, pd.MultiIndex):
                frames.append(raw[["Open", "Close"]])
        if frames:
            px = pd.concat(frames, axis=1)
            px.index = pd.to_datetime(px.index).tz_localize(None)
            px.to_parquet(cache)
    return pd.read_parquet(cache) if cache.exists() else pd.DataFrame()


def performance(ipos: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Listing-day gain and returns after listing, all measured from the issue price."""
    if px.empty:
        return pd.DataFrame()
    out = []
    closes, opens = px["Close"], px["Open"]
    for _, r in ipos.iterrows():
        t = r["Ticker"]
        if t not in closes:
            continue
        c = closes[t].dropna()
        c = c[c.index >= r["listing_date"] - pd.Timedelta(days=3)]
        if len(c) < 1:
            continue
        o = opens[t].dropna().reindex(c.index)
        ip = r["issue_price"]

        def after(days):
            target = c.index[0] + pd.Timedelta(days=days)
            return c[c.index >= target].iloc[0] / ip - 1 if c.index[-1] >= target else np.nan

        out.append({**r.to_dict(),
                    "listing_open_gain": (o.iloc[0] / ip - 1) if pd.notna(o.iloc[0]) else np.nan,
                    "listing_close_gain": c.iloc[0] / ip - 1,
                    "ret_1m": after(30), "ret_6m": after(182), "ret_1y": after(365),
                    "ret_now": c.iloc[-1] / ip - 1, "price_now": c.iloc[-1],
                    "below_high": c.iloc[-1] / c.max() - 1, "days_listed": (c.index[-1] - c.index[0]).days})
    return pd.DataFrame(out)


def summary_stats(perf: pd.DataFrame) -> pd.DataFrame:
    """Evidence table by IPO type: how often IPOs are above their issue price over time."""
    rows = []
    for typ, g in [("All", perf)] + list(perf.groupby("Type")):
        rows.append({"Type": typ, "IPOs": len(g),
                     "Listed above issue price": (g["listing_close_gain"] > 0).mean(),
                     "Median listing-day gain": g["listing_close_gain"].median(),
                     "Above issue price after 1 year": (g["ret_1y"].dropna() > 0).mean() if g["ret_1y"].notna().any() else np.nan,
                     "Median return after 1 year": g["ret_1y"].median(),
                     "Above issue price today": (g["ret_now"] > 0).mean()})
    return pd.DataFrame(rows)


def pop_vs_later(perf: pd.DataFrame) -> pd.DataFrame:
    """Do big listing-day pops last? Median 1-year return grouped by listing-day gain."""
    p = perf.dropna(subset=["listing_close_gain", "ret_1y"]).copy()
    if p.empty:
        return pd.DataFrame()
    bins = [-1, 0, 0.2, 0.5, 1.0, 100]
    labels = ["Listed below issue", "0–20% pop", "20–50% pop", "50–100% pop", "Over 100% pop"]
    p["Listing day"] = pd.cut(p["listing_close_gain"], bins=bins, labels=labels)
    g = p.groupby("Listing day", observed=True)
    res = g.agg(IPOs=("Ticker", "size"), median_listing=("listing_close_gain", "median"), median_1y=("ret_1y", "median"))
    res["gave_back"] = g.apply(lambda d: ((1 + d["ret_1y"]) < (1 + d["listing_close_gain"])).mean(), include_groups=False)
    return res.reset_index()


# ---------------- US ----------------
def us_ipos(months_back: int = 3) -> dict:
    """Nasdaq IPO calendar: upcoming, filed and recently priced US IPOs."""
    def fetch():
        out = {"priced": [], "upcoming": [], "filed": []}
        now = pd.Timestamp.now()
        for k in range(-months_back, 2):
            m = (now + pd.DateOffset(months=k)).strftime("%Y-%m")
            d = requests.get(f"https://api.nasdaq.com/api/ipo/calendar?date={m}",
                             headers={"User-Agent": UA, "Accept": "application/json"}, timeout=20).json().get("data") or {}
            out["priced"] += (d.get("priced") or {}).get("rows") or []
            out["filed"] += (d.get("filed") or {}).get("rows") or []
            up = d.get("upcoming") or {}
            out["upcoming"] += ((up.get("upcomingTable") or {}).get("rows") or []) if isinstance(up, dict) else []
        return out
    d = _cached_json("us_calendar", 6, fetch) or {"priced": [], "upcoming": [], "filed": []}
    return {k: pd.DataFrame(v).drop_duplicates() if v else pd.DataFrame() for k, v in d.items()}
