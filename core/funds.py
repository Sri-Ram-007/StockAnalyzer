"""Phase 3: Indian mutual funds.

Sources (free, no keys):
  - AMFI NAVAll.txt: every scheme with its category and latest NAV (official, updated daily)
  - mfapi.in: full NAV history per scheme
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from core.data import DATA_DIR

MF_DIR = DATA_DIR / "funds"
MF_DIR.mkdir(exist_ok=True)
AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
MFAPI_URL = "https://api.mfapi.in/mf/{}"
HEADERS = {"User-Agent": "Mozilla/5.0"}
YEAR = 252


def _age_hours(path):
    return (time.time() - path.stat().st_mtime) / 3600 if path.exists() else float("inf")


def get_fund_list() -> pd.DataFrame:
    """All open-ended schemes from AMFI with fund house, category, plan and option. Refreshed daily."""
    cache = MF_DIR / "amfi_list.csv"
    if _age_hours(cache) > 20:
        try:
            r = requests.get(AMFI_URL, headers=HEADERS, timeout=40)
            r.raise_for_status()
            r.encoding = "utf-8"
            rows, category, house = [], None, None
            for line in r.text.splitlines():
                line = line.strip()
                if not line or line.startswith("Scheme Code"):
                    continue
                if line.count(";") >= 5:
                    p = line.split(";")
                    rows.append({"code": int(p[0]), "name": p[3].strip(), "plan": p[4].strip(), "option": p[5].strip(),
                                 "nav": pd.to_numeric(p[6], errors="coerce"), "nav_date": p[7].strip(),
                                 "category": category, "fund_house": house})
                elif "Schemes(" in line:
                    m = re.match(r"(.*?)\((.*)\)", line)
                    category = m.group(2).strip() if m else line
                    scheme_type = m.group(1).strip() if m else ""
                    category = category if scheme_type.startswith("Open Ended") else None
                else:
                    house = line
            df = pd.DataFrame(rows)
            df = df[df["category"].notna()]
            df.to_csv(cache, index=False)
        except Exception:
            pass
    if not cache.exists():
        raise RuntimeError("Could not download the AMFI fund list and no cached copy exists.")
    df = pd.read_csv(cache)
    df["is_direct"] = df["plan"].str.contains("Direct", case=False, na=False) | df["name"].str.contains("Direct", case=False, na=False)
    df["is_growth"] = df["option"].str.contains("Growth", case=False, na=False) | df["name"].str.contains("Growth", case=False, na=False)
    df["display"] = df["name"] + " · " + df["plan"].fillna("") + " · " + df["option"].fillna("")
    return df


def get_nav(code: int) -> pd.Series:
    """Full daily NAV history for one scheme (oldest first), cached for 12 hours."""
    cache = MF_DIR / f"nav_{code}.parquet"
    if _age_hours(cache) > 12:
        try:
            j = requests.get(MFAPI_URL.format(code), headers=HEADERS, timeout=30).json()
            d = pd.DataFrame(j.get("data", []))
            if not d.empty:
                s = pd.Series(pd.to_numeric(d["nav"], errors="coerce").to_numpy(),
                              index=pd.to_datetime(d["date"], format="%d-%m-%Y"), name="nav")
                s = s[s > 0].sort_index()
                s = s[~s.index.duplicated()]
                s.to_frame().to_parquet(cache)
        except Exception:
            pass
    if not cache.exists():
        return pd.Series(dtype=float)
    return pd.read_parquet(cache)["nav"]


def _cagr(s: pd.Series, years: float):
    if s.empty or (s.index[-1] - s.index[0]).days < years * 365 - 10:
        return None
    start = s[s.index <= s.index[-1] - pd.DateOffset(years=years)]
    if start.empty:
        return None
    return (s.iloc[-1] / start.iloc[-1]) ** (1 / years) - 1


def crash_recoveries(s: pd.Series, threshold: float = 0.10) -> pd.DataFrame:
    """Every fall of at least `threshold` from an all-time high: how deep, and how long to get back."""
    peak_val, peak_date, rows = s.iloc[0], s.index[0], []
    in_fall, trough_val, trough_date = False, None, None
    for d, v in s.items():
        if v >= peak_val:
            if in_fall:
                rows.append({"peak": peak_date, "trough": trough_date, "fall": trough_val / peak_val - 1,
                             "recovered": d, "months_to_recover": (d - trough_date).days / 30.4,
                             "months_peak_to_recovery": (d - peak_date).days / 30.4})
                in_fall = False
            peak_val, peak_date = v, d
        else:
            if not in_fall and v / peak_val - 1 <= -threshold:
                in_fall, trough_val, trough_date = True, v, d
            if in_fall and v < trough_val:
                trough_val, trough_date = v, d
    if in_fall:
        rows.append({"peak": peak_date, "trough": trough_date, "fall": trough_val / peak_val - 1,
                     "recovered": pd.NaT, "months_to_recover": np.nan, "months_peak_to_recovery": np.nan})
    return pd.DataFrame(rows)


def fund_metrics(s: pd.Series) -> dict:
    """Returns, risk and consistency for one fund."""
    s = s.dropna()
    if len(s) < 30:
        return {}
    daily = s.pct_change().dropna()
    peak = s.cummax()
    roll3 = (s / s.shift(3 * YEAR)) ** (1 / 3) - 1 if len(s) > 3 * YEAR else pd.Series(dtype=float)
    roll3 = roll3.dropna()
    vol = float(daily.std() * np.sqrt(YEAR))
    c3 = _cagr(s, 3)
    return {
        "nav": float(s.iloc[-1]), "nav_date": s.index[-1], "start": s.index[0],
        "1Y": _cagr(s, 1), "3Y": c3, "5Y": _cagr(s, 5), "10Y": _cagr(s, 10),
        "since_start": (s.iloc[-1] / s.iloc[0]) ** (365 / max((s.index[-1] - s.index[0]).days, 1)) - 1,
        "volatility": vol,
        "max_drawdown": float((s / peak - 1).min()),
        "current_drawdown": float(s.iloc[-1] / peak.iloc[-1] - 1),
        "return_per_risk": (c3 / vol) if (c3 is not None and vol > 0) else None,
        "rolling3_min": float(roll3.min()) if len(roll3) else None,
        "rolling3_median": float(roll3.median()) if len(roll3) else None,
        "rolling3_pct_above_10": float((roll3 > 0.10).mean()) if len(roll3) else None,
    }


def category_table(fund_list: pd.DataFrame, category: str, direct=True, growth=True, progress=None) -> pd.DataFrame:
    """Metrics for every fund in a category (Direct + Growth by default), fetched in parallel."""
    f = fund_list[fund_list["category"] == category]
    if direct:
        f = f[f["is_direct"]]
    if growth:
        f = f[f["is_growth"]]
    f = f.drop_duplicates("name")
    codes = f["code"].tolist()
    out = []

    def one(code):
        return code, fund_metrics(get_nav(code))

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (code, m) in enumerate(pool.map(one, codes), 1):
            if m:
                out.append({"code": code, **m})
            if progress:
                progress(i / max(len(codes), 1))
    if not out:
        return pd.DataFrame()
    return f[["code", "name", "fund_house"]].merge(pd.DataFrame(out), on="code")


# ---------------- SIP calculator ----------------
def xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float | None:
    """Yearly return for irregular cash flows (money in negative, value out positive), by bisection."""
    if len(cashflows) < 2:
        return None
    t0 = cashflows[0][0]

    def npv(rate):
        return sum(cf / (1 + rate) ** ((d - t0).days / 365.0) for d, cf in cashflows)

    lo, hi = -0.99, 10.0
    if npv(lo) * npv(hi) > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def sip_simulation(nav: pd.Series, monthly: float, start: pd.Timestamp) -> dict:
    """Invest `monthly` rupees on the first NAV date of every month from `start`, using real past NAVs."""
    s = nav[nav.index >= start].dropna()
    if len(s) < 2:
        return {}
    buy_dates = s.groupby([s.index.year, s.index.month]).head(1).index
    units = monthly / s.loc[buy_dates]
    cum_units = units.cumsum().reindex(s.index, method="ffill").fillna(0)
    cum_invested = pd.Series(monthly, index=buy_dates).cumsum().reindex(s.index, method="ffill").fillna(0)
    value = cum_units * s
    flows = [(d, -monthly) for d in buy_dates] + [(s.index[-1], float(value.iloc[-1]))]
    lump_units = monthly * len(buy_dates) / s.iloc[0]
    return {
        "installments": len(buy_dates), "invested": float(cum_invested.iloc[-1]), "value": float(value.iloc[-1]),
        "units": float(units.sum()), "xirr": xirr(flows),
        "lump_value": float(lump_units * s.iloc[-1]),  # same total put in on day one instead
        "history": pd.DataFrame({"Invested": cum_invested, "Value": value}),
    }
