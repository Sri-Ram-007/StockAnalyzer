"""Phase 2: business health check ("is this a value trap?") and the recovery score.

Data: Yahoo Finance company info and annual income statements via yfinance (free, no key).
Cached on disk for 7 days per company.
"""
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf

from core.data import DATA_DIR

FUND_DIR = DATA_DIR / "fundamentals"
FUND_DIR.mkdir(exist_ok=True)
FUND_MAX_AGE_DAYS = 7
FINANCIAL_SECTORS = {"Financial Services"}  # banks/NBFCs borrow by design, so debt checks don't apply

INFO_KEYS = ["longName", "sector", "industry", "marketCap", "trailingPE", "priceToBook", "debtToEquity",
             "returnOnEquity", "profitMargins", "revenueGrowth", "earningsGrowth",
             "earningsQuarterlyGrowth", "trailingEps", "dividendYield", "currency", "financialCurrency"]


def _clean(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v


def get_fundamentals(ticker: str, prices: pd.Series | None = None) -> dict:
    """Key ratios plus up to 5 years of revenue, profit and EPS, and the P/E history."""
    cache = FUND_DIR / f"{ticker.replace('&', '_')}.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < FUND_MAX_AGE_DAYS * 86400:
        return json.loads(cache.read_text())

    out = {"ticker": ticker, "ok": False}
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        out.update({k: _clean(info.get(k)) for k in INFO_KEYS})
        inc = tk.income_stmt
        years = []
        if inc is not None and not inc.empty:
            for col in sorted(inc.columns):
                row = inc[col]
                rev, ni = row.get("Total Revenue"), row.get("Net Income")
                eps = row.get("Diluted EPS", row.get("Basic EPS"))
                if pd.notna(ni) or pd.notna(rev):
                    years.append({"year_end": str(pd.Timestamp(col).date()), "revenue": _clean(rev),
                                  "net_income": _clean(ni), "eps": _clean(eps)})
        out["years"] = years
        # P/E at each financial-year end = price then / that year's EPS
        # Some companies (e.g. Infosys) report accounts in US dollars while the share trades in rupees,
        # so P/E history is only computed when both use the same currency.
        same_ccy = (out.get("financialCurrency") or out.get("currency")) == out.get("currency")
        if prices is not None and len(prices) and same_ccy:
            pes = []
            for y in years:
                if y["eps"] and y["eps"] > 0:
                    p = prices[prices.index <= pd.Timestamp(y["year_end"])]
                    if len(p):
                        pes.append(float(p.iloc[-1]) / y["eps"])
            out["pe_history"] = pes
        out["ok"] = bool(info) or bool(years)
    except Exception as e:
        out["error"] = str(e)[:200]
    if out["ok"]:
        cache.write_text(json.dumps(out))
    return out


def fetch_many(tickers, prices_df: pd.DataFrame | None = None, progress=None) -> dict:
    """Fetch fundamentals for several companies in parallel (cached ones return instantly)."""
    results = {}

    def one(t):
        s = prices_df[t].dropna() if prices_df is not None and t in prices_df else None
        return t, get_fundamentals(t, s)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (t, f) in enumerate(pool.map(one, tickers), 1):
            results[t] = f
            if progress:
                progress(i / len(tickers))
    return results


# ---------------- Health checks ----------------
PASS, WARN, FAIL, NA = "pass", "warn", "fail", "na"


def _growth(years, key):
    vals = [y[key] for y in years if y.get(key) is not None]
    if len(vals) < 2 or vals[-2] == 0:
        return None
    return (vals[-1] - vals[-2]) / abs(vals[-2])


def health_checks(f: dict, drawdown: float | None = None, peer_drawdown: float | None = None,
                  market_drawdown: float | None = None) -> list[dict]:
    """Plain-English checks. Each returns status pass / warn / fail / na and a short explanation."""
    checks = []
    years = f.get("years", [])
    sector = f.get("sector")

    # 1. Profitable?
    eps, last_ni = f.get("trailingEps"), (years[-1]["net_income"] if years else None)
    if eps is not None or last_ni is not None:
        profitable = (eps if eps is not None else last_ni) > 0
        checks.append({"check": "Making a profit", "status": PASS if profitable else FAIL,
                       "detail": "The company is profitable." if profitable else "The company is currently making a loss."})
    else:
        checks.append({"check": "Making a profit", "status": NA, "detail": "Profit data not available."})

    # 2. Profit trend (last financial year vs the one before)
    g = _growth(years, "net_income")
    if g is not None:
        st = PASS if g >= 0 else WARN if g > -0.25 else FAIL
        checks.append({"check": "Profit trend", "status": st,
                       "detail": f"Yearly profit {'grew' if g >= 0 else 'fell'} {abs(g):.0%} in the last financial year."})
    else:
        checks.append({"check": "Profit trend", "status": NA, "detail": "Not enough yearly profit history."})

    # 3. Sales trend (latest quarter vs a year ago)
    rg = f.get("revenueGrowth")
    if rg is not None:
        st = PASS if rg >= 0 else WARN if rg > -0.10 else FAIL
        checks.append({"check": "Sales trend", "status": st,
                       "detail": f"Sales {'grew' if rg >= 0 else 'fell'} {abs(rg):.0%} compared with a year ago."})
    else:
        checks.append({"check": "Sales trend", "status": NA, "detail": "Sales growth not available."})

    # 4. Debt
    de = f.get("debtToEquity")
    if sector in FINANCIAL_SECTORS:
        checks.append({"check": "Debt level", "status": NA,
                       "detail": "Not used for banks and finance companies: borrowing is their business."})
    elif de is not None:
        ratio = de / 100
        st = PASS if ratio < 1 else WARN if ratio < 2 else FAIL
        checks.append({"check": "Debt level", "status": st,
                       "detail": f"Debt is {ratio:.1f}× shareholders' equity ({'low' if st == PASS else 'moderate' if st == WARN else 'high'})."})
    else:
        checks.append({"check": "Debt level", "status": NA, "detail": "Debt data not available."})

    # 5. Profit margin
    pm = f.get("profitMargins")
    if pm is not None:
        st = PASS if pm >= 0.08 else WARN if pm >= 0 else FAIL
        checks.append({"check": "Profit margin", "status": st,
                       "detail": f"Keeps {pm:.0%} of sales as profit."})
    else:
        checks.append({"check": "Profit margin", "status": NA, "detail": "Margin data not available."})

    # 6. Valuation compared with its own history
    pe, hist = f.get("trailingPE"), [p for p in f.get("pe_history", []) if 0 < p < 500]
    if pe and pe > 0 and len(hist) >= 2:
        med = float(np.median(hist))
        rel = pe / med - 1
        st = PASS if rel <= 0 else WARN if rel <= 0.25 else FAIL
        word = "cheaper" if rel <= 0 else "more expensive"
        checks.append({"check": "Price vs its usual valuation", "status": st,
                       "detail": f"P/E is {pe:.1f} vs its usual {med:.1f}: {abs(rel):.0%} {word} than normal."})
    else:
        checks.append({"check": "Price vs its usual valuation", "status": NA,
                       "detail": "Not enough earnings history to compare (or the company has losses)."})

    # 7. Was the fall company-specific?
    if drawdown is not None and (peer_drawdown is not None or market_drawdown is not None):
        ref = peer_drawdown if peer_drawdown is not None else market_drawdown
        who = "its industry" if peer_drawdown is not None else "the market"
        gap = drawdown - ref  # negative = fell more than peers
        st = PASS if gap > -0.10 else WARN if gap > -0.25 else FAIL
        detail = (f"Fell {abs(drawdown):.0%} vs {abs(ref):.0%} for {who}: "
                  + ("a broad fall, not just this company." if st == PASS else "it fell much more than " + who + ", so the reason may be company-specific."))
        checks.append({"check": "Fall vs its industry", "status": st, "detail": detail})
    return checks


STATUS_VALUE = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}


def health_score(checks: list[dict], exclude=("Price vs its usual valuation", "Fall vs its industry")) -> float | None:
    vals = [STATUS_VALUE[c["status"]] for c in checks if c["status"] in STATUS_VALUE and c["check"] not in exclude]
    return float(np.mean(vals)) if vals else None


# ---------------- Recovery score ----------------
WEIGHT_LEVELS = {"Off": 0, "Low": 1, "Medium": 2, "High": 3}
DEFAULT_WEIGHTS = {"Past recoveries": "High", "Business health": "High",
                   "Valuation": "Medium", "Broad vs company-specific fall": "Medium"}


def recovery_score(stock_rate, episodes, base_rate, checks, weights=None) -> dict:
    """Combine the evidence into one 0-100 score. Each part is 0..1 before weighting.

    Past recoveries: this stock's recovery rate, pulled towards the all-stocks rate when it has
    few past episodes (so 1 lucky recovery doesn't count as 100%).
    """
    weights = weights or DEFAULT_WEIGHTS
    parts = {}
    if base_rate is not None and not pd.isna(base_rate):
        n = episodes or 0
        r = stock_rate if (stock_rate is not None and not pd.isna(stock_rate)) else base_rate
        parts["Past recoveries"] = (n * r + 3 * base_rate) / (n + 3)
    hs = health_score(checks)
    if hs is not None:
        parts["Business health"] = hs
    by = {c["check"]: c["status"] for c in checks}
    if by.get("Price vs its usual valuation") in STATUS_VALUE:
        parts["Valuation"] = STATUS_VALUE[by["Price vs its usual valuation"]]
    if by.get("Fall vs its industry") in STATUS_VALUE:
        parts["Broad vs company-specific fall"] = STATUS_VALUE[by["Fall vs its industry"]]

    total_w = sum(WEIGHT_LEVELS[weights.get(k, "Medium")] for k in parts)
    if not parts or total_w == 0:
        return {"score": None, "parts": parts}
    score = sum(v * WEIGHT_LEVELS[weights.get(k, "Medium")] for k, v in parts.items()) / total_w
    return {"score": round(score * 100), "parts": parts}


def score_label(score):
    if score is None:
        return "Not enough data"
    return "Strong" if score >= 70 else "Mixed" if score >= 45 else "Weak"
