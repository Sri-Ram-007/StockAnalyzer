"""Drawdown and historical recovery analysis.

Question this answers: "This stock is X% below its 1-year high. In the past, when it fell
that much, how often did it get back to that high, and how long did it take?"
"""
import numpy as np
import pandas as pd

YEAR = 252  # trading days


def current_drawdown(prices: pd.Series, window: int = YEAR) -> dict:
    """Where the stock stands now relative to its high over the last `window` trading days."""
    p = prices.dropna()
    if len(p) < 30:
        return {}
    recent = p.iloc[-window:]
    high_date = recent.idxmax()
    high, price = float(recent.max()), float(p.iloc[-1])
    return {
        "price": price,
        "high": high,
        "high_date": high_date,
        "drawdown": price / high - 1,                 # e.g. -0.33 for 1800 -> 1200
        "gain_to_recover": high / price - 1,          # e.g. +0.50 needed to go 1200 -> 1800
        "days_since_high": (p.index[-1] - high_date).days,
    }


def recovery_episodes(prices: pd.Series, threshold: float, window: int = YEAR,
                      horizon: int = 2 * YEAR) -> pd.DataFrame:
    """Every past time the price fell at least `threshold` (e.g. 0.30) below its trailing high.

    For each episode, look forward up to `horizon` trading days and record whether the price
    got back to that old high, how long it took, and how much further it fell first.
    The current, still-open fall is excluded, so the stats only use completed history.
    """
    p = prices.dropna()
    vals = p.to_numpy()
    trailing_high = p.rolling(window, min_periods=window // 2).max().to_numpy()
    dd = vals / trailing_high - 1
    rows, i, n = [], window // 2, len(vals)

    while i < n:
        if not np.isnan(dd[i]) and dd[i] <= -threshold:
            target, start_price = trailing_high[i], vals[i]
            end = min(i + horizon, n - 1)
            future = vals[i + 1:end + 1]
            hit = np.nonzero(future >= target)[0]
            if len(hit):
                j = i + 1 + hit[0]
                lowest = vals[i:j + 1].min()
                rows.append({"start": p.index[i], "price_at_start": start_price, "old_high": target,
                             "recovered": True, "days_to_recover": (p.index[j] - p.index[i]).days,
                             "further_drop": lowest / start_price - 1})
                i = j + 1
                continue
            if i + horizon < n:  # enough future data and still no recovery
                lowest = vals[i:end + 1].min()
                rows.append({"start": p.index[i], "price_at_start": start_price, "old_high": target,
                             "recovered": False, "days_to_recover": np.nan,
                             "further_drop": lowest / start_price - 1})
                # wait until it climbs back above the threshold before counting a new episode
                k = i
                while k < n and (np.isnan(dd[k]) or dd[k] <= -threshold / 2):
                    k += 1
                i = k
                continue
            break  # still ongoing: not enough future data to judge
        i += 1
    return pd.DataFrame(rows)


def recovery_stats(episodes: pd.DataFrame) -> dict:
    if episodes.empty:
        return {"episodes": 0, "recovery_rate": np.nan, "median_days": np.nan, "median_further_drop": np.nan}
    rec = episodes[episodes["recovered"]]
    return {
        "episodes": len(episodes),
        "recovery_rate": len(rec) / len(episodes),
        "median_days": float(rec["days_to_recover"].median()) if len(rec) else np.nan,
        "median_further_drop": float(episodes["further_drop"].median()),
    }


def history_threshold(drawdown: float) -> float:
    """Round the current fall down to a 5% step, so 33% looks at past falls of 30% or more."""
    return max(0.10, np.floor(-drawdown * 20) / 20)


def biggest_one_day_drop(prices: pd.Series, days: int = YEAR) -> float:
    """Largest single-day fall in the period. A fall of 30%+ in one day is usually a demerger,
    split or bad data rather than a real crash, so the drawdown may be misleading."""
    return float(prices.dropna().iloc[-days:].pct_change().min())


def scan(closes: pd.DataFrame, min_dd: float, max_dd: float) -> pd.DataFrame:
    """Screen every stock: current fall from its 1-year high plus its own recovery history."""
    out = []
    for ticker in closes.columns:
        s = closes[ticker].dropna()
        if len(s) < YEAR:
            continue
        now = current_drawdown(s)
        if not now or not (-max_dd <= now["drawdown"] <= -min_dd):
            continue
        thr = history_threshold(now["drawdown"])
        stats = recovery_stats(recovery_episodes(s, thr))
        out.append({"Ticker": ticker, **now, "history_threshold": thr, **stats,
                    "years_of_data": round(len(s) / YEAR, 1),
                    "corporate_action_flag": biggest_one_day_drop(s) <= -0.30})
    return pd.DataFrame(out)


def base_rates(closes: pd.DataFrame, thresholds=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)) -> pd.DataFrame:
    """Market-wide evidence: pooling every stock's history, how often did a fall of at least
    X% from the 1-year high get back to that high within 2 years?"""
    rows = []
    for thr in thresholds:
        eps = [recovery_episodes(closes[t].dropna(), thr) for t in closes.columns if closes[t].notna().sum() >= YEAR]
        eps = pd.concat([e for e in eps if not e.empty], ignore_index=True) if any(not e.empty for e in eps) else pd.DataFrame()
        st = recovery_stats(eps)
        rows.append({"threshold": thr, **st})
    return pd.DataFrame(rows)
