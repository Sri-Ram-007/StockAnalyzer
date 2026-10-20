"""Sector view: how each sector is doing, built from its member stocks (median of the stocks in it)."""
import numpy as np
import pandas as pd

YEAR = 252


def stock_stats(closes: pd.DataFrame) -> pd.DataFrame:
    c = closes.ffill()
    last = c.iloc[-1]

    def ret(days):
        return last / c.shift(days).iloc[-1] - 1

    hi = c.iloc[-YEAR:].max()
    return pd.DataFrame({
        "price": last, "ret_1d": ret(1), "ret_1m": ret(21), "ret_3m": ret(63), "ret_1y": ret(YEAR),
        "below_high": last / hi - 1, "above_200dma": last > c.rolling(200).mean().iloc[-1],
    }).dropna(subset=["price"])


def sector_table(closes: pd.DataFrame, industries: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per sector (median of its stocks) and the per-stock table it is built from."""
    st = stock_stats(closes)
    st["sector"] = st.index.map(lambda t: industries.get(t) or "Other")
    g = st.groupby("sector")
    tbl = pd.DataFrame({
        "stocks": g.size(),
        "today": g["ret_1d"].median(), "1m": g["ret_1m"].median(), "3m": g["ret_3m"].median(),
        "1y": g["ret_1y"].median(), "below_high": g["below_high"].median(),
        "above_200dma": g["above_200dma"].mean(),
    })
    tbl = tbl[tbl["stocks"] >= 3].sort_values("3m", ascending=False)
    return tbl, st
