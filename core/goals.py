"""Goal planner: how much to invest each month to reach a goal, using real past returns of Indian
mutual fund categories instead of an assumed flat return."""
import numpy as np
import pandas as pd

from core.funds import category_table

# Each choice combines AMFI's two naming styles for the same category.
ASSET_CHOICES = {
    "Large cap equity funds": ["Equity Scheme - Large Cap Fund", "Equity Schemes - Large Cap Fund"],
    "Flexi cap equity funds": ["Equity Scheme - Flexi Cap Fund", "Equity Schemes - Flexi Cap Fund"],
    "Mid cap equity funds": ["Equity Scheme - Mid Cap Fund", "Equity Schemes - Mid Cap Fund"],
    "Small cap equity funds": ["Equity Scheme - Small Cap Fund", "Equity Schemes - Small Cap Fund"],
    "ELSS (tax-saving) funds": ["Equity Scheme - ELSS", "Equity Schemes - ELSS- Tax Saver Fund"],
    "Aggressive hybrid funds": ["Hybrid Scheme - Aggressive Hybrid Fund", "Hybrid Schemes - Aggressive Hybrid Fund"],
    "Balanced advantage funds": ["Hybrid Scheme - Dynamic Asset Allocation or Balanced Advantage",
                                 "Hybrid Schemes - Balanced Advantage Fund/ Dynamic Asset Allocation"],
    "Corporate bond (debt) funds": ["Debt Scheme - Corporate Bond Fund", "Income/Debt Oriented Schemes - Corporate Bond Fund"],
    "Liquid funds (very low risk)": ["Debt Scheme - Liquid Fund", "Income/Debt Oriented Schemes - Liquid Fund"],
}


def category_returns(fund_list: pd.DataFrame, choice: str) -> dict:
    """Spread of yearly returns earned by the funds in a category over 10 years (5 if too few have 10)."""
    frames = [category_table(fund_list, c) for c in ASSET_CHOICES[choice]]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    if df.empty:
        return {}
    col = "10Y" if df["10Y"].notna().sum() >= 5 else "5Y"
    r = df[col].dropna()
    if r.empty:
        return {}
    return {"period": "10 years" if col == "10Y" else "5 years", "funds": int(len(r)),
            "cautious": float(r.quantile(0.25)), "typical": float(r.median()), "good": float(r.quantile(0.75))}


def future_cost(amount_today: float, years: float, inflation: float) -> float:
    return amount_today * (1 + inflation) ** years


def monthly_sip_needed(target: float, years: float, annual_return: float, lump_sum_now: float = 0.0,
                       step_up: float = 0.0) -> float:
    """Monthly SIP (raised by `step_up` each year) needed to reach `target` in `years`."""
    months = int(round(years * 12))
    if months <= 0:
        return max(0.0, target - lump_sum_now)
    r = (1 + annual_return) ** (1 / 12) - 1

    def fv(sip):
        total, s = lump_sum_now, sip
        for m in range(months):
            if m and m % 12 == 0:
                s *= 1 + step_up
            total = total * (1 + r) + s
        return total * (1 + r) ** 0  # contributions at month end

    if fv(0) >= target:
        return 0.0
    lo, hi = 0.0, target
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fv(mid) < target else (lo, mid)
    return hi


def retirement_corpus(monthly_expense_today: float, years_to_retire: float, years_in_retirement: float,
                      inflation: float, return_after_retirement: float = 0.07) -> float:
    """Money needed on retirement day to pay inflation-rising expenses for `years_in_retirement`."""
    first_year = monthly_expense_today * 12 * (1 + inflation) ** years_to_retire
    real = (1 + return_after_retirement) / (1 + inflation) - 1
    n = years_in_retirement
    return first_year * n if abs(real) < 1e-9 else first_year * (1 - (1 + real) ** -n) / real * (1 + real)


def projection(years: float, annual_return: float, sip: float, lump_sum_now: float = 0.0, step_up: float = 0.0) -> pd.DataFrame:
    """Year-by-year value and money put in, for a chart."""
    r = (1 + annual_return) ** (1 / 12) - 1
    total, s, put_in, rows = lump_sum_now, sip, lump_sum_now, [(0, lump_sum_now, lump_sum_now)]
    for m in range(1, int(round(years * 12)) + 1):
        if m > 1 and (m - 1) % 12 == 0:
            s *= 1 + step_up
        total = total * (1 + r) + s
        put_in += s
        if m % 12 == 0:
            rows.append((m // 12, total, put_in))
    return pd.DataFrame(rows, columns=["Year", "Value", "Put in"])
