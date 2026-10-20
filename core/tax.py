"""Capital-gains tax estimate for the portfolio (India, resident individual).

Default rules: Finance (No. 2) Act 2024, for sales on or after 23 July 2024. Budgets can change these, so every
rate is a setting the user can edit. This is an estimate for planning, not tax advice: it ignores surcharge,
grandfathering (pre-2018 equity), indexation for very old assets, and lot-by-lot (FIFO) purchase records.
"""
import re

import numpy as np
import pandas as pd

DEFAULT_RULES = {
    "equity_lt_months": 12, "equity_stcg": 0.20, "equity_ltcg": 0.125, "equity_ltcg_exemption": 125_000,
    "gold_lt_months": 12, "foreign_lt_months": 24, "other_ltcg": 0.125, "cess": 0.04,
}
CLASS_LABEL = {"equity": "Indian equity", "debt": "Debt", "gold": "Gold / silver", "foreign": "Foreign"}

DEBT_WORDS = r"gilt|bond|debt|sdl|g-sec|gsec|liquid|money market|overnight|duration|credit risk|banking and psu|floater|treasury|government securities|bharat bond"
GOLD_WORDS = r"gold|silver"


def classify(kind: str, ident: str, name: str, fund_category: str = "", etf_underlying: str = "") -> str:
    text = f"{name} {fund_category} {etf_underlying}".lower()
    if kind == "fund":
        if re.search(GOLD_WORDS, text):
            return "gold"
        if re.search(DEBT_WORDS, text) and "equity" not in fund_category.lower() and "hybrid" not in fund_category.lower():
            return "debt"
        return "equity"
    if not str(ident).endswith((".NS", ".BO")):
        return "foreign"
    if re.search(GOLD_WORDS, etf_underlying.lower() + " " + name.lower()):
        return "gold"
    if etf_underlying and re.search(DEBT_WORDS, etf_underlying.lower()):
        return "debt"
    return "equity"


def lt_months(cls: str, rules: dict) -> int | None:
    return {"equity": rules["equity_lt_months"], "gold": rules["gold_lt_months"],
            "foreign": rules["foreign_lt_months"]}.get(cls)  # debt (bought after Apr 2023): always slab


def estimate(holdings: pd.DataFrame, classes: dict, slab_rate: float, rules: dict | None = None,
             today: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Tax if every holding were sold today. `holdings` comes from portfolio.value_portfolio()."""
    rules = {**DEFAULT_RULES, **(rules or {})}
    today = (today or pd.Timestamp.now()).normalize()
    rows = []
    for _, h in holdings.iterrows():
        cls = classes.get(h["id"], "equity")
        bought = pd.Timestamp(h["buy_date"]) if h.get("buy_date") else today
        months = lt_months(cls, rules)
        lt_date = bought + pd.DateOffset(months=months) + pd.Timedelta(days=1) if months else None
        is_lt = bool(lt_date is not None and today >= lt_date)
        gain = float(h["value_inr"] - h["invested_inr"]) if pd.notna(h["value_inr"]) else np.nan
        if cls == "equity":
            rate_txt = f"{rules['equity_ltcg']:.1%} above ₹{rules['equity_ltcg_exemption']:,.0f}/yr" if is_lt else f"{rules['equity_stcg']:.0%}"
        elif cls == "debt":
            rate_txt = "your slab rate"
        else:
            rate_txt = f"{rules['other_ltcg']:.1%}" if is_lt else "your slab rate"
        rows.append({"id": h["id"], "name": h["name"], "class": cls, "bought": bought, "days_held": (today - bought).days,
                     "term": "Long-term" if is_lt else "Short-term", "lt_date": lt_date,
                     "days_to_lt": (lt_date - today).days if (lt_date is not None and not is_lt) else None,
                     "gain": gain, "rate": rate_txt})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}

    # Buckets, with simplified loss set-off: short-term losses can offset any gain, long-term losses only long-term gains.
    def total(mask):
        return float(df.loc[mask, "gain"].sum())

    b = {"eq_st": total((df["class"] == "equity") & (df["term"] == "Short-term")),
         "eq_lt": total((df["class"] == "equity") & (df["term"] == "Long-term")),
         "slab_st": total((df["class"] == "debt") | (df["class"].isin(["gold", "foreign"]) & (df["term"] == "Short-term"))),
         "oth_lt": total(df["class"].isin(["gold", "foreign"]) & (df["term"] == "Long-term"))}

    # Short-term losses may offset any gain (applied to the highest-taxed gains first);
    # long-term losses may offset only long-term gains.
    st_loss = -min(0, b["eq_st"]) - min(0, b["slab_st"])
    b["eq_st"], b["slab_st"] = max(0, b["eq_st"]), max(0, b["slab_st"])
    lt_loss = -min(0, b["eq_lt"]) - min(0, b["oth_lt"])
    b["eq_lt"], b["oth_lt"] = max(0, b["eq_lt"]), max(0, b["oth_lt"])
    for k in ["slab_st", "eq_st", "oth_lt", "eq_lt"]:
        use = min(st_loss, b[k]); b[k] -= use; st_loss -= use
    for k in ["oth_lt", "eq_lt"]:
        use = min(lt_loss, b[k]); b[k] -= use; lt_loss -= use
    eq_st, eq_lt, slab_st, oth_lt = b["eq_st"], b["eq_lt"], b["slab_st"], b["oth_lt"]

    eq_lt_taxable = max(0, eq_lt - rules["equity_ltcg_exemption"])
    tax = (eq_st * rules["equity_stcg"] + eq_lt_taxable * rules["equity_ltcg"] + slab_st * slab_rate
           + oth_lt * rules["other_ltcg"]) * (1 + rules["cess"])
    summary = {"equity_st": eq_st, "equity_lt": eq_lt, "equity_lt_exempt_used": min(eq_lt, rules["equity_ltcg_exemption"]),
               "slab_gains": slab_st, "other_lt": oth_lt, "tax": tax,
               "unused_exemption": max(0, rules["equity_ltcg_exemption"] - eq_lt)}
    return df, summary
