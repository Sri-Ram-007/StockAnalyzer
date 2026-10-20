"""Insurance comparison (official IRDAI data) and cover calculators.

Scores compare insurers on their published track record: how many claims they pay, how fast, how
many complaints they get and how financially strong they are. They do not rate specific policies,
whose features (waiting periods, room-rent limits, co-pay...) are compared separately.
"""
import json
import os

import numpy as np
import pandas as pd

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "insurance")
LEVELS = {"Off": 0, "Low": 1, "Medium": 2, "High": 3}


def load():
    life = pd.read_csv(os.path.join(DIR, "life.csv"))
    health = pd.read_csv(os.path.join(DIR, "health.csv"))
    meta = json.load(open(os.path.join(DIR, "meta.json")))
    return life, health, meta


def _scale(x, lo, hi):
    """0 at `lo`, 1 at `hi` (works for reversed ranges too), clipped to 0..1."""
    if x is None or pd.isna(x):
        return np.nan
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def _icr_score(x):
    """Healthy retail ICR is roughly 65-95%: below that the insurer pays out little of what it collects,
    above ~100% it is losing money on the book (premiums may rise sharply)."""
    if pd.isna(x):
        return np.nan
    if x < 0.65:
        return float(np.clip((x - 0.45) / 0.20, 0, 1))
    if x > 0.95:
        return float(np.clip(1 - (x - 0.95) / 0.30, 0, 1))
    return 1.0


LIFE_PARTS = {
    "Claims paid (by number)": ("csr_number", lambda x: _scale(x, 0.95, 0.995)),
    "Claims paid (by amount)": ("csr_amount", lambda x: _scale(x, 0.90, 0.99)),
    "Paid within 30 days": ("within_30_days", lambda x: _scale(x, 0.85, 1.0)),
    "Financial strength (solvency)": ("solvency", lambda x: _scale(x, 1.5, 2.5)),
}
HEALTH_PARTS = {
    "Claims paid": ("claims_paid_pct", lambda x: _scale(x, 0.75, 0.96)),
    "Claims rejected (lower is better)": ("claims_repudiated_pct", lambda x: _scale(x, 0.12, 0.01)),
    "Claims ratio in a healthy range": ("icr_retail", _icr_score),
    "Complaints (lower is better)": ("complaints_per_10k_claims", lambda x: _scale(x, 80, 5)),
    "Paid within 3 months": ("paid_within_3_months", lambda x: _scale(x, 0.85, 1.0)),
    "Financial strength (solvency)": ("solvency", lambda x: _scale(x, 1.5, 2.5)),
}
LIFE_DEFAULT = {"Claims paid (by number)": "High", "Claims paid (by amount)": "Medium",
                "Paid within 30 days": "Medium", "Financial strength (solvency)": "Low"}
HEALTH_DEFAULT = {"Claims paid": "High", "Claims rejected (lower is better)": "High",
                  "Claims ratio in a healthy range": "Medium", "Complaints (lower is better)": "Medium",
                  "Paid within 3 months": "Low", "Financial strength (solvency)": "Low"}


def score(df: pd.DataFrame, parts: dict, weights: dict) -> pd.DataFrame:
    out = df.copy()
    total_w = pd.Series(0.0, index=out.index)
    acc = pd.Series(0.0, index=out.index)
    for label, (col, fn) in parts.items():
        w = LEVELS[weights.get(label, "Medium")]
        s = out[col].map(fn)
        out[f"part::{label}"] = s
        acc += s.fillna(0) * w
        total_w += s.notna() * w
    out["score"] = (acc / total_w.replace(0, np.nan) * 100).round()
    out.loc[out["small_sample"].fillna(False), "score"] = np.nan  # too little data to rank fairly
    return out.sort_values("score", ascending=False, na_position="last")


# ---------------- Calculators ----------------
def life_cover_needed(annual_expenses, years, loans, goals, savings, existing_cover, real_return=0.02):
    """Needs-based life cover: money your family would need if your income stopped.
    Future expenses are valued assuming savings earn ~2% a year above inflation."""
    r = real_return
    pv_expenses = annual_expenses * (1 - (1 + r) ** -years) / r if r else annual_expenses * years
    need = pv_expenses + loans + goals - savings - existing_cover
    return {"expenses_part": pv_expenses, "need": max(0.0, need)}


CITY_SI = {"Metro city": (1_000_000, 1_500_000), "Tier-2 city": (750_000, 1_000_000), "Smaller town": (500_000, 750_000)}


def health_cover_guide(city, members, oldest_age, employer_cover):
    """Rule-of-thumb family floater sum insured (a starting point, not a precise requirement)."""
    lo, hi = CITY_SI[city]
    factor = 1 + 0.25 * max(0, members - 2) + (0.5 if oldest_age >= 45 else 0) + (0.5 if oldest_age >= 60 else 0)
    lo, hi = lo * factor, hi * factor
    return {"base_low": lo, "base_high": hi, "own_policy_low": max(0, lo - employer_cover),
            "own_policy_high": max(0, hi - employer_cover), "topup": hi * 2}


# Health policy feature checklist used by the comparison tool
FEATURES = [
    ("room_rent", "Room rent limit", ["No limit / any room", "Single private room", "Capped (e.g. 1% of cover per day)"], [1.0, 0.8, 0.2]),
    ("copay", "Co-payment (you pay part of every claim)", ["None", "10%", "20% or more"], [1.0, 0.5, 0.1]),
    ("ped", "Waiting period for existing illnesses", ["1 year or less", "2 years", "3 years", "4 years"], [1.0, 0.8, 0.5, 0.2]),
    ("restore", "Restoration (cover refilled after a claim)", ["Unlimited", "Once a year", "None"], [1.0, 0.7, 0.2]),
    ("ncb", "No-claim bonus", ["50% or more per year", "10-25% per year", "None"], [1.0, 0.6, 0.2]),
    ("sublimits", "Sub-limits on treatments (e.g. cataract caps)", ["None", "A few", "Many"], [1.0, 0.5, 0.1]),
]


def policy_feature_score(choices: dict) -> float:
    vals = []
    for key, _, options, points in FEATURES:
        if choices.get(key) in options:
            vals.append(points[options.index(choices[key])])
    return round(100 * float(np.mean(vals))) if vals else np.nan
