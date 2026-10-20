"""Build clean insurer-wise datasets from IRDAI's official 'Handbook on Indian Insurance Statistics'.

Run once per year after IRDAI publishes a new handbook:
    python -m core.insurance_data "<folder with Part I.xlsx ... Part III.xlsx>" 2024-25

Writes data/insurance/life.csv, data/insurance/health.csv and data/insurance/meta.json.
Source: https://irdai.gov.in/handbook-of-indian-insurance
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "insurance")
os.makedirs(OUT, exist_ok=True)

ALIASES = {"lic": "LIC of India", "lifecorporationof": "LIC of India", "licof": "LIC of India"}


def key(name: str) -> str:
    """Normalise insurer names so the same company matches across tables."""
    s = str(name).lower().replace("&", "and")
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z ]", " ", s)
    stop = {"insurance", "company", "co", "ltd", "limited", "the", "india", "of", "pvt", "general", "health",
            "and", "allied", "corporation"}
    words = [w for w in s.split() if w not in stop]
    k = "".join(words).replace("life", "")
    return {"lic": "lic", "licof": "lic", "": "lic"}.get(k, k)


def _read(base, part, sheet):
    return pd.read_excel(os.path.join(base, part), sheet_name=sheet, header=None)


def _year_block(df, year, header_rows=range(0, 6)):
    """Start and end column of the block of columns belonging to `year`."""
    for r in header_rows:
        cols = [c for c in df.columns if str(df.iat[r, c]).strip() == year]
        if cols:
            start = cols[0]
            later = [c for c in df.columns if c > start and re.fullmatch(r"\d{4}-\d{2}", str(df.iat[r, c]).strip())]
            return r, start, (later[0] if later else df.shape[1])
    raise ValueError(f"year {year} not found")


def _num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return np.nan


def _insurer_rows(df, name_col=1, first_row=3):
    """Rows that hold an insurer (have a serial number in column 0 and a name)."""
    out = []
    for r in range(first_row, len(df)):
        sno, name = df.iat[r, 0], df.iat[r, name_col]
        if pd.notna(name) and re.fullmatch(r"\d+(\.0)?", str(sno).strip() or "x"):
            out.append((r, str(name).strip()))
    return out


# ---------------- Life ----------------
def life_claims(base, year):
    df = _read(base, "Part I.xlsx", "15")
    hr, c0, c1 = _year_block(df, year)
    cats = {}
    for c in range(c0, c1, 2):
        label = str(df.iat[hr + 1, c]).lower()
        for k in ["pending at start", "intimated", "total", "paid", "repudiated", "rejected", "unclaimed", "pending at end"]:
            if k in label and k not in cats:
                cats[k] = c
                break
    rows = []
    for r, name in _insurer_rows(df):
        tot = _num(df.iat[r, cats["total"]])
        if not tot or np.isnan(tot):
            continue
        get = lambda k, off=0: _num(df.iat[r, cats[k] + off]) if k in cats else np.nan
        rows.append({"key": key(name), "name_claims": name, "claims_total": tot, "claims_paid": get("paid"),
                     "csr_number": get("paid") / tot, "csr_amount": get("paid", 1) / get("total", 1),
                     "repudiated_pct": get("repudiated") / tot, "rejected_pct": get("rejected") / tot,
                     "pending_end": get("pending at end"), "amount_paid_cr": get("paid", 1)})
    return pd.DataFrame(rows)


def life_speed(base, year):
    """Share of individual death claims settled within 30 days of intimation (Table 18)."""
    df = _read(base, "Part I.xlsx", "18")
    out = []
    for c in range(1, df.shape[1]):
        insurer = df.iat[2, c]
        if pd.isna(insurer):
            continue
        # find this insurer's 2024-25 column inside its block
        c_end = next((x for x in range(c + 1, df.shape[1]) if pd.notna(df.iat[2, x])), df.shape[1])
        ycol = next((x for x in range(c, c_end) if str(df.iat[3, x]).strip() == year), None)
        if ycol is None:
            continue
        vals = [_num(df.iat[r, ycol]) for r in range(4, 9)]
        tot = np.nansum(vals)
        if tot > 0:
            out.append({"key": key(insurer), "within_30_days": vals[0] / tot})
    return pd.DataFrame(out)


def solvency(base, part, sheet, label="March 2025"):
    df = _read(base, part, sheet)
    hr = next(r for r in range(0, 5) if any(str(v).strip() == label for v in df.iloc[r]))
    col = [c for c in df.columns if str(df.iat[hr, c]).strip() == label][-1]
    return pd.DataFrame([{"key": key(n), "solvency": _num(df.iat[r, col])} for r, n in _insurer_rows(df, first_row=hr + 1)])


def grievances(base, part, sheet, year):
    df = _read(base, part, sheet)
    hr, c0, c1 = _year_block(df, year)
    labels = {str(df.iat[hr + 1, c]).strip().lower(): c for c in range(c0, c1)}
    rep = next(c for l, c in labels.items() if "reported" in l)
    res = next(c for l, c in labels.items() if "resolved" in l or "attended" in l)
    # newer years repeat their own S.No./Insurer columns inside the year block
    name_col = labels.get("insurer", 1)
    rows = []
    for r in range(hr + 2, len(df)):
        n = df.iat[r, name_col]
        if pd.notna(n) and pd.notna(df.iat[r, rep]) and not str(n).lower().startswith(("public", "private", "stand", "special", "total", "re")):
            rows.append({"key": key(n), "complaints": _num(df.iat[r, rep]), "complaints_resolved": _num(df.iat[r, res])})
    return pd.DataFrame(rows).drop_duplicates("key")


# ---------------- Health ----------------
def health_icr(base, year):
    df = _read(base, "Part III.xlsx", "62")
    hr, c0, c1 = _year_block(df, year)
    seg = {}
    for c in range(c0, c1):
        lab = str(df.iat[hr + 1, c]).lower()
        if lab != "nan":
            seg[lab] = c
    # labels changed over the years: "Individual Family Floater" / "Family/Floater Insurance excluding Individual",
    # and "Individual Other than Family Floater" / "Individual Insurance excluding Family/Floater"
    ff = next(c for l, c in seg.items() if ("family floater" in l or "family/floater insurance" in l) and "other than" not in l)
    oth = next(c for l, c in seg.items() if "other than family" in l or l.startswith("individual insurance excluding"))
    tot = next(c for l, c in seg.items() if l.startswith("total"))
    rows = []
    for r, n in _insurer_rows(df, first_row=hr + 2):
        nep = _num(df.iat[r, ff]) + _num(df.iat[r, oth])
        clm = _num(df.iat[r, ff + 1]) + _num(df.iat[r, oth + 1])
        rows.append({"key": key(n), "name_icr": n, "retail_premium_lakh": nep,
                     "icr_retail": clm / nep if nep and nep > 0 else np.nan,
                     "icr_total": _num(df.iat[r, tot + 2])})
    return pd.DataFrame(rows)


def health_claims(base, year):
    df = _read(base, "Part II.xlsx", "53")
    hr, c0, c1 = _year_block(df, year)
    lab = {str(df.iat[hr + 2, c]).lower(): c for c in range(c0, c1)}
    g = lambda word: next(c for l, c in lab.items() if word in l)
    rows = []
    for r, n in _insurer_rows(df, first_row=hr + 3):
        start, new = _num(df.iat[r, g("start")]), _num(df.iat[r, g("intimated")])
        total = np.nansum([start, new])
        if not total:
            continue
        rows.append({"key": key(n), "name_claims": n, "claims_handled": total,
                     "claims_paid_pct": _num(df.iat[r, g("paid during")]) / total,
                     "claims_repudiated_pct": _num(df.iat[r, g("repudiated")]) / total,
                     "paid_within_3_months": _num(df.iat[r, g("< 3 months")]) / 100})
    return pd.DataFrame(rows)


def _align(right: pd.DataFrame, left_keys) -> pd.DataFrame:
    """Map each insurer key in `right` to the closest key in the main table (IRDAI spells names
    differently across tables, e.g. 'Aegas Federal' vs 'Ageas Federal', 'Zurich Kotak' vs 'Kotak Mahindra')."""
    import difflib
    left = list(dict.fromkeys(left_keys))

    manual = {"zurichkotak": "kotakmahindra", "zurichkotakmahindra": "kotakmahindra"}

    def best(k):
        k = manual.get(k, k)
        if k in left:
            return k
        for lk in left:  # one name contained in the other, e.g. 'magmahdi' / 'magma'
            if len(lk) >= 4 and len(k) >= 4 and (lk in k or k in lk):
                return lk
        m = difflib.get_close_matches(k, left, n=1, cutoff=0.72)
        return m[0] if m else k

    out = right.copy()
    out["key"] = out["key"].map(best)
    return out.drop_duplicates("key")


def build(base: str, year: str = "2024-25"):
    sol_label = f"March 20{year[-2:]}"
    life = life_claims(base, year).drop_duplicates("key")
    for extra in [life_speed(base, year), solvency(base, "Part I.xlsx", "23", sol_label),
                  grievances(base, "Part I.xlsx", "37", year)]:
        life = life.merge(_align(extra, life["key"]), on="key", how="left")
    life["small_sample"] = life["claims_total"] < 300
    life["name"] = life["name_claims"].replace({"LIC": "LIC of India"})

    standalone = {"adityabirla", "care", "galaxy", "manipalcigna", "narayana", "nivabupa", "star"}
    health = health_icr(base, year).drop_duplicates("key")
    for extra in [health_claims(base, year), solvency(base, "Part II.xlsx", "49", sol_label),
                  grievances(base, "Part II.xlsx", " 56", year)]:
        health = health.merge(_align(extra, health["key"]), on="key", how="left")
    health = health[health["retail_premium_lakh"] > 0].copy()
    health["name"] = health["name_icr"]
    health["type"] = np.where(health["key"].isin(standalone), "Standalone health insurer", "General insurer")
    health["complaints_per_10k_claims"] = health["complaints"] / health["claims_handled"] * 10000
    # very small retail health books give unstable ratios
    health["small_sample"] = (health["retail_premium_lakh"] < 5000) | (health["claims_handled"] < 2000)

    life.drop(columns=["name_claims"]).to_csv(os.path.join(OUT, "life.csv"), index=False)
    health.drop(columns=["name_icr", "name_claims"], errors="ignore").to_csv(os.path.join(OUT, "health.csv"), index=False)
    json.dump({"year": year, "solvency_as_of": sol_label,
               "source": "IRDAI Handbook on Indian Insurance Statistics " + year,
               "url": "https://irdai.gov.in/handbook-of-indian-insurance"}, open(os.path.join(OUT, "meta.json"), "w"))
    return life, health


if __name__ == "__main__":
    life, health = build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "2024-25")
    print(f"life insurers: {len(life)}, health insurers: {len(health)}")
