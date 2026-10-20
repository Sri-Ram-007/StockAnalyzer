"""Phase 6: machine-learning outlook, honestly back-tested.

Question: "Will this stock be higher (or beat the market) in about 3 months?"

Honesty rules this module follows:
  1. Features only use prices up to that day (no future information, no fundamentals that would be
     "known" only later).
  2. Walk-forward testing: train on earlier years, test on the next year the model never saw,
     with a gap the length of the prediction horizon so training labels never overlap test dates.
  3. Every result is compared with a simple baseline ("always guess the most common outcome"),
     so you can see whether the model adds anything at all.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from core.data import DATA_DIR

MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
HORIZON = 63           # trading days, about 3 months
SAMPLE_EVERY = 5       # use every 5th day to reduce near-duplicate rows
TARGETS = {"up": "Higher in 3 months", "beat": "Beats the market index in 3 months"}

FEATURE_INFO = {
    "ret_1m": "Return over the last month", "ret_3m": "Return over the last 3 months",
    "ret_6m": "Return over the last 6 months", "ret_12m": "Return over the last 12 months",
    "vol_1m": "Volatility (last month)", "vol_3m": "Volatility (last 3 months)",
    "dd_1y": "Fall from 1-year high", "pos_52w": "Position in 52-week range",
    "vs_ma50": "Distance from 50-day average", "vs_ma200": "Distance from 200-day average",
    "rsi14": "RSI (14 days)", "rel_3m": "3-month return vs the index",
    "idx_ret_3m": "Index return (3 months)", "idx_dd_1y": "Index fall from its high", "idx_vol_1m": "Index volatility",
}


def _rsi(c: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_features(closes: pd.DataFrame, index: pd.Series) -> pd.DataFrame:
    """Long table (date, ticker) of features computed only from past prices."""
    c = closes.sort_index()
    idx = index.reindex(c.index).ffill()
    daily = c.pct_change(fill_method=None)
    hi, lo = c.rolling(252, min_periods=126).max(), c.rolling(252, min_periods=126).min()
    idx_r63 = idx / idx.shift(63) - 1
    wide = {
        "ret_1m": c / c.shift(21) - 1, "ret_3m": c / c.shift(63) - 1, "ret_6m": c / c.shift(126) - 1,
        "ret_12m": c / c.shift(252) - 1,
        "vol_1m": daily.rolling(21).std() * np.sqrt(252), "vol_3m": daily.rolling(63).std() * np.sqrt(252),
        "dd_1y": c / hi - 1, "pos_52w": (c - lo) / (hi - lo).replace(0, np.nan),
        "vs_ma50": c / c.rolling(50).mean() - 1, "vs_ma200": c / c.rolling(200).mean() - 1,
        "rsi14": _rsi(c),
        "rel_3m": (c / c.shift(63) - 1).sub(idx_r63, axis=0),
    }
    feats = pd.concat({k: v.stack(future_stack=True) for k, v in wide.items()}, axis=1)
    feats.index.names = ["date", "ticker"]
    market = pd.DataFrame({"idx_ret_3m": idx_r63,
                           "idx_dd_1y": idx / idx.rolling(252, min_periods=126).max() - 1,
                           "idx_vol_1m": idx.pct_change().rolling(21).std() * np.sqrt(252)})
    feats = feats.join(market, on="date")
    return feats


def build_labels(closes: pd.DataFrame, index: pd.Series, horizon: int = HORIZON) -> pd.DataFrame:
    c = closes.sort_index()
    idx = index.reindex(c.index).ffill()
    fwd = c.shift(-horizon) / c - 1
    idx_fwd = idx.shift(-horizon) / idx - 1
    lab = pd.concat({"fwd_ret": fwd.stack(future_stack=True),
                     "fwd_excess": fwd.sub(idx_fwd, axis=0).stack(future_stack=True)}, axis=1)
    lab.index.names = ["date", "ticker"]
    return lab


def _model():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=24,
                                          min_samples_leaf=300, l2_regularization=1.0, random_state=42)


def train_and_backtest(name: str, closes: pd.DataFrame, index: pd.Series, target: str = "up",
                       progress=None) -> dict:
    """Walk-forward back-test year by year, then train a final model on all labelled data."""
    t0 = time.time()
    closes = closes.loc[:, closes.notna().sum() > 300]
    feats = build_features(closes, index)
    labels = build_labels(closes, index)
    data = feats.join(labels).dropna(subset=["ret_3m", "dd_1y"])
    dates = data.index.get_level_values("date")
    keep_dates = pd.Index(sorted(dates.unique()))[::SAMPLE_EVERY]
    sampled = data[dates.isin(keep_dates)]
    labelled = sampled.dropna(subset=["fwd_ret"]).copy()
    labelled["y"] = (labelled["fwd_ret" if target == "up" else "fwd_excess"] > 0).astype(int)
    X_cols = list(FEATURE_INFO)

    ldates = labelled.index.get_level_values("date")
    years = sorted(ldates.year.unique())
    test_years = [y for y in years if y >= years[0] + 3]
    preds, per_year = [], []
    for k, yr in enumerate(test_years, 1):
        test_mask = ldates.year == yr
        cutoff = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=int(HORIZON * 1.5))  # embargo gap
        train_mask = ldates < cutoff
        if train_mask.sum() < 2000 or test_mask.sum() < 200:
            continue
        m = _model().fit(labelled.loc[train_mask, X_cols], labelled.loc[train_mask, "y"])
        te = labelled.loc[test_mask]
        p = m.predict_proba(te[X_cols])[:, 1]
        base_rate_train = labelled.loc[train_mask, "y"].mean()
        part = pd.DataFrame({"year": yr, "y": te["y"].to_numpy(), "p": p,
                             "fwd": te["fwd_ret" if target == "up" else "fwd_excess"].to_numpy(),
                             "base_guess": int(base_rate_train >= 0.5)}, index=te.index)
        preds.append(part)
        acc = ((p >= 0.5).astype(int) == part["y"]).mean()
        base_acc = (part["y"] == part["base_guess"]).mean()
        auc = roc_auc_score(part["y"], p) if part["y"].nunique() > 1 else np.nan
        q = pd.qcut(p, 5, labels=False, duplicates="drop")
        per_year.append({"year": int(yr), "rows": int(len(part)), "accuracy": float(acc), "baseline_accuracy": float(base_acc),
                         "auc": float(auc) if not np.isnan(auc) else None,
                         "top20_avg_return": float(part["fwd"][q == q.max()].mean()),
                         "all_avg_return": float(part["fwd"].mean()),
                         "bottom20_avg_return": float(part["fwd"][q == q.min()].mean())})
        if progress:
            progress(k / (len(test_years) + 1))

    report = {"name": name, "target": target, "target_label": TARGETS[target], "horizon_days": HORIZON,
              "trained_at": time.strftime("%Y-%m-%d %H:%M"), "stocks": int(closes.shape[1])}
    if preds:
        allp = pd.concat(preds)
        q = pd.qcut(allp["p"], 5, labels=False, duplicates="drop")
        calib = allp.assign(bin=pd.cut(allp["p"], [0, .3, .4, .5, .6, .7, 1.0])).groupby("bin", observed=True).agg(
            predicted=("p", "mean"), actual=("y", "mean"), rows=("y", "size")).reset_index()
        calib["bin"] = calib["bin"].astype(str)
        report.update({
            "test_years": [int(y) for y in allp["year"].unique()],
            "test_rows": int(len(allp)),
            "accuracy": float(((allp["p"] >= 0.5).astype(int) == allp["y"]).mean()),
            "baseline_accuracy": float((allp["y"] == allp["base_guess"]).mean()),
            "auc": float(roc_auc_score(allp["y"], allp["p"])),
            "share_positive": float(allp["y"].mean()),
            "top20_avg_return": float(allp["fwd"][q == q.max()].mean()),
            "all_avg_return": float(allp["fwd"].mean()),
            "bottom20_avg_return": float(allp["fwd"][q == q.min()].mean()),
            "top20_hit_rate": float(allp["y"][q == q.max()].mean()),
            "per_year": per_year,
            "calibration": calib.to_dict("records"),
        })

    final = _model().fit(labelled[X_cols], labelled["y"])
    imp_proxy = _importance(final, labelled[X_cols].iloc[-20000:], labelled["y"].iloc[-20000:])
    report["feature_importance"] = imp_proxy

    latest_date = data.index.get_level_values("date").max()
    latest = data.xs(latest_date, level="date")[X_cols].dropna(subset=["ret_3m", "dd_1y"])
    report["latest_date"] = str(latest_date.date())
    report["predictions"] = {t: float(p) for t, p in zip(latest.index, final.predict_proba(latest)[:, 1])}
    report["seconds"] = round(time.time() - t0, 1)

    joblib.dump(final, MODEL_DIR / f"{_key(name, target)}.joblib")
    (MODEL_DIR / f"{_key(name, target)}.json").write_text(json.dumps(report, default=str))
    if progress:
        progress(1.0)
    return report


def _importance(model, X, y, repeats: int = 3) -> dict:
    """Permutation importance (how much the AUC drops when a feature is shuffled)."""
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, n_repeats=repeats, random_state=0, scoring="roc_auc", n_jobs=1)
    return {c: float(v) for c, v in sorted(zip(X.columns, r.importances_mean), key=lambda kv: -kv[1])}


def _key(name, target):
    return f"{name.replace(' ', '_').replace('&', 'and')}_{target}"


def load_report(name: str, target: str = "up") -> dict | None:
    f = MODEL_DIR / f"{_key(name, target)}.json"
    return json.loads(f.read_text()) if f.exists() else None


def report_age_days(rep: dict) -> float:
    try:
        return (pd.Timestamp.now() - pd.Timestamp(rep["trained_at"])).days
    except Exception:
        return 999


def verdict(rep: dict) -> str:
    """Plain-English summary of how useful the model was on unseen years."""
    if not rep or "auc" not in rep:
        return "Not enough history to test the model."
    auc, acc, base = rep["auc"], rep["accuracy"], rep["baseline_accuracy"]
    edge = acc - base
    if auc < 0.52 or edge < 0.005:
        level = "no real edge"
    elif auc < 0.56:
        level = "a small edge"
    else:
        level = "a modest edge"
    return (f"On {len(rep['test_years'])} years it never saw, the model was right {acc:.0%} of the time, versus "
            f"{base:.0%} for simply always guessing the most common outcome: {level}.")


def _single_model():
    """Smaller trees: one stock has far fewer rows than a whole list."""
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=8,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=42)


def single_stock_test(close: pd.Series, index: pd.Series, target: str = "up", every: int = 3) -> dict | None:
    """Train and walk-forward test a model on ONE stock's own history.

    Same honesty rules as the list model: past prices only, train on earlier years, test on the next
    unseen year with an embargo gap, and compare with always guessing the most common outcome.
    """
    closes = close.dropna().to_frame("S")
    if len(closes) < 252 * 5:
        return None
    feats = build_features(closes, index)
    labels = build_labels(closes, index)
    data = feats.join(labels).dropna(subset=["ret_12m", "dd_1y"])
    data = data.iloc[::every]
    labelled = data.dropna(subset=["fwd_ret"]).copy()
    ycol = "fwd_ret" if target == "up" else "fwd_excess"
    labelled = labelled.dropna(subset=[ycol])
    labelled["y"] = (labelled[ycol] > 0).astype(int)
    X_cols = list(FEATURE_INFO)
    ld = labelled.index.get_level_values("date")
    years = sorted(ld.year.unique())
    preds, per_year = [], []
    for yr in [y for y in years if y >= years[0] + 4]:
        test_mask = ld.year == yr
        train_mask = ld < pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=int(HORIZON * 1.5))
        ytr = labelled.loc[train_mask, "y"]
        if train_mask.sum() < 250 or test_mask.sum() < 30 or ytr.nunique() < 2:
            continue
        m = _single_model().fit(labelled.loc[train_mask, X_cols], ytr)
        te = labelled.loc[test_mask]
        p = m.predict_proba(te[X_cols])[:, 1]
        guess = int(ytr.mean() >= 0.5)
        part = pd.DataFrame({"year": yr, "y": te["y"].to_numpy(), "p": p, "fwd": te[ycol].to_numpy(), "guess": guess})
        preds.append(part)
        per_year.append({"year": int(yr), "accuracy": float(((p >= 0.5) == part["y"]).mean()),
                         "baseline_accuracy": float((part["y"] == guess).mean())})
    if not preds:
        return None
    allp = pd.concat(preds)
    said_up, said_down = allp[allp["p"] >= 0.5], allp[allp["p"] < 0.5]
    final = _single_model().fit(labelled[X_cols], labelled["y"])
    latest = data[X_cols].iloc[[-1]]
    return {
        "target": target, "target_label": TARGETS[target], "test_years": [int(y) for y in allp["year"].unique()],
        "test_rows": int(len(allp)), "accuracy": float(((allp["p"] >= 0.5) == allp["y"]).mean()),
        "periods": int(len(allp) * every / HORIZON),  # roughly independent 3-month periods in the test
        "baseline_accuracy": float((allp["y"] == allp["guess"]).mean()),
        "always_up_accuracy": float(allp["y"].mean()),
        "auc": float(roc_auc_score(allp["y"], allp["p"])) if allp["y"].nunique() > 1 else 0.5,
        "share_positive": float(allp["y"].mean()),
        "avg_when_up": float(said_up["fwd"].mean()) if len(said_up) else None, "n_up": int(len(said_up)),
        "avg_when_down": float(said_down["fwd"].mean()) if len(said_down) else None, "n_down": int(len(said_down)),
        "avg_all": float(allp["fwd"].mean()), "per_year": per_year,
        "latest_date": str(data.index.get_level_values("date")[-1].date()),
        "probability": float(final.predict_proba(latest)[:, 1][0]),
    }
