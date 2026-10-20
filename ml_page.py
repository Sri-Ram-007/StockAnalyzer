"""ML outlook page (Phase 6): the honest back-test first, predictions second. Called from app.py."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.ml import FEATURE_INFO, TARGETS, load_report, report_age_days, train_and_backtest, verdict

GREEN, RED, GREY = "#1a7f5a", "#d63a3a", "#9ca3af"


def has_edge(rep) -> bool:
    return bool(rep and "auc" in rep and rep["auc"] >= 0.52 and rep["accuracy"] - rep["baseline_accuracy"] >= 0.005)


def render(list_name, closes, index_series, index_name, names, open_stock):
    st.subheader("🤖 ML outlook: can a model predict the next 3 months?")
    view = st.segmented_control("Test the model on", ["One stock", f"All of {list_name}"], default="One stock",
                                key="ml_view") or "One stock"
    if view == "One stock":
        opts = [t for t in closes.columns if closes[t].notna().sum() > 252 * 5]
        if not opts:
            st.info("No stock in this list has enough history (about 5 years).")
            return
        t = st.selectbox("Stock", opts, format_func=lambda x: f"{names.get(x, x)} ({x})", key="ml_pick")
        single_stock(t, names.get(t, t), closes[t].dropna(), index_series, index_name)
        if st.button(f"Open {names.get(t, t)}", type="tertiary"):
            open_stock(t)
            st.rerun()
        return
    st.write("A machine-learning model (gradient-boosted trees) looks at 15 price-based signals: momentum, trend, "
             "volatility, fall from high, 52-week position, RSI and how the whole market is doing. It estimates the chance "
             "a stock is higher (or beats the index) in about **3 months**. It is then tested **honestly** on years it never saw.")
    c1, c2 = st.columns([3, 2])
    target = c1.segmented_control("Question", list(TARGETS), default="up", format_func=lambda k: TARGETS[k].replace(
        "the market index", index_name))
    target = target or "up"
    rep = load_report(list_name, target)
    stale = rep is None or report_age_days(rep) > 7
    label = "Train and test the model" if rep is None else "Re-train and re-test (uses the latest prices)"
    if c2.button(label, type="primary" if stale else "secondary", use_container_width=True):
        bar = st.progress(0.0, text="Building features and testing year by year…")
        rep = train_and_backtest(list_name, closes, index_series, target,
                                 progress=lambda p: bar.progress(min(p, 1.0), text=f"Testing year by year… {p:.0%}"))
        bar.empty()
        st.rerun()
    if rep is None:
        st.info(f"No model yet for {list_name}. Click **Train and test the model** (about 15–60 seconds).")
        return

    # ---------- Honest result ----------
    edge = has_edge(rep)
    colour = GREEN if edge else RED
    st.markdown(f"""<div class="insight" style="border-left: 5px solid {colour}">
        <div style="font-size:1.15rem; font-weight:700; color:{colour}">
        {'The model showed some edge in testing' if edge else 'The model did NOT beat simple guessing'}</div>
        <div style="margin-top:6px">{verdict(rep)}</div>
        <div class="muted" style="margin-top:6px">Tested on {rep['test_rows']:,} predictions for {rep['stocks']} names in {list_name},
        years {rep['test_years'][0]}–{rep['test_years'][-1]}. Trained {rep['trained_at']}.</div></div>""", unsafe_allow_html=True)
    st.write("")

    cards = [("Model accuracy", f"{rep['accuracy']:.0%}"), ("Always-guess accuracy", f"{rep['baseline_accuracy']:.0%}"),
             ("AUC (0.5 = coin flip)", f"{rep['auc']:.3f}"),
             ("Top 20% picks, avg 3-mo return", f"{rep['top20_avg_return']:+.1%}"),
             ("All stocks, avg 3-mo return", f"{rep['all_avg_return']:+.1%}")]
    for c, (lab, val) in zip(st.columns(5), cards):
        c.markdown(f'<div class="stat-label">{lab}</div><div class="stat-val">{val}</div>', unsafe_allow_html=True)
    st.caption("AUC measures how well the model ranks winners above losers: 0.5 is random, 0.6+ would be meaningful. "
               "If the model's top picks don't beat the average stock, its predictions aren't useful.")

    py = pd.DataFrame(rep["per_year"])
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_bar(x=py["year"], y=py["accuracy"] * 100, name="Model", marker_color="#1f3a5f")
        fig.add_bar(x=py["year"], y=py["baseline_accuracy"] * 100, name="Always guess", marker_color=GREY)
        fig.update_layout(height=300, barmode="group", margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="white",
                          title=dict(text="Accuracy on each unseen year (%)", font=dict(size=14)),
                          legend=dict(orientation="h", y=-0.2), yaxis=dict(gridcolor="#f0f0f0", range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with b:
        fig = go.Figure()
        fig.add_bar(x=py["year"], y=py["top20_avg_return"] * 100, name="Model's top 20%", marker_color=GREEN)
        fig.add_bar(x=py["year"], y=py["all_avg_return"] * 100, name="All stocks", marker_color=GREY)
        fig.update_layout(height=300, barmode="group", margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="white",
                          title=dict(text="Average 3-month return of the model's picks (%)", font=dict(size=14)),
                          legend=dict(orientation="h", y=-0.2), yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("What did the model rely on?"):
        imp = pd.Series(rep.get("feature_importance", {})).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[FEATURE_INFO.get(k, k) for k in imp.index], orientation="h", marker_color="#1f3a5f"))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white",
                          xaxis=dict(title="Drop in AUC when this signal is scrambled", gridcolor="#f0f0f0"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Signals about the whole market rank highest: the model mostly learns 'when the market is calm or falling', "
                   "which does not carry over reliably to new years.")

    with st.expander("How was it tested? (why you can trust these numbers)"):
        st.markdown(f"""
- **No look-ahead:** every signal uses only prices up to that day. Company fundamentals are excluded because past values
  are only available as they are reported today, which would leak future information.
- **Walk-forward:** to test year *Y*, the model is trained only on data before *Y*, with a gap of about
  {int(rep['horizon_days'] * 1.5)} days so training answers never overlap the test year.
- **Baseline:** "always guess the most common outcome" (usually "up", because markets rise more often than they fall).
  A model is only useful if it beats this.
- **Result:** {verdict(rep)} This matches decades of research: short-term price patterns alone are very hard to predict.
""")

    # ---------- Predictions (only behind a warning when there's no edge) ----------
    preds = pd.Series(rep.get("predictions", {}), dtype=float).sort_values(ascending=False)
    if preds.empty:
        return
    st.markdown(f"#### Current probabilities ({rep['latest_date']})")
    show = True
    if not edge:
        show = st.toggle("Show them anyway, for learning only (the model failed its test, so treat these as noise)", value=False)
    if show:
        tbl = pd.DataFrame({"Name": [names.get(t, t) for t in preds.index], "Ticker": preds.index,
                            "Model probability": preds.values})
        ev = st.dataframe(tbl, hide_index=True, use_container_width=True, height=420, on_select="rerun",
                          selection_mode="single-row", column_config={
                              "Model probability": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        if ev.selection.rows:
            open_stock(tbl.iloc[ev.selection.rows[0]]["Ticker"])
            st.rerun()
    st.caption("A probability is not a promise. Not investment advice.")


def stock_box(list_name, ticker):
    """Small ML outlook note on the stock page."""
    rep = load_report(list_name, "up")
    if not rep or ticker not in rep.get("predictions", {}):
        return
    p = rep["predictions"][ticker]
    note = ("The model showed some edge in back-testing." if has_edge(rep)
            else "In honest back-testing this model did <b>not</b> beat simple guessing, so treat this number as noise.")
    st.markdown(f'<div class="insight"><b>🤖 ML outlook:</b> {p:.0%} estimated chance of being higher in about 3 months.'
                f'<br><span class="muted">{note} See the ML outlook page for the full test.</span></div>', unsafe_allow_html=True)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _single(ticker, _close, _index, target, last_date):
    from core.ml import single_stock_test
    return single_stock_test(_close, _index, target)


def single_stock(ticker, name, close, index, index_name):
    """ML tab on the stock page: a model trained and tested only on this stock's own history."""
    st.write("")
    st.markdown(f"#### 🤖 Can a model predict {name}?")
    st.caption("A model trained only on this stock's own past prices (15 signals: momentum, trend, volatility, fall from high, "
               "RSI and the market). It is tested on years it never saw, then asked about today.")
    target = st.segmented_control("Question", list(TARGETS), default="up", key=f"mlq_{ticker}",
                                  format_func=lambda k: TARGETS[k].replace("the market index", index_name)) or "up"
    with st.spinner("Training and testing on this stock's history…"):
        r = _single(ticker, close, index, target, str(close.index[-1].date()))
    if r is None:
        st.info("Not enough price history for this stock (needs about 5 years or more).")
        return
    best_guess = max(r["baseline_accuracy"], r["always_up_accuracy"], 1 - r["always_up_accuracy"])
    beats = r["accuracy"] - best_guess >= 0.05 and r["auc"] >= 0.6
    enough = r["periods"] >= 30
    if beats and enough:
        colour, head = GREEN, "The model did better than guessing on this stock"
    elif beats:
        colour, head = "#d97706", "Better than guessing, but on too few periods to trust"
    else:
        colour, head = RED, "The model did NOT beat simple guessing on this stock"
    q = TARGETS[target].replace("the market index", index_name).lower()
    st.markdown(f"""<div class="insight" style="border-left: 5px solid {colour}">
        <div style="font-size:1.1rem; font-weight:700; color:{colour}">{head}</div>
        <div style="margin-top:6px">On {len(r['test_years'])} years it never saw ({r['test_years'][0]}–{r['test_years'][-1]}),
        it was right <b>{r['accuracy']:.0%}</b> of the time. Simple guesses scored up to <b>{best_guess:.0%}</b>
        ("{q}" happened {r['always_up_accuracy']:.0%} of the time).</div>
        <div class="muted" style="margin-top:6px">That test covers only about {r['periods']} separate 3-month periods, so a few lucky
        calls can swing it a lot. One stock gives far less evidence than testing a whole list of stocks.</div>
        </div>""", unsafe_allow_html=True)
    st.write("")
    fmt = lambda v: "–" if v is None else f"{v:+.1%}"
    cards = [("Model accuracy", f"{r['accuracy']:.0%}"), ("Best simple guess", f"{best_guess:.0%}"),
             ("AUC (0.5 = coin flip)", f"{r['auc']:.2f}"),
             (f"Avg return when it said Yes ({r['n_up']})", fmt(r["avg_when_up"])),
             (f"Avg return when it said No ({r['n_down']})", fmt(r["avg_when_down"]))]
    for c, (lab, val) in zip(st.columns(5), cards):
        c.markdown(f'<div class="stat-label">{lab}</div><div class="stat-val">{val}</div>', unsafe_allow_html=True)
    py = pd.DataFrame(r["per_year"])
    fig = go.Figure()
    fig.add_bar(x=py["year"], y=py["accuracy"] * 100, name="Model", marker_color="#1f3a5f")
    fig.add_bar(x=py["year"], y=py["baseline_accuracy"] * 100, name="Guess the usual outcome", marker_color=GREY)
    fig.update_layout(height=260, barmode="group", margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="white",
                      title=dict(text="Accuracy on each unseen year (%)", font=dict(size=14)),
                      legend=dict(orientation="h", y=-0.25), yaxis=dict(gridcolor="#f0f0f0", range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    p = r["probability"]
    if r["similar_n"]:
        check = (f'<br><b>Reality check:</b> in the test years, when the model said about {p:.0%} '
                 f'({max(p - .1, 0):.0%}–{min(p + .1, 1):.0%}), the outcome actually happened '
                 f'<b>{r["similar_hit"]:.0%}</b> of the time (average move {r["similar_avg"]:+.1%}, '
                 f'about {max(r["similar_periods"], 1)} separate 3-month periods).')
    else:
        check = (f'<br><b>Reality check:</b> the model never gave a number this high or low in its test years, '
                 f'so there is no record of how such calls turned out.')
    st.markdown(f'<div class="insight"><b>Today ({r["latest_date"]}):</b> the model gives a <b>{p:.0%}</b> chance that '
                f'{name} is {q.replace("beats", "beating")}.{check}'
                f'<br><span class="muted">{"Its test record above is decent, but" if beats and enough else "Because the test above is red/amber,"} '
                f'this is not a reliable forecast: its percentages are not real-world odds.</span></div>',
                unsafe_allow_html=True)
    st.caption("Past prices only; news, results and fundamentals are not used. Not investment advice.")
