"""IPO pages: open & upcoming, recent listings, how IPOs perform, and US IPOs. Called from app.py."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.ipo import (india_open_and_upcoming, india_past_ipos, listing_prices, performance, pop_vs_later,
                      summary_stats, us_ipos)

GREEN, RED, GREY = "#1a7f5a", "#d63a3a", "#9ca3af"


def sp(v):
    """Signed percent, or a dash when there is no value yet."""
    return "–" if v is None or pd.isna(v) else f"{v:+.0%}"


@st.cache_data(ttl=1800, show_spinner=False)
def _open():
    return india_open_and_upcoming()


@st.cache_data(ttl=21600, show_spinner=False)
def _perf():
    past = india_past_ipos()
    px = listing_prices(past["Ticker"].tolist(), "india")
    return performance(past, px), len(past)


@st.cache_data(ttl=21600, show_spinner=False)
def _us():
    return us_ipos()


@st.cache_data(ttl=21600, show_spinner=False)
def _us_perf(tickers):
    px = listing_prices(list(tickers), "us")
    return px


def render(ss, open_stock):
    pages = ["Open & upcoming", "Recent listings", "How IPOs perform", "US IPOs"]
    nav = st.segmented_control("Menu", pages, default=ss.page if ss.page in pages else pages[0],
                               key=f"inav_{ss.page}", label_visibility="collapsed")
    if nav and nav != ss.page:
        ss.page = nav
        st.rerun()
    if ss.page not in pages:
        ss.page = pages[0]
    {"Open & upcoming": open_page, "Recent listings": listings_page,
     "How IPOs perform": evidence_page, "US IPOs": us_page}[ss.page](ss, open_stock)


# ---------------- India: open & upcoming ----------------
def open_page(ss, open_stock):
    st.subheader("Indian IPOs: open, upcoming and just closed")
    with st.spinner("Checking NSE…"):
        df = _open()
    if df.empty:
        st.info("Could not reach NSE right now. Try again in a few minutes.")
        return
    badge = {"Forthcoming": ("#1f3a5f", "Upcoming"), "Active": (GREEN, "Open now"), "Open": (GREEN, "Open now"),
             "Closed": (GREY, "Closed")}
    for _, r in df.iterrows():
        col, txt = badge.get(r["Status"], (GREY, r["Status"]))
        subs = r["Subscribed (times)"]
        subs_txt = "" if pd.isna(subs) else (
            f' · <b>{subs:,.1f}× subscribed</b> '
            f'<span class="muted">({"very high demand: allotment is a lottery, most applicants get nothing" if subs >= 10 else "good demand" if subs >= 2 else "weak demand"})</span>')
        st.markdown(f"""<div class="insight" style="margin-bottom:10px">
            <span style="background:{col}; color:white; border-radius:6px; padding:2px 8px; font-size:.8rem">{txt}</span>
            <span class="muted"> {r['Type']}</span>
            <div style="font-size:1.1rem; font-weight:700; margin-top:6px">{r['Company']}</div>
            <div class="muted">{r['Opens']:%d %b} → {r['Closes']:%d %b %Y} · Price band {r['Price band'].replace('Rs.', '₹')}{subs_txt}</div>
            </div>""", unsafe_allow_html=True)
    st.caption("Source: NSE (subscription figures update during the bidding days). 'Subscribed 30×' means bids for 30 times "
               "the shares on offer. See 'How IPOs perform' for what usually happens after listing. Not investment advice.")


# ---------------- India: recent listings ----------------
def listings_page(ss, open_stock):
    st.subheader("Recently listed Indian IPOs: how are they doing?")
    with st.spinner("Loading listing data (first time about 1–2 minutes)…"):
        perf, total = _perf()
    if perf.empty:
        st.info("No listing data available right now.")
        return
    c1, c2 = st.columns([2, 3])
    years = c1.segmented_control("Listed in the last", ["6 months", "1 year", "2 years", "5 years"], default="1 year")
    typ = c2.segmented_control("Type", ["All", "Mainboard", "SME"], default="Mainboard")
    days = {"6 months": 183, "1 year": 365, "2 years": 730, "5 years": 1826}[years or "1 year"]
    p = perf[perf["listing_date"] >= pd.Timestamp.now() - pd.Timedelta(days=days)]
    if typ and typ != "All":
        p = p[p["Type"] == typ]
    p = p.sort_values("listing_date", ascending=False).reset_index(drop=True)
    if p.empty:
        st.info("No IPOs in this range.")
        return
    above = (p["ret_now"] > 0).mean()
    for c, (lab, val) in zip(st.columns(4), [("IPOs", f"{len(p)}"), ("Listed above issue price", f"{(p['listing_close_gain'] > 0).mean():.0%}"),
                                              ("Above issue price today", f"{above:.0%}"),
                                              ("Median return since issue", f"{p['ret_now'].median():+.0%}")]):
        c.markdown(f'<div class="idx-card"><div class="idx-name">{lab}</div><div class="idx-val">{val}</div></div>', unsafe_allow_html=True)
    st.write("")
    tbl = pd.DataFrame({
        "Company": p["Company"], "Type": p["Type"], "Listed on": p["listing_date"].dt.strftime("%d %b %Y"),
        "Issue price": p["issue_price"].map(lambda v: f"₹{v:,.0f}"), "Price now": p["price_now"].map(lambda v: f"₹{v:,.2f}"),
        "Listing day": p["listing_close_gain"].map(sp), "After 1 month": p["ret_1m"].map(sp),
        "After 6 months": p["ret_6m"].map(sp), "Now vs issue price": p["ret_now"].map(sp),
        "Below its high": p["below_high"].map(lambda v: "–" if pd.isna(v) else f"{v:.0%}"),
    })
    st.caption("All returns are measured from the IPO issue price. Click a row to open the stock.")
    ev = st.dataframe(tbl, hide_index=True, use_container_width=True, height=520, on_select="rerun", selection_mode="single-row",
                      column_config={"Listing day": st.column_config.TextColumn(help="Listing-day close vs issue price"),
                                     "After 1 month": st.column_config.TextColumn(help="'–' = not listed that long yet")})
    if ev.selection.rows:
        r = p.iloc[ev.selection.rows[0]]
        ss.market = "India"
        open_stock(r["Ticker"], r["Company"])
        st.rerun()
    st.caption(f"{len(perf)} of {total} NSE IPOs since 2016 have price data on Yahoo Finance; renamed or delisted companies are "
               "missing. Not investment advice.")


# ---------------- India: evidence ----------------
def evidence_page(ss, open_stock):
    st.subheader("How have Indian IPOs actually performed?")
    st.write("Using every NSE IPO since 2016 with price data: what happened on listing day, and what happened next.")
    with st.spinner("Loading listing data (first time about 1–2 minutes)…"):
        perf, total = _perf()
    if perf.empty:
        st.info("No listing data available right now.")
        return
    stats = summary_stats(perf)
    allrow = stats.iloc[0]
    st.markdown(f"""<div class="insight">
        <b>{allrow['Listed above issue price']:.0%}</b> of IPOs closed above their issue price on listing day
        (median gain <b>{allrow['Median listing-day gain']:+.0%}</b>). But one year later only
        <b>{allrow['Above issue price after 1 year']:.0%}</b> were still above the issue price, with a median return of
        <b>{allrow['Median return after 1 year']:+.0%}</b>.<br><br>
        In other words, an IPO is roughly a <b>coin flip</b> after a year, even though most start with a listing-day gain.
        </div>""", unsafe_allow_html=True)
    st.write("")
    show = stats.copy()
    for col in ["Listed above issue price", "Above issue price after 1 year", "Above issue price today"]:
        show[col] = show[col].map(lambda v: "–" if pd.isna(v) else f"{v:.0%}")
    for col in ["Median listing-day gain", "Median return after 1 year"]:
        show[col] = show[col].map(lambda v: "–" if pd.isna(v) else f"{v:+.0%}")
    st.dataframe(show, hide_index=True, use_container_width=True)

    st.markdown("#### Do big listing-day pops last?")
    pv = pop_vs_later(perf)
    if not pv.empty:
        fig = go.Figure()
        fig.add_bar(x=pv["Listing day"], y=pv["median_listing"] * 100, name="Median listing-day gain", marker_color="#1f3a5f")
        fig.add_bar(x=pv["Listing day"], y=pv["median_1y"] * 100, name="Median return after 1 year",
                    marker_color=[GREEN if v >= 0 else RED for v in pv["median_1y"]])
        fig.update_layout(height=340, barmode="group", plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", y=-0.2), yaxis=dict(title="% vs issue price", gridcolor="#f0f0f0"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        worst = pv.iloc[0]
        st.markdown(f'<div class="insight">IPOs that <b>listed below</b> their issue price were a median '
                    f'<b>{worst["median_1y"]:+.0%}</b> a year later, so a weak listing is a real warning sign. '
                    f'Strong listings did better on average, but about <b>{pv["gave_back"].mean():.0%}</b> of IPOs in every group '
                    f'ended the year below their listing-day price: the pop is often given back.</div>', unsafe_allow_html=True)

    st.markdown("#### Spread of 1-year returns")
    r1 = perf["ret_1y"].dropna().clip(-1, 3) * 100
    fig = go.Figure(go.Histogram(x=r1, nbinsx=40, marker_color="#1f3a5f"))
    fig.add_vline(x=0, line_color=RED, line_dash="dot")
    fig.update_layout(height=300, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(title="Return after 1 year vs issue price (%)"), yaxis=dict(title="IPOs", gridcolor="#f0f0f0"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Based on {len(perf)} of {total} NSE IPOs since 2016. About {total - len(perf)} renamed or delisted companies have no "
               "price data, and failed small companies are more likely to be among them, so these results are, if anything, "
               "slightly optimistic. Not investment advice.")


# ---------------- US ----------------
def us_page(ss, open_stock):
    st.subheader("US IPOs (Nasdaq calendar)")
    with st.spinner("Loading the Nasdaq IPO calendar…"):
        d = _us()
    hide_spac = st.toggle("Hide SPACs (blank-cheque shell companies)", value=True,
                          help="SPACs list at $10 with no business yet; they are not normal company IPOs.")

    def clean(df):
        if df.empty:
            return df
        if hide_spac:
            name = df.get("companyName", pd.Series("", index=df.index)).fillna("")
            tick = df.get("proposedTickerSymbol", pd.Series("", index=df.index)).fillna("")
            df = df[~name.str.contains("Acquisition|Capital Corp|SPAC|Merger", case=False) & ~tick.str.endswith("U")]
        return df

    up, priced, filed = clean(d["upcoming"]), clean(d["priced"]), clean(d["filed"])
    st.markdown("#### Upcoming")
    if up.empty:
        st.caption("No upcoming US IPOs scheduled right now.")
    else:
        st.dataframe(up.rename(columns={"companyName": "Company", "proposedTickerSymbol": "Ticker", "proposedExchange": "Exchange",
                                        "proposedSharePrice": "Price range", "expectedPriceDate": "Expected date"})
                     [[c for c in ["Company", "Ticker", "Exchange", "Price range", "Expected date"] if c in up.rename(columns={
                         "companyName": "Company", "proposedTickerSymbol": "Ticker", "proposedExchange": "Exchange",
                         "proposedSharePrice": "Price range", "expectedPriceDate": "Expected date"}).columns]],
                     hide_index=True, use_container_width=True)

    st.markdown("#### Recently priced: how are they doing?")
    if priced.empty:
        st.caption("No recently priced IPOs.")
    else:
        pr = priced.copy()
        pr["ipo_price"] = pd.to_numeric(pr["proposedSharePrice"], errors="coerce")
        pr["date"] = pd.to_datetime(pr["pricedDate"], errors="coerce")
        pr = pr.dropna(subset=["ipo_price"]).sort_values("date", ascending=False)
        px = _us_perf(tuple(pr["proposedTickerSymbol"].dropna().unique()))
        closes = px["Close"] if not px.empty else pd.DataFrame()

        def now(t):
            return float(closes[t].dropna().iloc[-1]) if t in closes and closes[t].notna().any() else np.nan

        pr["price_now"] = pr["proposedTickerSymbol"].map(now)
        pr["vs_ipo"] = (pr["price_now"] / pr["ipo_price"] - 1) * 100
        tbl = pd.DataFrame({"Company": pr["companyName"], "Ticker": pr["proposedTickerSymbol"], "Priced on": pr["date"].dt.strftime("%d %b %Y"),
                            "IPO price": pr["ipo_price"].map(lambda v: f"${v:,.2f}"),
                            "Price now": pr["price_now"].map(lambda v: "–" if pd.isna(v) else f"${v:,.2f}"),
                            "Now vs IPO price": pr["vs_ipo"], "Raised": pr["dollarValueOfSharesOffered"]})
        ev = st.dataframe(tbl, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
                          column_config={"Now vs IPO price": st.column_config.NumberColumn(format="%+.0f%%")})
        if ev.selection.rows:
            r = pr.iloc[ev.selection.rows[0]]
            ss.market = "Global"
            open_stock(r["proposedTickerSymbol"], r["companyName"])
            st.rerun()

    with st.expander(f"Filed with the SEC, not yet priced ({len(filed)})"):
        if not filed.empty:
            st.dataframe(filed.rename(columns={"companyName": "Company", "proposedTickerSymbol": "Ticker",
                                               "filedDate": "Filed on", "dollarValueOfSharesOffered": "Planned raise"})
                         [[c for c in ["Company", "Ticker", "Filed on", "Planned raise"] if c in filed.rename(columns={
                             "companyName": "Company", "proposedTickerSymbol": "Ticker", "filedDate": "Filed on",
                             "dollarValueOfSharesOffered": "Planned raise"}).columns]], hide_index=True, use_container_width=True)
    st.caption("Source: Nasdaq IPO calendar and Yahoo Finance prices. Not investment advice.")
