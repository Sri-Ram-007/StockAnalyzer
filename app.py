"""Stock Analyzer: Indian & US stocks, ETFs, Indian mutual funds and worldwide search,
with drawdown/recovery analysis, business health checks and a recovery score.

Run with:  streamlit run app.py   (or double-click "Start Stock Analyzer.bat")
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data import (ALL_NSE, BSE_ONLY, get_all_nse, get_bse_only, get_ohlc, get_prices, get_universe, last_updated,
                       load_watchlist, save_watchlist)
from core.fundamentals import (DEFAULT_WEIGHTS, FUND_DIR, WEIGHT_LEVELS, fetch_many, get_fundamentals,
                               health_checks, recovery_score, score_label)
from core.funds import category_table, crash_recoveries, fund_metrics, get_fund_list, get_nav
from core.markets import ETF_TICKERS, currency_of, get_nse_etfs, get_us_all, global_search, is_etf as known_etf
import insurance_page
import ipo_page
from core.funds import sip_simulation
from core.news import get_news, news_query
from core.sectors import sector_table
import ml_page
import portfolio_page
from core.recovery import (YEAR, base_rates, biggest_one_day_drop, current_drawdown, history_threshold,
                           recovery_episodes, recovery_stats, scan)

st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

GREEN, RED, AMBER, MUTED = "#1a7f5a", "#d63a3a", "#b7791f", "#6b7280"

MARKETS = {
    "India": {"label": "🇮🇳 India", "lists": ["Nifty 50", "Nifty 100", "Nifty 200", "Nifty 500", ALL_NSE, BSE_ONLY, "Indian ETFs"],
              "default": "Nifty 100", "cur": "₹", "benchmark": "^NSEI",
              "indices": [("NIFTY 50", "^NSEI"), ("SENSEX", "^BSESN"), ("BANK NIFTY", "^NSEBANK")]},
    "US": {"label": "🇺🇸 US", "lists": ["S&P 500", "US ETFs"], "default": "S&P 500", "cur": "$", "benchmark": "^GSPC",
           "indices": [("S&P 500", "^GSPC"), ("NASDAQ", "^IXIC"), ("DOW JONES", "^DJI")]},
}
TOP = {"India": "India", "US": "US", "Funds": "Mutual Funds", "Global": "🌍 Global", "IPO": "🆕 IPOs", "Portfolio": "💼 Portfolio", "Insurance": "🛡️ Insurance"}

st.markdown(f"""
<style>
  #MainMenu, footer, [data-testid="stDecoration"] {{ display: none; }}
  .block-container {{ padding-top: 3.6rem; max-width: 1250px; }}  /* room for Streamlit Cloud's top bar */
  .brand {{ font-size: 1.45rem; font-weight: 700; color: #111827; padding-top: 2px; }}
  .idx-card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px 14px; background: #fff; }}
  .idx-name {{ font-size: .78rem; color: {MUTED}; font-weight: 600; letter-spacing: .02em; }}
  .idx-val {{ font-size: 1.05rem; font-weight: 600; color: #111827; }}
  .up {{ color: {GREEN}; }} .down {{ color: {RED}; }}
  .price-big {{ font-size: 2.1rem; font-weight: 700; color: #111827; line-height: 1.1; }}
  .chg {{ font-size: 1rem; font-weight: 600; }}
  .muted {{ color: {MUTED}; font-size: .9rem; }}
  .stat-label {{ color: {MUTED}; font-size: .8rem; }}
  .stat-val {{ font-weight: 600; font-size: 1rem; color: #111827; }}
  .insight {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px 18px; background: #fafbfc; }}
  .range-track {{ position: relative; height: 6px; background: #e5e7eb; border-radius: 4px; margin: 10px 0 4px; }}
  .range-dot {{ position: absolute; top: -5px; width: 16px; height: 16px; border-radius: 50%; background: #111827;
               border: 3px solid #fff; box-shadow: 0 0 0 1px #d1d5db; }}
  div[data-testid="stButton"] button[kind="tertiary"] {{ justify-content: flex-start; width: 100%; padding: 6px 4px; }}
  div[data-testid="stButton"] button[kind="tertiary"]:hover {{ background: #f3f4f6; }}
  [data-testid="stButtonGroup"] > div {{ flex-wrap: wrap; }}
  @media (max-width: 640px) {{
    .block-container {{ padding-left: .7rem; padding-right: .7rem; padding-top: 3.2rem; }}
    .brand {{ font-size: 1.2rem; }}
    .st-key-idxstrip [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important; gap: .35rem; }}
    .st-key-idxstrip [data-testid="stColumn"] {{ min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }}
    .st-key-idxstrip .idx-card {{ padding: 6px 7px; }}
    .st-key-idxstrip .idx-name {{ font-size: .62rem; }}
    .st-key-idxstrip .idx-val {{ font-size: .78rem; }}
    .st-key-idxstrip .chg {{ display: block; font-size: .7rem; }}
    .price-big {{ font-size: 1.7rem; }}
  }}
</style>""", unsafe_allow_html=True)


# ---------------- Cached loaders ----------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_universe(name):
    return get_universe(name)


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(name, tickers, token=0):
    return get_prices(name, list(tickers), force=token > 0)


@st.cache_data(ttl=3600, show_spinner=False)
def load_ohlc(ticker):
    return get_ohlc(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def load_scan(name, _closes, version):
    return scan(_closes, 0.0, 1.0)


@st.cache_data(ttl=3600, show_spinner=False)
def load_base_rates(name, _closes, version):
    cols = _closes.columns
    if len(cols) > 600:  # "All NSE": use the liquid Nifty 500 names for the market-wide base rate
        n500 = set(get_universe("Nifty 500")["Ticker"])
        cols = [c for c in cols if c in n500] or list(cols[:500])
    return base_rates(_closes[cols])


@st.cache_data(ttl=86400, show_spinner=False)
def load_all_nse():
    return get_all_nse()


@st.cache_data(ttl=86400, show_spinner=False)
def load_bse_only():
    return get_bse_only()


@st.cache_data(ttl=86400, show_spinner=False)
def load_nse_etfs():
    return get_nse_etfs()


@st.cache_data(ttl=86400, show_spinner=False)
def load_us_all():
    return get_us_all()


@st.cache_data(ttl=3600, show_spinner=False)
def load_news(query, india):
    return get_news(query, india)


@st.cache_data(ttl=86400, show_spinner=False)
def load_fund(ticker, _prices):
    return get_fundamentals(ticker, _prices)


@st.cache_data(ttl=43200, show_spinner=False)
def load_fund_list():
    return get_fund_list()


@st.cache_data(ttl=43200, show_spinner=False)
def load_nav(code):
    return get_nav(code)


@st.cache_data(ttl=43200, show_spinner=False)
def load_category(category, direct, growth):
    return category_table(load_fund_list(), category, direct, growth)


@st.cache_data(ttl=3600, show_spinner=False)
def load_search(q):
    return global_search(q)


@st.cache_data(ttl=86400, show_spinner=False)
def load_currency(t):
    return currency_of(t)


def money(x, cur="₹"):
    return "–" if x is None or pd.isna(x) else f"{cur}{x:,.2f}"


def pct(x, signed=False, digits=0):
    if x is None or pd.isna(x):
        return "–"
    return f"{x:+.{digits}%}" if signed else f"{x:.{digits}%}"


def cls(x):
    return "up" if x >= 0 else "down"


def lan_url():
    """Address of this app on the home Wi-Fi network, e.g. http://192.168.1.5:8501"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        return None
    return f"http://{ip}:{st.get_option('server.port')}"


def phone_qr():
    """Show a QR code for opening the app on a phone on the same Wi-Fi."""
    url = lan_url()
    if st.get_option("server.address") not in ("0.0.0.0", "::") or not url:
        st.caption("📱 To open this on your phone, close this window and start the app with "
                   "**Start for Phone (same Wi-Fi).bat** instead.")
        return
    import io
    import qrcode
    buf = io.BytesIO()
    qrcode.make(url, box_size=6, border=2).save(buf, format="PNG")
    st.markdown("**📱 Open on your phone**")
    st.image(buf.getvalue(), width=170)
    st.caption(f"Scan with your phone camera, or type **{url}** in its browser. The phone must be on the same Wi-Fi.")


def index_strip(indices):
    quotes = []
    for name, tk in indices:
        df = load_ohlc(tk)
        if len(df) >= 2:
            last, prev = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
            quotes.append((name, last, last - prev, last / prev - 1))
    if quotes:
        box = st.container(key="idxstrip")
        for c, (name, val, ch, p) in zip(box.columns(len(quotes)), quotes):
            c.markdown(f'<div class="idx-card"><div class="idx-name">{name}</div><div class="idx-val">{val:,.2f} '
                       f'<span class="chg {cls(p)}">{ch:+,.2f} ({p:+.2%})</span></div></div>', unsafe_allow_html=True)


# ---------------- State ----------------
ss = st.session_state
ss.setdefault("market", "India")
ss.setdefault("page", "Explore")
ss.setdefault("ticker", None)
ss.setdefault("refresh", 0)
ss.setdefault("lists", {m: MARKETS[m]["default"] for m in MARKETS})
ss.setdefault("watchlist", load_watchlist())
ss.setdefault("fund_code", None)
ss.setdefault("compare", [])
ss.setdefault("extra_names", {})


def open_stock(t, name=None):
    if name:
        ss.extra_names[t] = name
    ss.ticker, ss.page = t, "Stock"


def open_fund(code):
    ss.fund_code, ss.page = int(code), "Fund"


# ---------------- Header ----------------
h1, h2 = st.columns([2, 5])
h1.markdown('<div class="brand">📈 Stock Analyzer</div>', unsafe_allow_html=True)
mk = h2.segmented_control("Market", list(TOP), default=ss.market, format_func=lambda k: TOP[k],
                          key=f"mk_{ss.market}", label_visibility="collapsed")
if mk and mk != ss.market:
    ss.market = mk
    ss.page = {"Funds": "Explore", "Global": "Search", "Portfolio": "Holdings", "IPO": "Open & upcoming", "Insurance": "Health insurers"}.get(mk, "Explore")
    st.rerun()

market = ss.market
cfg = MARKETS.get(market, MARKETS["US"] if market == "Global" else MARKETS["India"])

# ================================================================
#                     STOCK MARKETS (India / US / Global)
# ================================================================
if market in ("India", "US", "Global"):
    list_market = market if market in MARKETS else "US"
    list_name = ss.lists[list_market]
    try:
        universe = load_universe(list_name)
        with st.spinner(f"Loading prices for {len(universe)} {'funds' if 'ETF' in list_name else 'stocks'} "
                        "(first time can take a few minutes)…"):
            closes = load_prices(list_name, tuple(universe["Ticker"]), ss.refresh)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()
    closes = closes.loc[:, closes.notna().sum() > 30]
    version = f"{closes.index.max()}-{closes.shape}"
    names = universe.set_index("Ticker")["Company Name"].to_dict()
    industries = universe.set_index("Ticker")["Industry"].to_dict()
    if market == "India":
        all_nse = load_all_nse()
        bse_only = load_bse_only()
        names = {**bse_only.set_index("Ticker")["Company Name"].to_dict(),
                 **all_nse.set_index("Ticker")["Company Name"].to_dict(), **names}
        industries = {**bse_only.set_index("Ticker")["Industry"].to_dict(),
                      **all_nse.set_index("Ticker")["Industry"].to_dict(), **industries}
        nse_etfs = load_nse_etfs()
        names = {**nse_etfs.set_index("Ticker")["Company Name"].to_dict(), **names}
        extra_search = set(all_nse["Ticker"]) | set(bse_only["Ticker"]) | set(nse_etfs["Ticker"])
    elif market == "US":
        us_all = load_us_all()
        names = {**us_all.set_index("Ticker")["Company Name"].to_dict(), **names}
        industries = {**us_all.set_index("Ticker")["Industry"].to_dict(), **industries}
        extra_search = set(us_all["Ticker"])
    else:
        extra_search = set()
    names = {**names, **ss.extra_names}
    is_etf_list = "ETF" in list_name
    cur = cfg["cur"]

    # Search + settings row
    s1, s2 = st.columns([5, 1.3])
    if market == "Global":
        q = s1.text_input("Search", placeholder="🔍  Search any stock or ETF worldwide, e.g. Toyota, Samsung, Nestle, VOO",
                          label_visibility="collapsed")
    else:
        search_list = sorted(set(closes.columns) | extra_search,
                             key=lambda t: names.get(t, t))
        pick = s1.selectbox("Search", search_list, index=None, label_visibility="collapsed",
                            placeholder=f"🔍  Search {len(search_list):,} {'NSE & BSE' if market == 'India' else 'US'} stocks and ETFs",
                            format_func=lambda t: f"{names.get(t, t)}  ·  {t.replace('.NS', '').replace('.BO', ' (BSE)')}")
        if pick and pick != ss.get(f"last_pick_{market}"):
            ss[f"last_pick_{market}"] = pick
            open_stock(pick)
    with s2.popover("⚙️ Settings", use_container_width=True):
        new_l = st.selectbox("List for Explore and Fallen", cfg["lists"], index=cfg["lists"].index(list_name),
                             help="'All NSE stocks' scans ~2,350 companies: the first load takes about 10 minutes, then it is cached.")
        if new_l != list_name:
            ss.lists[list_market] = new_l
            st.rerun()
        if st.button("Refresh prices now"):
            ss.refresh += 1
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Prices updated {last_updated(list_name)}. Latest trading day {closes.index.max():%d %b %Y}.")
        st.divider()
        phone_qr()

    index_strip(cfg["indices"])
    st.write("")

    pages = ["Explore", "Fallen", "Sectors", "ML", "Watchlist"] if market != "Global" else ["Search", "Watchlist"]
    if ss.ticker and ss.page == "Stock":
        pages.append("Stock")
    nav = st.segmented_control("Menu", pages, default=ss.page if ss.page in pages else pages[0],
                               key=f"nav_{market}_{ss.page}_{ss.ticker}", label_visibility="collapsed",
                               format_func=lambda p: {"Fallen": "Fallen from high", "ML": "🤖 ML outlook"}.get(p, p))
    if nav and nav != ss.page:
        ss.page = nav
        st.rerun()
    if ss.page not in pages:
        ss.page = pages[0]

    last2 = closes.ffill().iloc[-2:]
    day = pd.DataFrame({"price": last2.iloc[-1], "chg": last2.iloc[-1] / last2.iloc[-2] - 1}).dropna()
    scan_all = load_scan(list_name, closes, version).set_index("Ticker")

    def stock_rows(tickers, extra=None, key=""):
        for t in tickers:
            if t not in day.index:
                continue
            r = day.loc[t]
            val = extra(t) if extra else pct(r["chg"], True, 2)
            color = ("green" if r["chg"] >= 0 else "red") if extra is None else "gray"
            st.button(f"**{names.get(t, t)}**  ·  {money(r['price'], cur)}  ·  :{color}[{val}]",
                      key=f"{key}_{t}", type="tertiary", on_click=open_stock, args=(t,))

    # ---------- Explore ----------
    if ss.page == "Explore":
        a, b = st.columns(2)
        with a.container(border=True):
            st.markdown("**Top gainers today**")
            stock_rows(day.sort_values("chg", ascending=False).index[:6], key="gain")
        with b.container(border=True):
            st.markdown("**Top losers today**")
            stock_rows(day.sort_values("chg").index[:6], key="lose")
        c, d = st.columns(2)
        with c.container(border=True):
            st.markdown("**Biggest falls from 1-year high**")
            clean = scan_all[~scan_all["corporate_action_flag"]].sort_values("drawdown")
            stock_rows(clean.index[:6], extra=lambda t: f"{scan_all.loc[t, 'drawdown']:.0%} from high", key="fall")
        with d.container(border=True):
            st.markdown("**Near 1-year high**")
            stock_rows(scan_all.sort_values("drawdown", ascending=False).index[:6],
                       extra=lambda t: f"{scan_all.loc[t, 'drawdown']:.0%} from high", key="near")
        st.caption(f"Showing: {list_name}. Change the list in ⚙️ Settings. Click any row to open it. Not investment advice.")

    # ---------- Global search ----------
    elif ss.page == "Search":
        if q:
            results = load_search(q)
            if not results:
                st.info("No stocks or ETFs found. Try another name or a ticker symbol.")
            with st.container(border=True):
                for r in results:
                    st.button(f"**{r['name']}**  ·  {r['ticker']}  ·  :gray[{r['exchange']} · {r['type'].title()}]",
                              key=f"g_{r['ticker']}", type="tertiary", on_click=open_stock, args=(r["ticker"], r["name"]))
        else:
            st.info("Type a company name above. Works for stocks and ETFs on most exchanges: US, UK, Europe, Japan, "
                    "Korea, Hong Kong and more. Indian stocks are under India.")

    # ---------- Fallen ----------
    elif ss.page == "Fallen":
        st.subheader("Fallen from high: will they come back?")
        st.write("Trading well below the highest price of the last year, with how often each one climbed back "
                 "to that high after a similar fall in the past.")
        f1, f2 = st.columns([3, 2])
        level = f1.segmented_control("How far below the 1-year high", ["Down 10%+", "Down 20%+", "Down 30%+", "Down 40%+", "Down 50%+"],
                                     default="Down 20%+" if is_etf_list else "Down 30%+")
        strong = f2.toggle("Only with 3 or more past falls (stronger evidence)", value=False)
        min_dd = int((level or "Down 30%+").split()[1].rstrip("%+")) / 100
        rates = load_base_rates(list_name, closes, version).set_index("threshold")["recovery_rate"].to_dict()
        res = scan_all[(scan_all["drawdown"] <= -min_dd) & (~scan_all["corporate_action_flag"])].copy()
        if strong:
            res = res[res["episodes"] >= 3]
        res = res.sort_values("drawdown")

        if res.empty:
            st.info("Nothing matches. Try a smaller fall.")
        else:
            funds = {}
            if not is_etf_list:
                cached = {t for t in res.index if (FUND_DIR / f"{t.replace('&', '_')}.json").exists()}
                missing = [t for t in res.index if t not in cached]
                if missing:
                    b1, b2 = st.columns([2, 3])
                    if b1.button(f"🩺 Check business health for {len(missing)} stocks", type="primary"):
                        bar = st.progress(0.0, text="Fetching company financials…")
                        fetch_many(missing, closes, progress=lambda p: bar.progress(p, text=f"Fetching company financials… {p:.0%}"))
                        bar.empty()
                        st.rerun()
                    b2.caption(f"Adds a business health check and recovery score. About {max(5, len(missing) // 4)} "
                               "seconds the first time, then saved for 7 days.")
                funds = fetch_many(sorted(cached), closes) if cached else {}
            bench = load_ohlc(cfg["benchmark"])
            market_dd = current_drawdown(bench["Close"]).get("drawdown") if not bench.empty else None

            def peer_dd(t):
                ind = industries.get(t)
                peers = [p for p in scan_all.index if industries.get(p) == ind and p != t] if ind else []
                return float(scan_all.loc[peers, "drawdown"].median()) if len(peers) >= 3 else None

            health_col, score_col = [], []
            for t, r in res.iterrows():
                f = funds.get(t)
                if not f or not f.get("ok"):
                    health_col.append("–"), score_col.append(None)
                    continue
                ch = health_checks(f, r["drawdown"], peer_dd(t), market_dd)
                health_col.append(f"{sum(c['status'] == 'pass' for c in ch)} of {sum(c['status'] != 'na' for c in ch)} good")
                score_col.append(recovery_score(r["recovery_rate"], r["episodes"], rates.get(r["history_threshold"]), ch,
                                                ss.get("weights"))["score"])

            def history_text(r):
                if r["episodes"] == 0:
                    return "Never fell this much before"
                return f"Recovered {int(round(r['recovery_rate'] * r['episodes']))} of {int(r['episodes'])} times"

            table = pd.DataFrame({
                "Name": res.index.map(lambda t: names.get(t, t)),
                "Score": score_col,
                "Price": res["price"].map(lambda v: money(v, cur)),
                "Down from 1-yr high": res["drawdown"] * 100,
                "Needs to rise": res["gain_to_recover"] * 100,
                "Its own history": res.apply(history_text, axis=1),
                "Typical time to recover": (res["median_days"] / 30.4).map(lambda m: "–" if pd.isna(m) else f"{m:.0f} months"),
                "All in list, same fall": res["history_threshold"].map(rates),
                "Business health": health_col,
            })
            if is_etf_list:
                table = table.drop(columns=["Score", "Business health"])
            st.caption(f"{len(table)} results. Click a row to open it.")
            event = st.dataframe(table, hide_index=True, use_container_width=True, height=560,
                                 on_select="rerun", selection_mode="single-row", column_config={
                                     "Score": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100,
                                         help="Recovery score out of 100. Open a stock to see the breakdown."),
                                     "Down from 1-yr high": st.column_config.NumberColumn(format="%.0f%%"),
                                     "Needs to rise": st.column_config.NumberColumn(format="+%.0f%%",
                                         help="Going from 1,200 back to 1,800 needs +50%, not +33%."),
                                     "All in list, same fall": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1,
                                         help="Across everything in this list, how often a fall this big got back to the old high within 2 years."),
                                 })
            if event.selection.rows:
                open_stock(res.index[event.selection.rows[0]])
                st.rerun()
        st.caption("'Recovered' = got back to its old 1-year high within 2 years. Possible splits/demergers hidden. Not investment advice.")

    # ---------- Sectors ----------
    elif ss.page == "Sectors":
        st.subheader("Sectors: what's rising, what's fallen most")
        sec_list = "Nifty 500" if market == "India" else "S&P 500"
        su = load_universe(sec_list)
        with st.spinner(f"Loading {sec_list} prices (first time can take a few minutes)…"):
            sc = load_prices(sec_list, tuple(su["Ticker"]), ss.refresh)
        sc = sc.loc[:, sc.notna().sum() > 30]
        sind = su.set_index("Ticker")["Industry"].to_dict()
        snames = su.set_index("Ticker")["Company Name"].to_dict()
        tbl, per_stock = sector_table(sc, sind)
        sort = st.segmented_control("Sort by", ["3 months", "1 month", "1 year", "Fallen most from high", "Today"],
                                    default="3 months")
        col = {"3 months": "3m", "1 month": "1m", "1 year": "1y", "Fallen most from high": "below_high",
               "Today": "today"}[sort or "3 months"]
        tbl = tbl.sort_values(col, ascending=(col == "below_high"))
        fig = go.Figure(go.Bar(x=tbl[col] * 100, y=tbl.index, orientation="h",
                               marker_color=[GREEN if v >= 0 else RED for v in tbl[col]]))
        fig.update_layout(height=max(320, 26 * len(tbl)), margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white",
                          yaxis=dict(autorange="reversed"),
                          xaxis=dict(title=f"Median {sort or '3 months'} (%)", gridcolor="#f0f0f0"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        show = pd.DataFrame({"Sector": tbl.index, "Stocks": tbl["stocks"].values,
                             "Today": tbl["today"].values * 100, "1 month": tbl["1m"].values * 100,
                             "3 months": tbl["3m"].values * 100, "1 year": tbl["1y"].values * 100,
                             "Below 1-yr high": tbl["below_high"].values * 100,
                             "Above 200-day avg": tbl["above_200dma"].values})
        pc = st.column_config.NumberColumn(format="%+.1f%%")
        st.caption(f"Each figure is the median of the sector's stocks in the {sec_list}. Click a sector to see its stocks.")
        ev = st.dataframe(show, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
                          column_config={"Today": pc, "1 month": pc, "3 months": pc, "1 year": pc,
                                         "Below 1-yr high": st.column_config.NumberColumn(format="%.0f%%"),
                                         "Above 200-day avg": st.column_config.ProgressColumn(
                                             format="percent", min_value=0, max_value=1,
                                             help="Share of the sector's stocks above their 200-day average (an uptrend sign).")})
        if ev.selection.rows:
            ss.sector = show.iloc[ev.selection.rows[0]]["Sector"]
        if ss.get("sector") in tbl.index:
            st.markdown(f"#### {ss.sector}")
            ps = per_stock[per_stock["sector"] == ss.sector].sort_values("below_high")
            stbl = pd.DataFrame({"Name": [snames.get(x, x) for x in ps.index], "Ticker": ps.index,
                                 "Price": ps["price"].map(lambda v: money(v, cur)),
                                 "1 month": ps["ret_1m"] * 100, "3 months": ps["ret_3m"] * 100,
                                 "1 year": ps["ret_1y"] * 100, "Below 1-yr high": ps["below_high"] * 100})
            ev2 = st.dataframe(stbl, hide_index=True, use_container_width=True, on_select="rerun",
                               selection_mode="single-row",
                               column_config={"1 month": pc, "3 months": pc, "1 year": pc,
                                              "Below 1-yr high": st.column_config.NumberColumn(format="%.0f%%")})
            if ev2.selection.rows:
                t_sel = stbl.iloc[ev2.selection.rows[0]]["Ticker"]
                open_stock(t_sel, snames.get(t_sel))
                st.rerun()
        st.caption("Not investment advice.")

    # ---------- ML outlook ----------
    elif ss.page == "ML":
        bench_close = load_ohlc(cfg["benchmark"])["Close"]
        ml_page.render(list_name, closes, bench_close, cfg["indices"][0][0].title().replace("Nifty", "Nifty").replace("S&P", "S&P"),
                       names, open_stock)

    # ---------- Watchlist ----------
    elif ss.page == "Watchlist":
        st.subheader("Your watchlist")
        if not ss.watchlist:
            st.info("Your watchlist is empty. Open any stock or ETF and tap ☆ Add to watchlist.")
        else:
            with st.container(border=True):
                for t in ss.watchlist:
                    o = load_ohlc(t)
                    if len(o) < 2:
                        continue
                    p, pv = float(o["Close"].iloc[-1]), float(o["Close"].iloc[-2])
                    c = load_currency(t)
                    st.button(f"**{names.get(t, ss.extra_names.get(t, t))}**  ·  {money(p, c)}  ·  "
                              f":{'green' if p >= pv else 'red'}[{pct(p / pv - 1, True, 2)}]",
                              key=f"w_{t}", type="tertiary", on_click=open_stock, args=(t,))

    # ---------- Stock page ----------
    elif ss.page == "Stock" and ss.ticker:
        t = ss.ticker
        ohlc = load_ohlc(t)
        if ohlc.empty and t not in closes:
            st.error("No price data found for this ticker on Yahoo Finance.")
            st.stop()
        s = ohlc["Close"].dropna() if not ohlc.empty else closes[t].dropna()
        scur = load_currency(t)
        now = current_drawdown(s)
        price, prev = float(s.iloc[-1]), float(s.iloc[-2])
        chg = price / prev - 1
        is_etf = known_etf(t, names.get(t, ""))

        top_l, top_r = st.columns([5, 1.4])
        top_l.markdown(f"### {names.get(t, t)}")
        top_l.markdown(f'<span class="muted">{t} · {"ETF" if is_etf else industries.get(t, "")}</span>', unsafe_allow_html=True)
        in_list = t in ss.watchlist
        if top_r.button("★ In watchlist" if in_list else "☆ Add to watchlist", use_container_width=True):
            ss.watchlist = [x for x in ss.watchlist if x != t] if in_list else ss.watchlist + [t]
            save_watchlist(ss.watchlist)
            st.rerun()
        st.markdown(f'<div class="price-big">{money(price, scur)}</div>'
                    f'<div class="chg {cls(chg)}">{price - prev:+,.2f} ({chg:+.2%}) on {s.index[-1]:%d %b}</div>',
                    unsafe_allow_html=True)

        period = st.segmented_control("Period", ["1M", "6M", "1Y", "3Y", "5Y", "MAX"], default="1Y", label_visibility="collapsed")
        days = {"1M": 31, "6M": 183, "1Y": 366, "3Y": 1098, "5Y": 1830, "MAX": 36600}[period or "1Y"]
        data = ohlc if not ohlc.empty else s.to_frame("Close")
        data = data[data.index >= data.index.max() - pd.Timedelta(days=days)]
        up_period = data["Close"].iloc[-1] >= data["Close"].iloc[0]
        fig = go.Figure()
        if {"Open", "High", "Low"} <= set(data.columns) and days <= 366:
            fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
                                         increasing_line_color=GREEN, decreasing_line_color=RED, name="Price"))
        else:
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], line=dict(color=GREEN if up_period else RED, width=2),
                                     fill="tozeroy", fillcolor="rgba(26,127,90,0.08)" if up_period else "rgba(214,58,58,0.08)"))
        fig.add_hline(y=now["high"], line_dash="dot", line_color="#9ca3af",
                      annotation_text=f"1-yr closing high {money(now['high'], scur)}", annotation_position="top left")
        lo = data["Low"].min() if "Low" in data else data["Close"].min()
        hi = max(data["High"].max() if "High" in data else data["Close"].max(), now["high"])
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, showlegend=False,
                          plot_bgcolor="white", hovermode="x unified",
                          yaxis=dict(range=[lo * 0.97, hi * 1.03], gridcolor="#f0f0f0", side="right"), xaxis=dict(gridcolor="#f7f7f7"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        yr = ohlc[ohlc.index >= ohlc.index.max() - pd.Timedelta(days=366)] if not ohlc.empty else s.to_frame("Close")
        low52 = float(yr["Low"].min() if "Low" in yr else yr["Close"].min())
        high52 = float(yr["High"].max() if "High" in yr else yr["Close"].max())
        bar = ohlc.iloc[-1] if not ohlc.empty else None
        stats = [("Open", money(bar["Open"], scur) if bar is not None else "–"),
                 ("Day high", money(bar["High"], scur) if bar is not None else "–"),
                 ("Day low", money(bar["Low"], scur) if bar is not None else "–"),
                 ("Prev. close", money(prev, scur)),
                 ("200-day average", money(s.rolling(200).mean().iloc[-1], scur)),
                 ("Down from 1-yr closing high", pct(now["drawdown"], digits=1))]
        for c, (lab, val) in zip(st.columns(len(stats)), stats):
            c.markdown(f'<div class="stat-label">{lab}</div><div class="stat-val">{val}</div>', unsafe_allow_html=True)
        pos = 0 if high52 == low52 else (price - low52) / (high52 - low52)
        st.markdown(f"""<div style="margin-top:14px" class="stat-label">52-week range (intraday)</div>
            <div class="range-track"><div class="range-dot" style="left: calc({pos * 100:.1f}% - 8px)"></div></div>
            <div style="display:flex; justify-content:space-between" class="muted"><span>Low {money(low52, scur)}</span>
            <span>High {money(high52, scur)}</span></div>""", unsafe_allow_html=True)

        fund = {}
        if not is_etf:
            with st.spinner("Loading company financials…"):
                fund = load_fund(t, s)
            if fund.get("ok"):
                mc, fcur = fund.get("marketCap"), scur
                mc_txt = (f"₹{mc / 1e7:,.0f} Cr" if scur == "₹" else f"{fcur}{mc / 1e9:,.1f} B") if mc else "–"
                snap = [("Market cap", mc_txt),
                        ("P/E ratio", f"{fund['trailingPE']:.1f}" if fund.get("trailingPE") else "–"),
                        ("Price / book", f"{fund['priceToBook']:.1f}" if fund.get("priceToBook") else "–"),
                        ("Dividend yield", f"{fund['dividendYield']:.2f}%" if fund.get("dividendYield") is not None else "–"),
                        ("Profit margin", pct(fund.get("profitMargins"))),
                        ("Sector", fund.get("sector") or "–")]
                st.write("")
                for c, (lab, val) in zip(st.columns(len(snap)), snap):
                    c.markdown(f'<div class="stat-label">{lab}</div><div class="stat-val">{val}</div>', unsafe_allow_html=True)

        # Recovery insight
        st.write("")
        st.markdown("#### Recovery insight")
        stt, thr, rates = None, None, {}
        if now["drawdown"] > -0.10:
            st.markdown(f'<div class="insight">It is <b>{abs(now["drawdown"]):.0%} below</b> its 1-year closing high of '
                        f'<b>{money(now["high"], scur)}</b>, so it is trading close to its high. The recovery analysis is for '
                        f'prices that have fallen 10% or more.</div>', unsafe_allow_html=True)
        else:
            thr = history_threshold(now["drawdown"])
            eps = recovery_episodes(s, thr)
            stt = recovery_stats(eps)
            rates = load_base_rates(list_name, closes, version).set_index("threshold")["recovery_rate"].to_dict()
            if stt["episodes"] == 0:
                hist = (f"In the last {len(s) / YEAR:.0f} years it has <b>never fallen {thr:.0%} or more</b> before, "
                        "so there is no past example to learn from.")
            else:
                rec = int(round(stt["recovery_rate"] * stt["episodes"]))
                word = "it took about" if rec == 1 else "typically in about"
                timing = f", {word} <b>{stt['median_days'] / 30.4:.0f} months</b>" if rec else ""
                hist = (f"In the last {len(s) / YEAR:.0f} years it fell {thr:.0%} or more <b>{stt['episodes']} time"
                        f"{'s' if stt['episodes'] != 1 else ''}</b> and got back to its old high <b>{rec} of those times</b>{timing}.")
            base = rates.get(thr)
            base_txt = (f" Across the {len(closes.columns)} names in {list_name}, falls this big recovered <b>{base:.0%}</b> "
                        "of the time within 2 years.") if base is not None and not pd.isna(base) else ""
            warn = ('<br><br>⚠️ <b>Check corporate actions:</b> a one-day fall of 30%+ in the last year usually means a '
                    'demerger or split, so this fall may not be real.') if biggest_one_day_drop(s) <= -0.30 else ""
            st.markdown(f"""<div class="insight">It is <b>{abs(now['drawdown']):.0%} below</b> its 1-year closing high of
                <b>{money(now['high'], scur)}</b> (reached {now['high_date']:%d %b %Y}). To get back there it needs to rise
                <b>{now['gain_to_recover']:.0%}</b>.<br><br>{hist}{base_txt}{warn}
                <br><br><span class="muted">Past recoveries are evidence, not a promise.</span></div>""", unsafe_allow_html=True)
            if not eps.empty:
                with st.expander("See every past fall"):
                    st.dataframe(pd.DataFrame({
                        "Fall started": eps["start"].dt.strftime("%d %b %Y"),
                        "Price then": eps["price_at_start"].map(lambda v: money(v, scur)),
                        "Old high": eps["old_high"].map(lambda v: money(v, scur)),
                        "Got back to old high?": eps["recovered"].map({True: "Yes", False: "No (within 2 yrs)"}),
                        "Time taken": (eps["days_to_recover"] / 30.4).map(lambda m: "–" if pd.isna(m) else f"{m:.0f} months"),
                        "Fell further first": eps["further_drop"].map(lambda v: f"{v:.0%}"),
                    }), hide_index=True, use_container_width=True)

        # Business health + score
        checks = []
        if is_etf:
            st.info("This is an ETF: it holds a basket of stocks or assets, so company health checks don't apply. "
                    "Use the recovery insight above and compare it with similar ETFs.")
        else:
            st.write("")
            st.markdown("#### Business health check")
            ref_market = cfg["benchmark"] if market != "Global" else None
            bench = load_ohlc(ref_market) if ref_market else pd.DataFrame()
            market_dd = current_drawdown(bench["Close"]).get("drawdown") if not bench.empty else None
            ind = industries.get(t)
            peers = [p for p in scan_all.index if industries.get(p) == ind and p != t] if ind else []
            peer_dd = float(scan_all.loc[peers, "drawdown"].median()) if len(peers) >= 3 else None
            if not fund.get("ok"):
                st.info("Company financials are not available on Yahoo Finance for this stock.")
            else:
                checks = health_checks(fund, now["drawdown"] if now["drawdown"] <= -0.10 else None, peer_dd, market_dd)
                icons = {"pass": "✅", "warn": "⚠️", "fail": "❌", "na": "➖"}
                st.markdown('<div class="insight">' + "".join(
                    f'<div style="padding:7px 0; border-bottom:1px solid #eef0f2">{icons[c["status"]]} <b>{c["check"]}</b>'
                    f'<span class="muted"> · {c["detail"]}</span></div>' for c in checks) + "</div>", unsafe_allow_html=True)
                st.caption("✅ good · ⚠️ watch · ❌ warning sign · ➖ not available. From Yahoo Finance; check the latest results too.")

        if checks and stt is not None:
            st.write("")
            st.markdown("#### Recovery score")
            weights = ss.get("weights", DEFAULT_WEIGHTS)
            res_sc = recovery_score(stt["recovery_rate"], stt["episodes"], rates.get(thr), checks, weights)
            score = res_sc["score"]
            colour = GREEN if (score or 0) >= 70 else AMBER if (score or 0) >= 45 else RED
            left, right = st.columns([1, 2])
            left.markdown(f'<div class="insight" style="text-align:center"><div class="stat-label">Recovery score</div>'
                          f'<div style="font-size:3rem; font-weight:700; color:{colour}">{score if score is not None else "–"}</div>'
                          f'<div class="stat-val" style="color:{colour}">{score_label(score)}</div><div class="muted">out of 100</div></div>',
                          unsafe_allow_html=True)
            with right:
                for part, val in res_sc["parts"].items():
                    st.progress(float(val), text=f"{part}: {val:.0%}  ·  weight {weights.get(part, 'Medium')}")
                with st.popover("⚖️ Change what matters to you"):
                    new_w = {p: st.select_slider(p, options=list(WEIGHT_LEVELS), value=weights.get(p, "Medium"), key=f"w_{p}")
                             for p in DEFAULT_WEIGHTS}
                    if new_w != weights:
                        ss.weights = new_w
                        st.rerun()
            st.caption("70+ Strong · 45–69 Mixed · below 45 Weak. The score summarises evidence; it is not a buy or sell signal.")
        st.write("")
        ml_page.stock_box(list_name, t)

        st.write("")
        st.markdown("#### Latest news")
        nq, india_news = news_query(names.get(t, t), t)
        headlines = load_news(nq, india_news)
        if not headlines:
            st.caption("No recent headlines found.")
        else:
            st.markdown('<div class="insight">' + "".join(
                f'<div style="padding:7px 0; border-bottom:1px solid #eef0f2"><a href="{h["link"]}" target="_blank" '
                f'style="color:#111827; text-decoration:none; font-weight:600">{h["title"]}</a><br>'
                f'<span class="muted">{h["source"]} · {pd.Timestamp(h["date"]):%d %b %Y, %H:%M}</span></div>'
                for h in headlines) + "</div>", unsafe_allow_html=True)
            st.caption("Headlines from Google News. Click to read on the publisher's site.")
        st.caption("Prices from Yahoo Finance, adjusted for splits and dividends. Educational tool, not investment advice.")

# ================================================================
#                          MUTUAL FUNDS
# ================================================================
elif market == "Funds":
    try:
        fl = load_fund_list()
    except Exception as e:
        st.error(f"Could not load the mutual fund list: {e}")
        st.stop()
    index_strip(MARKETS["India"]["indices"])
    st.write("")

    s1, _ = st.columns([5, 1.3])
    dg = fl[fl["is_direct"] & fl["is_growth"]]
    lookup = dict(zip(fl["code"], fl["display"]))
    pick = s1.selectbox("Search funds", dg["code"].tolist() + fl[~fl.index.isin(dg.index)]["code"].tolist(), index=None,
                        placeholder=f"🔍  Search {len(fl):,} mutual funds, e.g. Parag Parikh Flexi Cap, HDFC Mid Cap",
                        format_func=lambda c: lookup.get(c, str(c)), label_visibility="collapsed")
    if pick and pick != ss.get("last_fund_pick"):
        ss.last_fund_pick = pick
        open_fund(pick)

    pages = ["Explore", "Compare"] + (["Fund"] if ss.fund_code else [])
    nav = st.segmented_control("Menu", pages, default=ss.page if ss.page in pages else "Explore",
                               key=f"fnav_{ss.page}_{ss.fund_code}", label_visibility="collapsed",
                               format_func=lambda p: {"Explore": "Explore categories", "Compare": f"Compare ({len(ss.compare)})"}.get(p, p))
    if nav and nav != ss.page:
        ss.page = nav
        st.rerun()
    if ss.page not in pages:
        ss.page = "Explore"

    def fpct(v):
        return np.nan if v is None or pd.isna(v) else v * 100

    def fret(v):
        return "–" if v is None or pd.isna(v) else f"{v:+.1%}"

    # ---------- Explore categories ----------
    if ss.page == "Explore":
        st.subheader("Find the best funds in a category")
        cats = sorted(fl["category"].dropna().unique())
        default_cat = "Equity Scheme - Flexi Cap Fund"
        c1, c2, c3 = st.columns([3, 1, 1])
        cat = c1.selectbox("Category", cats, index=cats.index(default_cat) if default_cat in cats else 0)
        direct = c2.toggle("Direct plans only", value=True, help="Direct plans have lower costs than Regular plans.")
        growth = c3.toggle("Growth option only", value=True, help="Growth reinvests profits; IDCW pays them out.")
        sort_by = st.segmented_control("Sort by", ["3Y return", "5Y return", "1Y return", "Return per risk", "Most consistent", "Smallest worst fall"],
                                       default="3Y return")
        with st.spinner("Loading NAV history for every fund in this category (first time about 10–30 seconds)…"):
            ct = load_category(cat, direct, growth)
        if ct.empty:
            st.info("No funds with enough history in this category.")
        else:
            key = {"3Y return": ("3Y", False), "5Y return": ("5Y", False), "1Y return": ("1Y", False),
                   "Return per risk": ("return_per_risk", False), "Most consistent": ("rolling3_pct_above_10", False),
                   "Smallest worst fall": ("max_drawdown", False)}[sort_by or "3Y return"]
            ct = ct.sort_values(key[0], ascending=key[1], na_position="last").reset_index(drop=True)
            table = pd.DataFrame({
                "Fund": ct["name"], "Fund house": ct["fund_house"],
                "1Y": ct["1Y"].map(fret), "3Y": ct["3Y"].map(fret), "5Y": ct["5Y"].map(fret), "10Y": ct["10Y"].map(fret),
                "Ups & downs (volatility)": ct["volatility"].map(fpct),
                "Worst fall ever": ct["max_drawdown"].map(fpct),
                "Below its high now": ct["current_drawdown"].map(fpct),
                "3Y periods beating 10%": ct["rolling3_pct_above_10"],
            })
            pc = st.column_config.NumberColumn(format="%.1f%%")
            st.caption(f"{len(table)} funds. Returns are yearly (CAGR). Click a row to open the fund.")
            ev = st.dataframe(table, hide_index=True, use_container_width=True, height=560, on_select="rerun",
                              selection_mode="single-row", column_config={
                                  "Ups & downs (volatility)": st.column_config.NumberColumn(format="%.1f%%",
                                      help="How much the NAV swings in a typical year. Lower = smoother ride."),
                                  "Worst fall ever": st.column_config.NumberColumn(format="%.0f%%",
                                      help="The biggest drop from a high to a low in the fund's history."),
                                  "Below its high now": st.column_config.NumberColumn(format="%.0f%%"),
                                  "3Y periods beating 10%": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1,
                                      help="Of every 3-year holding period in the fund's history, how many earned more than 10% a year. Higher = more consistent."),
                              })
            if ev.selection.rows:
                open_fund(ct.loc[ev.selection.rows[0], "code"])
                st.rerun()
        st.caption("NAV data from AMFI and mfapi.in. Past returns do not guarantee future returns. Not investment advice.")

    # ---------- Fund page ----------
    elif ss.page == "Fund" and ss.fund_code:
        code = ss.fund_code
        row = fl[fl["code"] == code]
        nav_s = load_nav(code)
        if nav_s.empty:
            st.error("NAV history not available for this fund.")
            st.stop()
        info = row.iloc[0] if not row.empty else None
        m = fund_metrics(nav_s)
        top_l, top_r = st.columns([5, 1.6])
        top_l.markdown(f"### {info['name'] if info is not None else code}")
        if info is not None:
            top_l.markdown(f'<span class="muted">{info["fund_house"]} · {info["category"]} · {info["plan"]} · {info["option"]}</span>',
                           unsafe_allow_html=True)
        in_cmp = code in ss.compare
        if top_r.button("✓ In compare" if in_cmp else "+ Add to compare", use_container_width=True,
                        disabled=(not in_cmp and len(ss.compare) >= 4)):
            ss.compare = [c for c in ss.compare if c != code] if in_cmp else ss.compare + [code]
            st.rerun()
        d1 = nav_s.iloc[-1] / nav_s.iloc[-2] - 1
        st.markdown(f'<div class="price-big">₹{m["nav"]:,.4f}</div><div class="chg {cls(d1)}">{d1:+.2%} on {m["nav_date"]:%d %b %Y} '
                    f'<span class="muted">(NAV)</span></div>', unsafe_allow_html=True)

        period = st.segmented_control("Period", ["1Y", "3Y", "5Y", "10Y", "MAX"], default="5Y", label_visibility="collapsed")
        yrs = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "MAX": 100}[period or "5Y"]
        part = nav_s[nav_s.index >= nav_s.index[-1] - pd.DateOffset(years=yrs)]
        growth_fund = 10000 * part / part.iloc[0]
        bench = load_ohlc("^NSEI")["Close"]
        bench = bench[bench.index >= part.index[0]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=growth_fund.index, y=growth_fund, name="This fund", line=dict(color=GREEN, width=2)))
        if len(bench) > 5:
            fig.add_trace(go.Scatter(x=bench.index, y=10000 * bench / bench.iloc[0], name="Nifty 50 (price index)",
                                     line=dict(color="#9ca3af", width=1.5, dash="dot")))
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="white", hovermode="x unified",
                          title=dict(text="Growth of ₹10,000", font=dict(size=14)), legend=dict(orientation="h", y=1.12),
                          yaxis=dict(gridcolor="#f0f0f0", side="right", tickprefix="₹"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"₹10,000 invested {part.index[0]:%d %b %Y} is worth ₹{growth_fund.iloc[-1]:,.0f} today. "
                   "Nifty 50 line excludes dividends, so it slightly understates the index.")

        st.markdown("#### Returns (per year)")
        rets = [("1 year", m["1Y"]), ("3 years", m["3Y"]), ("5 years", m["5Y"]), ("10 years", m["10Y"]), ("Since start", m["since_start"])]
        for c, (lab, v) in zip(st.columns(5), rets):
            c.markdown(f'<div class="stat-label">{lab}</div><div class="stat-val {cls(v) if v is not None else ""}">{pct(v, True, 1)}</div>',
                       unsafe_allow_html=True)
        st.write("")
        st.markdown("#### Risk and consistency")
        risk = [("Ups & downs (volatility)", pct(m["volatility"], digits=1)), ("Worst fall ever", pct(m["max_drawdown"])),
                ("Below its high now", pct(m["current_drawdown"])), ("Worst 3-year return", pct(m["rolling3_min"], True, 1)),
                ("3Y periods beating 10%", pct(m["rolling3_pct_above_10"]))]
        for c, (lab, v) in zip(st.columns(5), risk):
            c.markdown(f'<div class="stat-label">{lab}</div><div class="stat-val">{v}</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown("#### Falls and recoveries")
        cr = crash_recoveries(nav_s, 0.10)
        if cr.empty:
            st.markdown('<div class="insight">This fund has <b>never fallen 10% or more</b> from a high.</div>', unsafe_allow_html=True)
        else:
            done = cr.dropna(subset=["months_peak_to_recovery"])
            worst = cr.loc[cr["fall"].idxmin()]
            worst_txt = (f"Its worst fall was <b>{abs(worst['fall']):.0%}</b> ({worst['peak']:%b %Y} to {worst['trough']:%b %Y})"
                         + (f", and it was back at its high <b>{worst['months_peak_to_recovery']:.0f} months</b> after the peak."
                            if not pd.isna(worst["months_peak_to_recovery"]) else ", and it has not yet fully recovered."))
            typ = (f" Across its {len(cr)} falls of 10% or more, it typically got back to its high in "
                   f"<b>{done['months_peak_to_recovery'].median():.0f} months</b> from the peak.") if len(done) else ""
            now_txt = (f"<br><br>Right now it is <b>{abs(m['current_drawdown']):.0%} below</b> its high."
                       if m["current_drawdown"] <= -0.02 else "<br><br>Right now it is at or near its high.")
            st.markdown(f'<div class="insight">{worst_txt}{typ}{now_txt}<br><br>'
                        '<span class="muted">Past recoveries are evidence, not a promise.</span></div>', unsafe_allow_html=True)
            with st.expander("See every fall of 10% or more"):
                st.dataframe(pd.DataFrame({
                    "High on": cr["peak"].dt.strftime("%d %b %Y"), "Low on": cr["trough"].dt.strftime("%d %b %Y"),
                    "Fall": cr["fall"].map(lambda v: f"{v:.0%}"),
                    "Back at high on": cr["recovered"].map(lambda d: "Not yet" if pd.isna(d) else f"{d:%d %b %Y}"),
                    "Months from high to recovery": cr["months_peak_to_recovery"].map(lambda v: "–" if pd.isna(v) else f"{v:.0f}"),
                }), hide_index=True, use_container_width=True)

        st.write("")
        st.markdown("#### SIP calculator")
        st.caption("What a monthly SIP would be worth today, using this fund's real past NAVs "
                   "(bought on the first NAV date of each month).")
        k1, k2 = st.columns(2)
        amount = k1.number_input("Monthly SIP (₹)", min_value=100, max_value=1_000_000, value=5000, step=500)
        max_years = max(1, int((nav_s.index[-1] - nav_s.index[0]).days // 365))
        span_opts = [y for y in (1, 3, 5, 10, 15) if y <= max_years] + ["Since start"]

        def span_label(v):
            return v if isinstance(v, str) else f"{v} year{'s' if v > 1 else ''} ago"

        span = k2.segmented_control("Started", span_opts, default=5 if 5 in span_opts else span_opts[0],
                                    format_func=span_label)
        start = nav_s.index[0] if span in (None, "Since start") else nav_s.index[-1] - pd.DateOffset(years=int(span))
        sip = sip_simulation(nav_s, amount, start)
        if sip:
            gain = sip["value"] - sip["invested"]
            xr = sip["xirr"]
            cards = [("You put in", f"₹{sip['invested']:,.0f}", ""),
                     ("Worth today", f"₹{sip['value']:,.0f}", cls(gain)),
                     ("Gain", f"{'+' if gain >= 0 else '-'}₹{abs(gain):,.0f}", cls(gain)),
                     ("Yearly return (XIRR)", pct(xr, True, 1) if xr is not None else "–", cls(xr or 0))]
            for c, (lab, val, k) in zip(st.columns(4), cards):
                c.markdown(f'<div class="idx-card"><div class="idx-name">{lab}</div><div class="idx-val {k}">{val}</div></div>',
                           unsafe_allow_html=True)
            h = sip["history"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=h.index, y=h["Invested"], name="Money put in", line=dict(color="#9ca3af", width=1.5)))
            fig.add_trace(go.Scatter(x=h.index, y=h["Value"], name="Value of your SIP", line=dict(color=GREEN, width=2),
                                     fill="tonexty", fillcolor="rgba(26,127,90,0.08)"))
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white", hovermode="x unified",
                              legend=dict(orientation="h", y=1.1),
                              yaxis=dict(gridcolor="#f0f0f0", side="right", tickprefix="₹"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"{sip['installments']} monthly instalments from {start:%b %Y}. Putting the same "
                       f"₹{sip['invested']:,.0f} in all at once on day one would be worth ₹{sip['lump_value']:,.0f} today. "
                       "XIRR is the yearly return that accounts for each instalment being invested for a different time.")
        st.caption("NAV data from AMFI and mfapi.in. Past returns do not guarantee future returns. Not investment advice.")

    # ---------- Compare ----------
    elif ss.page == "Compare":
        st.subheader("Compare funds")
        chosen = st.multiselect("Funds to compare (up to 4)", fl["code"].tolist(), default=ss.compare, max_selections=4,
                                format_func=lambda c: lookup.get(c, str(c)), placeholder="Search and add funds")
        if chosen != ss.compare:
            ss.compare = chosen
            st.rerun()
        if len(chosen) < 2:
            st.info("Add at least two funds to compare. Tip: open a fund and tap + Add to compare.")
        else:
            navs = {c: load_nav(c) for c in chosen}
            navs = {c: s for c, s in navs.items() if not s.empty}
            start = max(s.index[0] for s in navs.values())
            colors = [GREEN, "#1f3a5f", AMBER, "#7c3aed"]
            fig = go.Figure()
            for (c, s), col in zip(navs.items(), colors):
                part = s[s.index >= start]
                fig.add_trace(go.Scatter(x=part.index, y=10000 * part / part.iloc[0], name=fl.set_index("code")["name"].get(c, str(c)),
                                         line=dict(color=col, width=2)))
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="white", hovermode="x unified",
                              title=dict(text=f"Growth of ₹10,000 since {start:%d %b %Y} (common start date)", font=dict(size=14)),
                              legend=dict(orientation="h", y=-0.15), yaxis=dict(gridcolor="#f0f0f0", side="right", tickprefix="₹"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            rows = []
            for c, s in navs.items():
                mm = fund_metrics(s)
                rows.append({"Fund": fl.set_index("code")["name"].get(c, str(c)), "1Y": fret(mm["1Y"]), "3Y": fret(mm["3Y"]),
                             "5Y": fret(mm["5Y"]), "10Y": fret(mm["10Y"]), "Volatility": fpct(mm["volatility"]),
                             "Worst fall": fpct(mm["max_drawdown"]), "Worst 3Y return": fpct(mm["rolling3_min"]),
                             "3Y periods beating 10%": mm["rolling3_pct_above_10"]})
            pc = st.column_config.NumberColumn(format="%.1f%%")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
                "Volatility": pc, "Worst fall": pc, "Worst 3Y return": pc,
                "3Y periods beating 10%": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
        st.caption("Returns are yearly (CAGR). Past returns do not guarantee future returns. Not investment advice.")

# ================================================================
#                     INSURANCE (kept separate from investing)
# ================================================================
elif market == "Insurance":
    insurance_page.render(ss)

# ================================================================
#                              IPOs
# ================================================================
elif market == "IPO":
    ipo_page.render(ss, open_stock)

# ================================================================
#                          PORTFOLIO + ALERTS
# ================================================================
else:
    portfolio_page.render(ss, open_stock, open_fund, load_all_nse, load_search, load_fund_list, load_nav, load_ohlc)
