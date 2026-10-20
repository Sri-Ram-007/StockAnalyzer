"""Portfolio, add-holding and alerts pages (Phase 5). Called from app.py."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.alerts import build_message, load_config, load_rules, save_config, save_rules, send
from core.markets import INDIAN_ETFS, currency_code
from core.portfolio import add_holding, load_holdings, remove_holding, summary, value_portfolio
from core.goals import ASSET_CHOICES, category_returns, future_cost, monthly_sip_needed, projection, retirement_corpus
from core.markets import get_nse_etfs
from core.tax import CLASS_LABEL, DEFAULT_RULES, classify, estimate


def inr(x, signed=False):
    if x is None or pd.isna(x):
        return "–"
    x = 0.0 if abs(x) < 0.5 else x
    return f"{'-' if x < 0 else ('+' if signed and x > 0 else '')}₹{abs(x):,.0f}"


def cls(x):
    return "up" if x >= 0 else "down"


def render(ss, open_stock, open_fund, load_all_nse, load_search, load_fund_list, load_nav, load_ohlc):
    pages = ["Holdings", "Add holding", "Tax", "Goals", "Alerts"]
    nav = st.segmented_control("Menu", pages, default=ss.page if ss.page in pages else "Holdings",
                               key=f"pnav_{ss.page}", label_visibility="collapsed")
    if nav and nav != ss.page:
        ss.page = nav
        st.rerun()
    if ss.page not in pages:
        ss.page = "Holdings"

    if ss.page == "Holdings":
        holdings_page(ss, open_stock, open_fund)
    elif ss.page == "Add holding":
        add_page(ss, load_all_nse, load_search, load_fund_list, load_nav, load_ohlc)
    elif ss.page == "Tax":
        tax_page(ss, load_fund_list)
    elif ss.page == "Goals":
        goals_page(ss, load_fund_list)
    else:
        alerts_page(ss)


# ---------------- Holdings ----------------
def holdings_page(ss, open_stock, open_fund):
    holdings = load_holdings()
    if not holdings:
        st.info("Your portfolio is empty. Go to **Add holding** to add the stocks, ETFs or mutual funds you own.")
        return
    with st.spinner("Valuing your portfolio…"):
        df = value_portfolio(holdings)
    sm = summary(df)
    cards = [("Current value", inr(sm["value"]), None), ("Invested", inr(sm["invested"]), None),
             ("Total profit / loss", inr(sm["pnl"], True), sm["pnl_pct"]), ("Today", inr(sm["day_pnl"], True), sm["day_pct"])]
    for c, (lab, val, p) in zip(st.columns(4), cards):
        has_p = p is not None and not pd.isna(p)
        sub = f'<span class="chg {cls(p)}"> ({p:+.2%})</span>' if has_p else ""
        c.markdown(f'<div class="idx-card"><div class="idx-name">{lab}</div>'
                   f'<div class="idx-val {cls(p) if has_p else ""}">{val}{sub}</div></div>', unsafe_allow_html=True)
    st.write("")

    table = pd.DataFrame({
        "Name": df["name"],
        "Type": df["kind"].map({"stock": "Stock / ETF", "fund": "Mutual fund"}),
        "Quantity": df["qty"],
        "Avg buy": df.apply(lambda r: f"{r['buy_price']:,.2f} {r['currency']}", axis=1),
        "Now": df.apply(lambda r: f"{r['price']:,.2f} {r['currency']}", axis=1),
        "Value (₹)": df["value_inr"].round(0),
        "Profit / loss (₹)": df["pnl_inr"].round(0),
        "Profit / loss %": df["pnl_pct"] * 100,
        "Today %": df["day_pct"] * 100,
        "Below 1-yr high": df["below_1y_high"] * 100,
        "Share of portfolio": df["weight"],
    })
    st.caption("Click a row to open its full analysis. Foreign holdings are converted to rupees: cost at the exchange "
               "rate on your buy date, value at today's rate.")
    ev = st.dataframe(table, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
                      column_config={
                          "Value (₹)": st.column_config.NumberColumn(format="₹%,.0f"),
                          "Profit / loss (₹)": st.column_config.NumberColumn(format="₹%,.0f"),
                          "Profit / loss %": st.column_config.NumberColumn(format="%+.1f%%"),
                          "Today %": st.column_config.NumberColumn(format="%+.2f%%"),
                          "Below 1-yr high": st.column_config.NumberColumn(format="%.0f%%"),
                          "Share of portfolio": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                      })
    if ev.selection.rows:
        r = df.iloc[ev.selection.rows[0]]
        if r["kind"] == "fund":
            ss.market = "Funds"
            open_fund(r["ident"])
        else:
            ss.market = "India" if str(r["ident"]).endswith((".NS", ".BO")) else "Global"
            open_stock(r["ident"], r["name"])
        st.rerun()

    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Pie(labels=df["name"], values=df["value_inr"].clip(lower=0), hole=0.55, textinfo="percent", sort=False))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), title=dict(text="Where your money is", font=dict(size=14)),
                          legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with b:
        st.markdown("**Manage**")
        by_id = df.set_index("id")
        drop = st.selectbox("Remove a holding", df["id"], index=None, placeholder="Choose a holding to remove",
                            format_func=lambda i: f"{by_id.loc[i, 'name']} · {by_id.loc[i, 'qty']:g} units")
        if drop and st.button("Remove"):
            remove_holding(drop)
            st.rerun()
        st.download_button("⬇️ Download portfolio (CSV)", table.to_csv(index=False).encode("utf-8"),
                           file_name="my_portfolio.csv", mime="text/csv")
    st.caption("Your holdings are saved only on this computer, in data/portfolio.json. Not investment advice.")


# ---------------- Add holding ----------------
def add_page(ss, load_all_nse, load_search, load_fund_list, load_nav, load_ohlc):
    st.subheader("Add a holding")
    kind = st.segmented_control("What did you buy?", ["Indian stock / ETF", "US or global stock / ETF", "Mutual fund"],
                                default="Indian stock / ETF")
    ident = name = None
    if kind == "Indian stock / ETF":
        nse = load_all_nse()
        opts = dict(zip(nse["Ticker"], nse["Company Name"]))
        opts.update({t: n for t, n, _ in INDIAN_ETFS})
        ident = st.selectbox("Stock or ETF", sorted(opts, key=lambda t: opts[t]), index=None,
                             format_func=lambda t: f"{opts[t]} · {t.replace('.NS', '')}", placeholder="Search NSE stocks and ETFs")
        name = opts.get(ident)
    elif kind == "US or global stock / ETF":
        q = st.text_input("Search by name or ticker", placeholder="e.g. Apple, Microsoft, VOO, Toyota")
        res = load_search(q) if q else []
        if q and not res:
            st.info("Nothing found. Try another name or the ticker symbol.")
        if res:
            i = st.selectbox("Choose", range(len(res)), index=None, placeholder="Pick from the results",
                             format_func=lambda k: f"{res[k]['name']} · {res[k]['ticker']} · {res[k]['exchange']}")
            if i is not None:
                ident, name = res[i]["ticker"], res[i]["name"]
    elif kind == "Mutual fund":
        fl = load_fund_list()
        lookup = dict(zip(fl["code"], fl["display"]))
        dg = fl[fl["is_direct"] & fl["is_growth"]]["code"].tolist()
        dg_set = set(dg)
        code = st.selectbox("Mutual fund", dg + [c for c in fl["code"] if c not in dg_set], index=None,
                            format_func=lambda c: lookup.get(c, str(c)), placeholder="Search mutual funds")
        if code is not None:
            ident, name = str(code), fl.set_index("code")["name"].get(code, str(code))

    if ident:
        is_fund = kind == "Mutual fund"
        if is_fund:
            series = load_nav(int(ident))
        else:
            o = load_ohlc(ident)
            series = o["Close"] if not o.empty else pd.Series(dtype=float)
        cur_code = "INR" if is_fund else currency_code(ident)
        latest = float(series.iloc[-1]) if len(series) else 0.0
        st.caption(f"Latest {'NAV' if is_fund else 'price'}: {latest:,.2f} {cur_code}")
        c1, c2, c3 = st.columns(3)
        qty = c1.number_input("Units" if is_fund else "Number of shares", min_value=0.0, value=1.0, step=1.0,
                              format="%.4f" if is_fund else "%.2f")
        price = c2.number_input(f"Average buy {'NAV' if is_fund else 'price'} ({cur_code})", min_value=0.0,
                                value=round(latest, 2), step=1.0, format="%.4f" if is_fund else "%.2f")
        when = c3.date_input("Buy date", value=pd.Timestamp.now().date(), max_value=pd.Timestamp.now().date())
        if st.button("Add to portfolio", type="primary", disabled=qty <= 0 or price <= 0):
            add_holding("fund" if is_fund else "stock", ident, name, qty, price, str(when), cur_code)
            ss.page = "Holdings"
            st.rerun()
    st.caption("Tip: if you bought the same stock several times, enter the total quantity and your average buy price.")


# ---------------- Alerts ----------------
def alerts_page(ss):
    st.subheader("Daily alerts")
    st.write("Get one message a day on **Telegram** and/or **email** with your portfolio summary and anything new: "
             "a holding or watchlist item crossing a fall level, a big one-day move, a price target being hit, "
             "or a holding's loss crossing your limit.")
    rules = load_rules()
    with st.container(border=True):
        st.markdown("**1. What should I alert you about?**")
        c1, c2, c3 = st.columns(3)
        falls = c1.multiselect("Fall below the 1-year high", [10, 15, 20, 25, 30, 40, 50], default=rules["fall_from_high"],
                               format_func=lambda v: f"{v}%")
        day_move = c2.number_input("Moves more than (% in one day)", 0.0, 50.0, float(rules["day_move"]), 0.5)
        loss = c3.number_input("Holding loss crosses (%)", 0.0, 90.0, float(rules["holding_loss"]), 1.0)
        daily = st.toggle("Include a daily portfolio summary", value=rules["daily_summary"])
        ipos = st.toggle("Include Indian IPOs (opening, open, just closed with subscription, just listed)",
                         value=rules.get("include_ipos", True))
        names_map = {h["ident"]: h["name"] for h in load_holdings()}
        names_map.update({t: t for t in ss.watchlist if t not in names_map})
        st.markdown("Price targets (optional): alert when the price rises above or falls below a level you choose.")
        tg = pd.DataFrame(rules.get("targets") or [], columns=["ident", "above", "below"])
        tg = st.data_editor(tg, num_rows="dynamic", use_container_width=True, column_config={
            "ident": st.column_config.SelectboxColumn("Holding / watchlist item", options=list(names_map)),
            "above": st.column_config.NumberColumn("Alert if above"),
            "below": st.column_config.NumberColumn("Alert if below")})
        if st.button("Save alert rules", type="primary"):
            targets = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in tg.dropna(subset=["ident"]).to_dict("records")]
            save_rules({"fall_from_high": sorted(falls), "day_move": day_move, "holding_loss": loss,
                        "daily_summary": daily, "include_ipos": ipos, "targets": targets})
            st.success("Saved.")

    with st.container(border=True):
        st.markdown("**2. Where should I send them?**")
        cfg = load_config()
        with st.expander("How to set up Telegram (recommended, about 2 minutes)"):
            st.markdown("""
1. In Telegram, open **@BotFather**, send `/newbot`, and follow the steps. It gives you a **bot token**.
2. Open your new bot and send it any message (for example "hi").
3. Open **@userinfobot** and send any message. It replies with your **chat id** (a number).
4. Paste both below, press **Save delivery settings**, then **Send a test message now**.
""")
        with st.expander("How to set up email (Gmail)"):
            st.markdown("""
1. Turn on 2-Step Verification for your Google account.
2. Create an **App password** (Google Account → Security → App passwords). It is a 16-letter code.
3. Enter your Gmail address and that app password below. Never use your normal Gmail password here.
""")
        a, b = st.columns(2)
        token = a.text_input("Telegram bot token", value=cfg.get("TELEGRAM_BOT_TOKEN", ""), type="password")
        chat = b.text_input("Telegram chat id", value=cfg.get("TELEGRAM_CHAT_ID", ""))
        c, d, e = st.columns(3)
        em = c.text_input("Gmail address", value=cfg.get("EMAIL_ADDRESS", ""))
        pw = d.text_input("Gmail app password", value=cfg.get("EMAIL_APP_PASSWORD", ""), type="password")
        to = e.text_input("Send to (optional)", value=cfg.get("EMAIL_TO", ""), placeholder="defaults to your Gmail")
        if st.button("Save delivery settings"):
            save_config({"TELEGRAM_BOT_TOKEN": token.strip(), "TELEGRAM_CHAT_ID": chat.strip(), "EMAIL_ADDRESS": em.strip(),
                         "EMAIL_APP_PASSWORD": pw.replace(" ", ""), "EMAIL_TO": to.strip()})
            st.success("Saved on this computer only (data/alerts_config.json).")

    with st.container(border=True):
        st.markdown("**3. Preview and test**")
        p1, p2 = st.columns(2)
        if p1.button("👀 Preview today's message"):
            with st.spinner("Checking your holdings and watchlist…"):
                ss.preview = build_message()
        if p2.button("📨 Send a test message now"):
            with st.spinner("Sending…"):
                for r in send(build_message()):
                    (st.success if r.endswith("sent") else st.info)(r)
        if ss.get("preview"):
            st.markdown(f'<div class="insight">{ss.preview.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**4. Make it automatic**")
        st.markdown("""
**Already set up: GitHub (free, works even when your laptop is off, no passwords).**
The project is on GitHub, and every weekday at about 5 pm India time it posts the report as an *issue* in your private
repository. GitHub notifies you by email and in the GitHub app automatically.
- To test it now: GitHub → your repository → **Actions** → *Daily stock alerts* → **Run workflow**.
- After changing holdings, watchlist or rules here, double-click **`Sync to GitHub.bat`** so the report uses your latest data.

Telegram and email (sections 2 and 3 above) are optional extras; you don't need them.

**Option B, only while your laptop is on (Windows Task Scheduler).** Double-click `Send Alerts Now.bat` to test, then
open *Task Scheduler → Create Basic Task → Daily, 5:00 PM → Start a program* and choose that `.bat` file.
""")
    st.caption("Alerts report what happened; they are not buy or sell signals.")


# ---------------- Tax ----------------
@st.cache_data(ttl=86400, show_spinner=False)
def _goal_returns(choice):
    from core.funds import get_fund_list
    return category_returns(get_fund_list(), choice)


def tax_page(ss, load_fund_list):
    st.subheader("Tax if you sold today")
    st.caption("An estimate of capital-gains tax on your holdings, so you can plan sales. Uses the rules from Budget July 2024 "
               "by default; change the rates below if a later Budget changed them.")
    holdings = load_holdings()
    if not holdings:
        st.info("Add your holdings first (Add holding). The tax estimate uses your buy dates and prices.")
        return
    with st.spinner("Valuing your portfolio…"):
        df = value_portfolio(holdings)
    fl = load_fund_list()
    cat = dict(zip(fl["code"].astype(str), fl["category"].fillna("")))
    try:
        etf = get_nse_etfs().set_index("Ticker")["Industry"].to_dict()
    except Exception:
        etf = {}
    classes = {r["id"]: classify(r["kind"], r["ident"], r["name"], cat.get(str(r["ident"]), ""), etf.get(r["ident"], ""))
               for _, r in df.iterrows()}

    c1, c2 = st.columns([1, 2])
    slab = c1.selectbox("Your income-tax slab rate", [0, 5, 10, 15, 20, 25, 30], index=6,
                        format_func=lambda v: f"{v}%", help="Used for debt funds and short-term gains on gold and foreign shares.")
    with c2.expander("Tax rates used (edit if the rules have changed)"):
        r = dict(DEFAULT_RULES)
        a, b, c = st.columns(3)
        r["equity_stcg"] = a.number_input("Equity short-term %", 0.0, 50.0, DEFAULT_RULES["equity_stcg"] * 100, 0.5) / 100
        r["equity_ltcg"] = b.number_input("Equity long-term %", 0.0, 50.0, DEFAULT_RULES["equity_ltcg"] * 100, 0.5) / 100
        r["equity_ltcg_exemption"] = c.number_input("Yearly equity LTCG exemption (₹)", 0, 1_000_000,
                                                    DEFAULT_RULES["equity_ltcg_exemption"], 5000)
        a, b, c = st.columns(3)
        r["equity_lt_months"] = a.number_input("Equity long-term after (months)", 1, 60, DEFAULT_RULES["equity_lt_months"])
        r["foreign_lt_months"] = b.number_input("Foreign shares long-term after (months)", 1, 60, DEFAULT_RULES["foreign_lt_months"])
        r["other_ltcg"] = c.number_input("Gold / foreign long-term %", 0.0, 50.0, DEFAULT_RULES["other_ltcg"] * 100, 0.5) / 100
    t, s = estimate(df, classes, slab / 100, r)

    cards = [("Equity short-term gains", inr(s["equity_st"])), ("Equity long-term gains", inr(s["equity_lt"])),
             ("Other taxable gains", inr(s["slab_gains"] + s["other_lt"])), ("Estimated tax (incl. 4% cess)", inr(s["tax"]))]
    for col, (lab, val) in zip(st.columns(4), cards):
        col.markdown(f'<div class="idx-card"><div class="idx-name">{lab}</div><div class="idx-val">{val}</div></div>',
                     unsafe_allow_html=True)
    st.write("")

    tips = []
    if s["unused_exemption"] > 0:
        tips.append(f"You can still book up to <b>{inr(s['unused_exemption'])}</b> of long-term equity gains this financial year "
                    f"tax-free. Selling and buying back (\"tax-gain harvesting\") resets your cost and saves tax later.")
    soon = t[(t["term"] == "Short-term") & t["days_to_lt"].notna() & (t["days_to_lt"] <= 60) & (t["gain"] > 0)]
    for _, x in soon.iterrows():
        tips.append(f"<b>{x['name']}</b> becomes long-term in <b>{int(x['days_to_lt'])} days</b> ({x['lt_date']:%d %b %Y}). "
                    f"Waiting would lower the tax on its {inr(x['gain'])} gain.")
    losers = t[t["gain"] < 0]
    if len(losers) and (s["equity_st"] + s["equity_lt"] + s["slab_gains"] + s["other_lt"]) > 0:
        tips.append(f"Holdings at a loss ({', '.join(losers['name'].head(4))}) can be sold to offset gains "
                    "(\"tax-loss harvesting\").")
    if tips:
        st.markdown('<div class="insight">' + "<br><br>".join("💡 " + x for x in tips) + "</div>", unsafe_allow_html=True)
        st.write("")

    table = pd.DataFrame({
        "Holding": t["name"], "Type": t["class"].map(CLASS_LABEL),
        "Held for": t["days_held"].map(lambda d: f"{d // 365}y {(d % 365) // 30}m" if d >= 365 else f"{d // 30}m {d % 30}d"),
        "Status": [("Slab rate (no long-term)" if c == "debt" else term) for c, term in zip(t["class"], t["term"])],
        "Long-term from": [("–" if (c == "debt" or term == "Long-term" or pd.isna(d)) else f"{d:%d %b %Y}")
                           for c, term, d in zip(t["class"], t["term"], t["lt_date"])],
        "Gain if sold today": t["gain"].map(lambda v: inr(v, True)), "Tax rate": t["rate"]})
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Estimate only, not tax advice. Assumes one purchase per holding (your average price and buy date). Ignores surcharge, "
               "grandfathering of equity bought before Feb 2018, and older debt-fund rules. ELSS funds can only be sold after "
               "3 years. Confirm with your tax filing or a tax professional.")


# ---------------- Goals ----------------
def goals_page(ss, load_fund_list):
    st.subheader("🎯 Plan a goal")
    st.caption("How much to invest each month to reach a goal, using the real past returns of mutual funds in the category you "
               "choose, not an assumed flat return.")
    kind = st.segmented_control("Goal", ["A goal with a target amount", "Retirement"], default="A goal with a target amount")
    c1, c2 = st.columns(2)
    if kind == "Retirement":
        expense = c1.number_input("Monthly expenses today (₹)", 0, 10_000_000, 50_000, 5_000)
        age = c2.number_input("Your age", 18, 70, 30)
        retire = c1.number_input("Retire at age", 30, 80, 60)
        live = c2.number_input("Plan money until age", 60, 100, 85)
        inflation = c1.slider("Expected inflation (% a year)", 3.0, 10.0, 6.0, 0.5) / 100
        years = max(1, retire - age)
        target = retirement_corpus(expense, years, max(1, live - retire), inflation)
        goal_text = (f"To pay {inr(expense)} a month in today's money from age {retire} to {live}, you need about "
                     f"<b>{inr(target)}</b> on retirement day (expenses grow with inflation; the money is assumed to earn about "
                     f"7% a year after you retire).")
    else:
        amount = c1.number_input("Goal amount in today's money (₹)", 0, 1_000_000_000, 2_000_000, 100_000,
                                 help="e.g. child's education, a house down payment")
        years = c2.number_input("Years until you need it", 1, 40, 10)
        inflation = c1.slider("Expected inflation for this goal (% a year)", 3.0, 12.0, 6.0, 0.5) / 100
        target = future_cost(amount, years, inflation)
        goal_text = f"{inr(amount)} today will cost about <b>{inr(target)}</b> in {years} years at {inflation:.1%} inflation."
    saved = c2.number_input("Already saved for this goal (₹)", 0, 1_000_000_000, 0, 50_000)
    step = c1.slider("Increase the SIP each year by (%)", 0, 20, 5, help="A 'step-up SIP' that grows with your salary.") / 100
    choice = c2.selectbox("Invest in", list(ASSET_CHOICES), index=1)
    st.markdown(f'<div class="insight">{goal_text}</div>', unsafe_allow_html=True)

    with st.spinner("Checking real returns of funds in this category…"):
        cr = _goal_returns(choice)
    if not cr:
        st.warning("Not enough fund history in this category. Try another one.")
        return
    st.write("")
    st.markdown(f"**Real returns:** over the last {cr['period']}, the {cr['funds']} funds in this category earned between "
                f"**{cr['cautious']:.1%}** (lower quarter) and **{cr['good']:.1%}** (upper quarter) a year; the middle fund earned "
                f"**{cr['typical']:.1%}**.")
    scen = [("If returns are like the weaker funds", cr["cautious"]), ("If returns are typical", cr["typical"]),
            ("If returns are like the better funds", cr["good"])]
    for col, (lab, rate) in zip(st.columns(3), scen):
        sip = monthly_sip_needed(target, years, rate, saved, step)
        col.markdown(f'<div class="idx-card"><div class="idx-name">{lab} ({rate:.1%})</div>'
                     f'<div class="idx-val">{inr(sip)} / month</div>'
                     f'<div class="muted">starting SIP{", rising " + f"{step:.0%} a year" if step else ""}</div></div>',
                     unsafe_allow_html=True)
    sip_typ = monthly_sip_needed(target, years, cr["typical"], saved, step)
    pr = projection(years, cr["typical"], sip_typ, saved, step)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pr["Year"], y=pr["Put in"], name="Money you put in", line=dict(color="#9ca3af")))
    fig.add_trace(go.Scatter(x=pr["Year"], y=pr["Value"], name="Expected value (typical returns)", line=dict(color="#1a7f5a", width=2),
                             fill="tonexty", fillcolor="rgba(26,127,90,0.08)"))
    fig.add_hline(y=target, line_dash="dot", line_color="#d63a3a", annotation_text=f"Goal {inr(target)}")
    fig.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white", hovermode="x unified",
                      legend=dict(orientation="h", y=1.1), xaxis=dict(title="Years from now"),
                      yaxis=dict(gridcolor="#f0f0f0", tickprefix="₹"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if choice.startswith(("Small cap", "Mid cap")) and years < 7:
        st.warning("Small and mid cap funds can fall 40–50% in bad years. For goals under 7 years, a steadier category is safer.")
    if years <= 3 and "Liquid" not in choice and "bond" not in choice:
        st.warning("For goals within 3 years, equity is risky: a market fall just before you need the money may not recover in time.")
    st.caption("Past returns of today's funds (funds that closed are not included, so these figures lean optimistic). "
               "Returns vary year to year; review the plan every year. Not investment advice.")
