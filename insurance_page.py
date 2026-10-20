"""Insurance section: compare health and life insurers on official IRDAI data, cover calculators and a
policy comparison checklist. Kept separate from the investing sections. Called from app.py."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.insurance import (FEATURES, HEALTH_DEFAULT, HEALTH_PARTS, LEVELS, LIFE_DEFAULT, LIFE_PARTS, health_cover_guide,
                            life_cover_needed, load, policy_feature_score, score)

GREEN, RED, AMBER = "#1a7f5a", "#d63a3a", "#b7791f"


@st.cache_data(ttl=86400, show_spinner=False)
def _data():
    return load()


def lakh(x):
    """₹ amount in lakh / crore, the way Indian policies are quoted."""
    if x is None or pd.isna(x):
        return "–"
    return f"₹{x / 1e7:,.2f} crore" if x >= 1e7 else f"₹{x / 1e5:,.1f} lakh"


def pc(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x:.{d}%}"


def render(ss):
    st.markdown('<div class="insight" style="border-left:5px solid #1f3a5f; margin-bottom:10px">'
                '<b>🛡️ Insurance</b> <span class="muted">· separate from investing. Insurance protects you; '
                'it is not an investment. Insurer figures are official IRDAI data.</span></div>', unsafe_allow_html=True)
    pages = ["Health insurers", "Life insurers", "How much cover?", "Compare policies"]
    nav = st.segmented_control("Menu", pages, default=ss.page if ss.page in pages else pages[0],
                               key=f"insnav_{ss.page}", label_visibility="collapsed")
    if nav and nav != ss.page:
        ss.page = nav
        st.rerun()
    if ss.page not in pages:
        ss.page = pages[0]
    try:
        life, health, meta = _data()
    except Exception as e:
        st.error(f"Insurance data not found: {e}")
        return
    {"Health insurers": health_page, "Life insurers": life_page,
     "How much cover?": cover_page, "Compare policies": compare_page}[ss.page](ss, life, health, meta)


def _weights_popover(ss, key, parts, default):
    weights = ss.get(key, default)
    with st.popover("⚖️ Change what matters to you"):
        st.caption("Choose how much each factor counts towards the score.")
        new_w = {p: st.select_slider(p, options=list(LEVELS), value=weights.get(p, "Medium"), key=f"{key}_{p}") for p in parts}
        if new_w != weights:
            ss[key] = new_w
            st.rerun()
    return weights


def _top3(df, label):
    top = df.dropna(subset=["score"]).head(3)
    if top.empty:
        return
    names = ", ".join(f"<b>{n}</b> ({int(s)})" for n, s in zip(top["name"], top["score"]))
    st.markdown(f'<div class="insight">Highest scores on the official record, with your current weights: {names}.<br>'
                f'<span class="muted">This ranks insurers, not policies. Before buying, also compare the policy itself '
                f'(see <b>Compare policies</b>): its waiting periods, room-rent limits, co-pay and price. '
                f'This is information, not a recommendation to buy.</span></div>', unsafe_allow_html=True)


# ---------------- Health insurers ----------------
def health_page(ss, life, health, meta):
    st.subheader("Health insurers: who pays claims?")
    with st.expander("What these numbers mean (read this first)", expanded=False):
        st.markdown("""
- **Claims paid:** of all claims the insurer handled in the year, the share it paid. Higher is better.
- **Claims rejected:** the share it refused to pay. Lower is better.
- **Claims ratio (retail health):** how much of the premium it received from individual and family policies it paid back as
  claims. Around **65–95% is healthy**. Much lower means it pays out little; above 100% means it is losing money on
  those policies, which can lead to sharp premium hikes.
- **Complaints per 10,000 claims:** lower is better.
- **Paid within 3 months:** speed of payment.
- **Solvency:** financial strength. The legal minimum is **1.5**. The three government-owned insurers are below it but are
  backed by the government.
- For general insurers (which also sell motor, fire and other cover), *claims paid*, *rejected* and *complaints* include
  all their lines, not only health. Standalone health insurers' figures are health-only.
""")
    weights = _weights_popover(ss, "w_health", HEALTH_PARTS, HEALTH_DEFAULT)
    df = score(health, HEALTH_PARTS, weights)
    typ = st.segmented_control("Show", ["All", "Standalone health insurers", "General insurers"], default="All")
    if typ and typ != "All":
        df = df[df["type"] == typ.rstrip("s")]
    _top3(df, "health")
    st.write("")
    table = pd.DataFrame({
        "Insurer": df["name"].str.replace("$", "", regex=False).str.strip(),
        "Type": df["type"].str.replace(" insurer", ""),
        "Score": df["score"],
        "Claims paid": df["claims_paid_pct"] * 100, "Claims rejected": df["claims_repudiated_pct"] * 100,
        "Claims ratio (retail)": df["icr_retail"] * 100,
        "Complaints / 10k claims": df["complaints_per_10k_claims"],
        "Paid within 3 months": df["paid_within_3_months"] * 100, "Solvency": df["solvency"],
        "Note": np.where(df["small_sample"], "Small health book: ratios unstable, not ranked",
                         np.where(df["solvency"] < 1.5, "Solvency below 1.5 (government-backed)" , "")),
    })
    ev = st.dataframe(table, hide_index=True, use_container_width=True, height=560, on_select="rerun", selection_mode="single-row",
                      column_config={
                          "Score": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100),
                          "Claims paid": st.column_config.NumberColumn(format="%.1f%%"),
                          "Claims rejected": st.column_config.NumberColumn(format="%.1f%%"),
                          "Claims ratio (retail)": st.column_config.NumberColumn(format="%.0f%%"),
                          "Complaints / 10k claims": st.column_config.NumberColumn(format="%.0f"),
                          "Paid within 3 months": st.column_config.NumberColumn(format="%.0f%%"),
                          "Solvency": st.column_config.NumberColumn(format="%.2f")})
    if ev.selection.rows:
        _detail(df.iloc[ev.selection.rows[0]], HEALTH_PARTS, weights)
    st.caption(f"Source: {meta['source']} (financial year {meta['year']}; solvency as of {meta['solvency_as_of']}). "
               f"{meta['url']}. Click an insurer for a plain-English summary.")


# ---------------- Life insurers ----------------
def life_page(ss, life, health, meta):
    st.subheader("Life insurers: who pays death claims?")
    ind = life.loc[life["name"] == "LIC of India", "csr_number"]
    with st.expander("What these numbers mean (read this first)", expanded=False):
        st.markdown(f"""
- **Claims paid (by number):** share of individual death claims the insurer paid. This is the classic *claim settlement
  ratio*. Most insurers are above 98%, so small differences matter.
- **Claims paid (by amount):** the same by money. If it is much lower than *by number*, bigger claims were refused or
  delayed more often.
- **Rejected / repudiated:** claims refused, usually over non-disclosure (for example hiding an illness when buying).
  **Always declare your health honestly.**
- **Paid within 30 days:** speed of payment after the claim was made.
- **Solvency:** financial strength (legal minimum 1.5).
- **Complaints** are shown as counts for reference only and are not used in the score, because IRDAI does not publish the
  insurer-wise policy counts needed for a fair comparison.
""")
    weights = _weights_popover(ss, "w_life", LIFE_PARTS, LIFE_DEFAULT)
    df = score(life, LIFE_PARTS, weights)
    _top3(df, "life")
    st.write("")
    table = pd.DataFrame({
        "Insurer": df["name"], "Score": df["score"],
        "Claims paid (number)": df["csr_number"] * 100, "Claims paid (amount)": df["csr_amount"] * 100,
        "Rejected": df["repudiated_pct"] * 100, "Paid within 30 days": df["within_30_days"] * 100,
        "Solvency": df["solvency"], "Death claims in year": df["claims_total"], "Complaints": df["complaints"],
        "Note": np.where(df["small_sample"], "Too few claims to judge", ""),
    })
    ev = st.dataframe(table, hide_index=True, use_container_width=True, height=560, on_select="rerun", selection_mode="single-row",
                      column_config={
                          "Score": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100),
                          "Claims paid (number)": st.column_config.NumberColumn(format="%.2f%%"),
                          "Claims paid (amount)": st.column_config.NumberColumn(format="%.2f%%"),
                          "Rejected": st.column_config.NumberColumn(format="%.2f%%"),
                          "Paid within 30 days": st.column_config.NumberColumn(format="%.0f%%"),
                          "Solvency": st.column_config.NumberColumn(format="%.2f"),
                          "Death claims in year": st.column_config.NumberColumn(format="%,.0f"),
                          "Complaints": st.column_config.NumberColumn(format="%,.0f")})
    if ev.selection.rows:
        _detail(df.iloc[ev.selection.rows[0]], LIFE_PARTS, weights)
    st.caption(f"Source: {meta['source']} (individual death claims, financial year {meta['year']}; solvency as of "
               f"{meta['solvency_as_of']}). {meta['url']}")


def _detail(row, parts, weights):
    st.markdown(f"#### {row['name']}")
    if pd.isna(row["score"]):
        st.info("Not ranked: too little data for this insurer to give stable figures.")
    a, b = st.columns([1, 2])
    sc = row["score"]
    colour = GREEN if (sc or 0) >= 75 else AMBER if (sc or 0) >= 55 else RED
    a.markdown(f'<div class="insight" style="text-align:center"><div class="stat-label">Track-record score</div>'
               f'<div style="font-size:2.6rem; font-weight:700; color:{colour}">{"–" if pd.isna(sc) else int(sc)}</div>'
               f'<div class="muted">out of 100</div></div>', unsafe_allow_html=True)
    with b:
        for label in parts:
            v = row.get(f"part::{label}")
            if v is not None and not pd.isna(v):
                st.progress(float(v), text=f"{label}: {v:.0%} · weight {weights.get(label, 'Medium')}")


# ---------------- Cover calculators ----------------
def cover_page(ss, life, health, meta):
    st.subheader("How much cover do I need?")
    t1, t2 = st.tabs(["Life cover (term insurance)", "Health cover"])
    with t1:
        st.caption("If your income stopped tomorrow, how much would your family need? A **term plan** is the cheapest "
                   "way to buy this cover; endowment/ULIP plans mix insurance with investment and give far less cover per rupee.")
        c1, c2 = st.columns(2)
        income = c1.number_input("Your yearly income (₹)", 0, 100_000_000, 1_000_000, 50_000)
        expenses = c2.number_input("Family's yearly expenses without you (₹)", 0, 100_000_000, 600_000, 50_000)
        years = c1.slider("For how many years would they need support?", 5, 40, 20)
        loans = c2.number_input("Outstanding loans (home, car…) (₹)", 0, 500_000_000, 0, 100_000)
        goals = c1.number_input("Future goals to fund (children's education, marriage…) (₹)", 0, 500_000_000, 2_000_000, 100_000)
        savings = c2.number_input("Savings and investments your family could use (₹)", 0, 500_000_000, 500_000, 100_000)
        existing = c1.number_input("Life cover you already have (₹)", 0, 500_000_000, 0, 100_000)
        r = life_cover_needed(expenses, years, loans, goals, savings, existing)
        st.markdown(f'<div class="insight"><div class="stat-label">Suggested additional life cover</div>'
                    f'<div style="font-size:2rem; font-weight:700">{lakh(r["need"])}</div>'
                    f'<div class="muted">= {years} years of expenses ({lakh(r["expenses_part"])}, assuming savings earn about '
                    f'2% above inflation) + loans + goals − savings − existing cover. A common quick check is 10–15× yearly '
                    f'income: {lakh(income * 10)} to {lakh(income * 15)}.</div></div>', unsafe_allow_html=True)
        st.caption("Choose cover until your retirement age, buy from an insurer with a strong claims record (see Life insurers), "
                   "and declare your health and habits honestly so claims are not rejected.")
    with t2:
        st.caption("A rule-of-thumb starting point for a family floater policy. Hospital costs vary a lot by city and hospital.")
        c1, c2 = st.columns(2)
        city = c1.selectbox("Where do you live?", ["Metro city", "Tier-2 city", "Smaller town"])
        members = c2.number_input("People to cover", 1, 10, 4)
        oldest = c1.number_input("Age of the oldest person", 0, 100, 35)
        employer = c2.number_input("Cover from your employer (₹)", 0, 100_000_000, 0, 100_000)
        g = health_cover_guide(city, members, oldest, employer)
        st.markdown(f'<div class="insight"><div class="stat-label">Total health cover to aim for</div>'
                    f'<div style="font-size:2rem; font-weight:700">{lakh(g["base_low"])} – {lakh(g["base_high"])}</div>'
                    f'<div class="muted">Your own policy (beyond employer cover, which ends if you change jobs): '
                    f'{lakh(g["own_policy_low"])} – {lakh(g["own_policy_high"])}.<br>'
                    f'A cheaper way to get high cover: a base policy of ₹5–10 lakh plus a <b>super top-up</b> of about '
                    f'{lakh(g["topup"])}, which pays once bills cross the base amount.</div></div>', unsafe_allow_html=True)
        if oldest >= 60:
            st.info("For people above 60, look for policies built for senior citizens and check co-payment terms carefully.")


# ---------------- Policy comparison ----------------
def compare_page(ss, life, health, meta):
    st.subheader("Compare policies you are considering")
    t1, t2 = st.tabs(["Health policies", "Term (life) plans"])
    insurers_h = sorted(health["name"].str.replace("$", "", regex=False).str.strip())
    with t1:
        st.caption("Enter up to 3 health policies from their brochures or policy wordings. The checklist scores the "
                   "features that most often cause claim surprises.")
        cols = st.columns(3)
        rows = []
        for i, c in enumerate(cols):
            with c.container(border=True):
                name = st.text_input("Policy name", value=f"Policy {i + 1}", key=f"hp_name{i}")
                ins = st.selectbox("Insurer", insurers_h, index=None, key=f"hp_ins{i}", placeholder="Choose insurer")
                cover = st.number_input("Cover (₹)", 0, 100_000_000, 1_000_000, 100_000, key=f"hp_cov{i}")
                prem = st.number_input("Yearly premium (₹)", 0, 10_000_000, 15_000, 500, key=f"hp_prem{i}")
                choices = {k: st.selectbox(lab, opts, key=f"hp_{k}{i}") for k, lab, opts, _ in FEATURES}
                rows.append((name, ins, cover, prem, choices))
        scored = score(health, HEALTH_PARTS, ss.get("w_health", HEALTH_DEFAULT))
        scored["clean"] = scored["name"].str.replace("$", "", regex=False).str.strip()
        out = []
        for name, ins, cover, prem, ch in rows:
            ins_score = scored.loc[scored["clean"] == ins, "score"].iloc[0] if ins else np.nan
            flags = [lab for k, lab, opts, pts in FEATURES if pts[opts.index(ch[k])] <= 0.2]
            out.append({"Policy": name, "Insurer": ins or "–", "Policy features score": policy_feature_score(ch),
                        "Insurer track-record score": ins_score,
                        "Premium per ₹1 lakh cover": prem / (cover / 1e5) if cover else np.nan,
                        "Watch out for": ", ".join(flags) if flags else "–"})
        st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True, column_config={
            "Policy features score": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100),
            "Insurer track-record score": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100),
            "Premium per ₹1 lakh cover": st.column_config.NumberColumn(format="₹%.0f")})
        st.caption("A good policy usually has both a strong insurer record and few restrictions. A low premium with many "
                   "restrictions often costs more when you actually claim.")
    with t2:
        st.caption("Compare term-plan quotes (from insurer websites or aggregators) for the same cover and term.")
        lnames = sorted(life["name"])
        lscored = score(life, LIFE_PARTS, ss.get("w_life", LIFE_DEFAULT)).set_index("name")
        cols = st.columns(3)
        out = []
        for i, c in enumerate(cols):
            with c.container(border=True):
                ins = st.selectbox("Insurer", lnames, index=None, key=f"tp_ins{i}", placeholder="Choose insurer")
                cover = st.number_input("Cover (₹)", 0, 1_000_000_000, 10_000_000, 500_000, key=f"tp_cov{i}")
                prem = st.number_input("Yearly premium (₹)", 0, 10_000_000, 12_000, 500, key=f"tp_prem{i}")
                term = st.number_input("Policy term (years)", 5, 85, 30, key=f"tp_term{i}")
            if ins:
                r = lscored.loc[ins]
                out.append({"Insurer": ins, "Cover": lakh(cover), "Term": f"{term} yrs",
                            "Premium per ₹1 lakh cover": prem / (cover / 1e5) if cover else np.nan,
                            "Claims paid (number)": r["csr_number"] * 100, "Claims paid (amount)": r["csr_amount"] * 100,
                            "Track-record score": r["score"]})
        if out:
            st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True, column_config={
                "Premium per ₹1 lakh cover": st.column_config.NumberColumn(format="₹%.1f"),
                "Claims paid (number)": st.column_config.NumberColumn(format="%.2f%%"),
                "Claims paid (amount)": st.column_config.NumberColumn(format="%.2f%%"),
                "Track-record score": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100)})
        else:
            st.info("Choose an insurer in at least one column to compare.")
    st.caption(f"Insurer data: {meta['source']}. Policy features come from what you enter. Not insurance or financial advice.")
