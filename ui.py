"""Shared look and feel: theme CSS, cards, sparklines and list rows."""
import hashlib
import itertools

import numpy as np
import streamlit as st

GREEN, RED, MUTED, INK, ACCENT = "#16a34a", "#dc2626", "#6b7280", "#0f172a", "#0f766e"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea { font-family: 'Inter', system-ui, sans-serif !important; }
#MainMenu, footer, [data-testid="stDecoration"] { display: none; }
.stApp { background: #f4f6f9; }
.block-container { padding-top: 4.6rem; max-width: 1240px; }   /* room for Streamlit Cloud's top bar */

/* ---------- brand ---------- */
.brand-wrap { display: flex; align-items: center; gap: 12px; }
.brand-logo { width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #0f766e, #22c55e); color: #fff; font-weight: 800; font-size: 1.05rem;
  box-shadow: 0 4px 12px rgba(15,118,110,.25); }
.brand { font-size: 1.35rem; font-weight: 800; color: #0f172a; line-height: 1.1; letter-spacing: -.01em; }
.brand-sub { font-size: .75rem; color: #6b7280; }

/* ---------- pill menus (segmented controls) ---------- */
[data-testid="stButtonGroup"] > div { flex-wrap: wrap; background: #e9edf2; border-radius: 12px; padding: 4px; gap: 2px;
  width: fit-content; max-width: 100%; }
button[data-variant="segmented_control"] { border: none !important; background: transparent !important; border-radius: 9px !important;
  color: #475569 !important; font-weight: 500 !important; padding: 4px 14px !important; min-height: 34px; box-shadow: none !important; }
button[data-variant="segmented_control"]:hover { color: #0f172a !important; background: rgba(255,255,255,.55) !important; }
button[data-variant="segmented_control"][aria-checked="true"] { background: #fff !important; color: #0f766e !important;
  font-weight: 600 !important; box-shadow: 0 1px 3px rgba(15,23,42,.12) !important; }

/* ---------- cards ---------- */
[class*="st-key-card_"] { background: #fff; border: 1px solid #e6e9ef !important; border-radius: 16px !important;
  box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.06); padding: 14px 16px 8px 16px !important; }
.card-title { font-weight: 700; color: #0f172a; font-size: 1rem; margin-bottom: 4px; }
.insight { border: 1px solid #e6e9ef; border-radius: 14px; padding: 16px 18px; background: #fff;
  box-shadow: 0 1px 2px rgba(16,24,40,.04); }
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { background: #fff; border-radius: 14px; padding: 4px; }
[data-testid="stExpander"] details { background: #fff; border-radius: 12px !important; border-color: #e6e9ef !important; }
[data-testid="stPopoverButton"], .stTextInput input, [data-baseweb="select"] > div { border-radius: 10px !important; }
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input { background: #fff !important; }

/* ---------- index cards ---------- */
.idx-card { border: 1px solid #e6e9ef; border-radius: 14px; padding: 10px 14px; background: #fff;
  box-shadow: 0 1px 2px rgba(16,24,40,.04); display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.idx-name { font-size: .72rem; color: #6b7280; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }
.idx-val { font-size: 1.1rem; font-weight: 700; color: #0f172a; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: .78rem; font-weight: 600; }
.pill.up { background: #dcfce7; color: #15803d; } .pill.down { background: #fee2e2; color: #b91c1c; }
.pill.flat { background: #f1f5f9; color: #475569; }
.up { color: #16a34a; } .down { color: #dc2626; }

/* ---------- list rows ---------- */
.row { display: flex; align-items: center; gap: 12px; padding: 6px 2px; }
.avatar { width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: .78rem; flex: none; border: 1px solid #eef1f5; }
.row-name { font-weight: 600; color: #0f172a; font-size: .92rem; line-height: 1.2; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; max-width: 100%; }
.row-sub { font-size: .75rem; color: #6b7280; }
.row-right { text-align: right; }
.row-price { font-weight: 600; color: #0f172a; font-size: .92rem; }
div[data-testid="stButton"] button[kind="tertiary"] { justify-content: center; width: 100%; padding: 4px; color: #94a3b8; }
div[data-testid="stButton"] button[kind="tertiary"]:hover { background: #f1f5f9; color: #0f766e; border-radius: 8px; }
[class*="st-key-card_"] [data-testid="stHorizontalBlock"] { border-bottom: 1px solid #f1f5f9; position: relative; border-radius: 10px; }
/* the whole list row is clickable: stretch the row's open-button over the row */
[class*="st-key-card_"] [data-testid="stHorizontalBlock"] :is([data-testid="stColumn"], [data-testid="stVerticalBlock"],
  [data-testid="stElementContainer"], [data-testid="stButton"], [data-testid="stTooltipHoverTarget"]) { position: static !important; }
[class*="st-key-card_"] [data-testid="stHorizontalBlock"] button[kind="tertiary"] { position: absolute; inset: 0; width: 100%;
  height: 100%; justify-content: flex-end; padding-right: 4px; background: transparent; z-index: 2; border-radius: 10px; }
[class*="st-key-card_"] [data-testid="stHorizontalBlock"] button[kind="tertiary"] > div { width: 100%; display: flex;
  justify-content: flex-end; }
[class*="st-key-card_"] [data-testid="stHorizontalBlock"] button[kind="tertiary"] p { text-align: right; font-size: 1.2rem; }
[class*="st-key-card_"] [data-testid="stHorizontalBlock"] button[kind="tertiary"]:hover { background: rgba(15,118,110,.06); }

/* ---------- stock page ---------- */
.price-big { font-size: 2.2rem; font-weight: 800; color: #0f172a; line-height: 1.1; letter-spacing: -.02em; }
.chg { font-size: 1rem; font-weight: 600; }
.muted { color: #6b7280; font-size: .9rem; }
.chip { display: inline-block; padding: 2px 10px; border-radius: 999px; background: #eef2f7; color: #334155; font-size: .78rem;
  font-weight: 600; margin-right: 6px; }
.stat-label { color: #6b7280; font-size: .78rem; }
.stat-val { font-weight: 600; font-size: 1rem; color: #0f172a; }
.range-track { position: relative; height: 6px; background: linear-gradient(90deg, #fecaca, #e5e7eb, #bbf7d0); border-radius: 4px; margin: 10px 0 4px; }
.range-dot { position: absolute; top: -5px; width: 16px; height: 16px; border-radius: 50%; background: #0f172a;
  border: 3px solid #fff; box-shadow: 0 0 0 1px #cbd5e1; }
[data-baseweb="tab-list"] { gap: 4px; }
[data-baseweb="tab"] { font-weight: 600; }
[data-baseweb="tab-highlight"] { background-color: #0f766e !important; }
h3 { letter-spacing: -.01em; }

@media (max-width: 640px) {
  .block-container { padding-left: .7rem; padding-right: .7rem; padding-top: 4rem; }
  .brand { font-size: 1.15rem; } .brand-logo { width: 34px; height: 34px; }
  .st-key-idxstrip [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .35rem; }
  .st-key-idxstrip [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }
  .st-key-idxstrip .idx-card { padding: 6px 7px; display: block; }
  .st-key-idxstrip .idx-name { font-size: .58rem; } .st-key-idxstrip .idx-val { font-size: .8rem; }
  .st-key-idxstrip .pill { font-size: .62rem; padding: 1px 5px; } .st-key-idxstrip svg { display: none; }
  .price-big { font-size: 1.8rem; }
  [class*="st-key-card_"] .avatar { display: none; }
  [class*="st-key-card_"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
  [class*="st-key-card_"] [data-testid="stColumn"] { min-width: 0 !important; }
}
</style>"""

_counter = itertools.count()


def start_run():
    """Inject the theme and reset per-run counters. Call once at the top of the app."""
    global _counter
    _counter = itertools.count()
    st.markdown(CSS, unsafe_allow_html=True)


def card(parent=None):
    """A white rounded card (bordered container with a stable CSS hook)."""
    return (parent or st).container(border=True, key=f"card_{next(_counter)}")


def pill(value, digits=2, signed=True):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '<span class="pill flat">–</span>'
    kind = "up" if value > 0 else "down" if value < 0 else "flat"
    arrow = "▲ " if value > 0 else "▼ " if value < 0 else ""
    txt = f"{abs(value):.{digits}%}" if signed else f"{value:.{digits}%}"
    return f'<span class="pill {kind}">{arrow}{txt}</span>'


def sparkline(values, width=96, height=30, color=None):
    """Tiny inline SVG trend line."""
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) < 2:
        return ""
    lo, hi = v.min(), v.max()
    span = hi - lo or 1
    pts = " ".join(f"{i * (width - 2) / (len(v) - 1) + 1:.1f},{height - 2 - (x - lo) / span * (height - 4):.1f}"
                   for i, x in enumerate(v))
    col = color or (GREEN if v[-1] >= v[0] else RED)
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
            f'<polyline fill="none" stroke="{col}" stroke-width="1.8" stroke-linejoin="round" points="{pts}"/></svg>')


_AVATAR_COLORS = ["#0f766e", "#2563eb", "#7c3aed", "#db2777", "#ea580c", "#0891b2", "#4f46e5", "#16a34a", "#b45309"]


def avatar(name):
    words = [w for w in str(name).replace("&", " ").split() if w[0].isalnum()]
    initials = "".join(w[0] for w in words[:2]).upper() or "?"
    color = _AVATAR_COLORS[int(hashlib.md5(str(name).encode()).hexdigest(), 16) % len(_AVATAR_COLORS)]
    return f'<div class="avatar" style="background:{color}14; color:{color}">{initials}</div>'


def list_row(name, sub, price_text, right_html, key, on_click, args):
    """A broker-style list row: avatar, name and ticker, price and change pill, and an open button."""
    c1, c2, c3 = st.columns([6, 3, 0.7], vertical_alignment="center")
    c1.markdown(f'<div class="row">{avatar(name)}<div style="min-width:0"><div class="row-name">{name}</div>'
                f'<div class="row-sub">{sub}</div></div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="row-right"><div class="row-price">{price_text}</div>{right_html}</div>', unsafe_allow_html=True)
    c3.button("›", key=key, type="tertiary", on_click=on_click, args=args, help="Open")
