"""
Fundamental Intelligence  --  Single-Stock Deep Dive
=====================================================
Target user : Equity analyst researching one company
Tabs        : Snapshot | Valuation | Quality | Financials | Ownership
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

DB        = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Database"))
OHLCV_DIR = os.path.join(DB, "OHLCV")

COMPANIES = sorted([
    d for d in os.listdir(DB)
    if os.path.isdir(os.path.join(DB, d)) and d != "OHLCV"
])

NAMES = {
    "ADANIPORTS": "Adani Ports",          "ASIANPAINT": "Asian Paints",
    "AXISBANK":   "Axis Bank",            "BAJFINANCE": "Bajaj Finance",
    "BAJAJFINSV": "Bajaj Finserv",        "BHARTIARTL": "Bharti Airtel",
    "HCLTECH":    "HCL Technologies",     "HDFCBANK":   "HDFC Bank",
    "HINDUNILVR": "Hindustan Unilever",   "ICICIBANK":  "ICICI Bank",
    "INDUSINDBK": "IndusInd Bank",        "INFY":       "Infosys",
    "ITC":        "ITC",                  "JSWSTEEL":   "JSW Steel",
    "KOTAKBANK":  "Kotak Mahindra Bank",  "LT":         "Larsen & Toubro",
    "M&M":        "Mahindra & Mahindra",  "MARUTI":     "Maruti Suzuki",
    "NTPC":       "NTPC",                 "POWERGRID":  "Power Grid",
    "RELIANCE":   "Reliance Industries",  "SBIN":       "State Bank of India",
    "SUNPHARMA":  "Sun Pharma",           "TCS":        "TCS",
    "TATAMOTORS": "Tata Motors",          "TATASTEEL":  "Tata Steel",
    "TECHM":      "Tech Mahindra",        "TITAN":      "Titan",
    "ULTRACEMCO": "UltraTech Cement",     "WIPRO":      "Wipro",
    "ADANIENT":   "Adani Enterprises",    "APOLLOHOSP": "Apollo Hospitals",
    "BAJAJ-AUTO": "Bajaj Auto",           "BPCL":       "BPCL",
    "BRITANNIA":  "Britannia",            "CIPLA":      "Cipla",
    "COALINDIA":  "Coal India",           "DIVISLAB":   "Divi's Laboratories",
    "DRREDDY":    "Dr. Reddy's",          "EICHERMOT":  "Eicher Motors",
    "GRASIM":     "Grasim Industries",    "HDFCLIFE":   "HDFC Life",
    "HEROMOTOCO": "Hero MotoCorp",        "HINDALCO":   "Hindalco",
    "NESTLEIND":  "Nestle India",         "ONGC":       "ONGC",
    "SBILIFE":    "SBI Life Insurance",   "SHRIRAMFIN": "Shriram Finance",
    "TATACONSUMER":"Tata Consumer",       "TRENT":      "Trent",
    "ADANIGREEN": "Adani Green Energy",   "AMBUJACEM":  "Ambuja Cements",
    "AUROPHARMA": "Aurobindo Pharma",     "BANDHANBNK": "Bandhan Bank",
    "BERGEPAINT": "Berger Paints",        "BEL":        "Bharat Electronics",
    "CHOLAFIN":   "Cholamandalam Finance","COLPAL":     "Colgate-Palmolive",
    "DABUR":      "Dabur India",          "DLF":        "DLF",
    "GAIL":       "GAIL India",           "GODREJCP":   "Godrej Consumer",
    "HAVELLS":    "Havells India",        "ICICIPRULI": "ICICI Pru Life",
    "INDUSTOWER": "Indus Towers",         "IRCTC":      "IRCTC",
    "JINDALSTEL": "Jindal Steel",         "LICI":       "LIC of India",
    "LUPIN":      "Lupin",                "MUTHOOTFIN": "Muthoot Finance",
    "NAUKRI":     "Info Edge (Naukri)",   "OFSS":       "Oracle Financial",
    "PERSISTENT": "Persistent Systems",   "PIDILITIND": "Pidilite Industries",
    "SBICARD":    "SBI Cards",            "SIEMENS":    "Siemens India",
    "SRF":        "SRF",                  "TORNTPHARM": "Torrent Pharma",
    "TVSMOTOR":   "TVS Motor",            "VBL":        "Varun Beverages",
    "ZOMATO":     "Zomato (Eternal)",     "ZYDUSLIFE":  "Zydus Life",
    "ADANIENSOL": "Adani Energy Solutions","DMART":     "DMart",
    "MARICO":     "Marico",               "INDHOTEL":   "Indian Hotels (Taj)",
    "BOSCHLTD":   "Bosch India",          "CGPOWER":    "CG Power",
    "POLYCAB":    "Polycab India",        "MOTHERSON":  "Samvardhana Motherson",
    "PAGEIND":    "Page Industries",
}

SECTORS = {
    "Banks":               ["AXISBANK","HDFCBANK","ICICIBANK","INDUSINDBK","KOTAKBANK","SBIN","BANDHANBNK"],
    "Financial Services":  ["BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","SBICARD","SHRIRAMFIN"],
    "Insurance":           ["HDFCLIFE","ICICIPRULI","SBILIFE","LICI"],
    "IT & Technology":     ["TCS","INFY","WIPRO","HCLTECH","TECHM","PERSISTENT","OFSS","NAUKRI"],
    "Oil & Gas":           ["RELIANCE","BPCL","ONGC","GAIL"],
    "Auto":                ["MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT","TVSMOTOR","MOTHERSON"],
    "FMCG":                ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","GODREJCP","MARICO","COLPAL","VBL","TATACONSUMER"],
    "Pharma & Healthcare": ["SUNPHARMA","CIPLA","DRREDDY","DIVISLAB","LUPIN","APOLLOHOSP","AUROPHARMA","ZYDUSLIFE","TORNTPHARM"],
    "Metals & Mining":     ["TATASTEEL","JSWSTEEL","HINDALCO","COALINDIA","JINDALSTEL"],
    "Cement & Materials":  ["ULTRACEMCO","AMBUJACEM","GRASIM"],
    "Capital Goods":       ["LT","SIEMENS","BEL","CGPOWER","BOSCHLTD","HAVELLS","POLYCAB"],
    "Consumer Durables":   ["TITAN","PAGEIND","ASIANPAINT","BERGEPAINT","PIDILITIND"],
    "Energy & Power":      ["NTPC","POWERGRID","ADANIGREEN","ADANIENSOL"],
    "Telecom & Infra":     ["BHARTIARTL","INDUSTOWER","ADANIPORTS"],
    "Retail & Hospitality":["TRENT","DMART","IRCTC","INDHOTEL","ZOMATO"],
    "Chemicals":           ["SRF"],
    "Conglomerates":       ["ADANIENT"],
}

SECTOR_OF = {t: s for s, ts in SECTORS.items() for t in ts}

# ── COLOURS ─────────────────────────────────────────────────────────
BLUE   = "#4C9BE8"
GREEN  = "#27AE60"
RED    = "#E74C3C"
ORANGE = "#F39C12"
PURPLE = "#8E44AD"
GREY   = "#7F8C8D"
TEAL   = "#1ABC9C"

PERIOD_MAP = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 1095, "5Y": 1825, "Max": 99999}

st.set_page_config(
    page_title="Fundamental Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════

def _pct(v):
    if isinstance(v, str) and v.strip().endswith("%"):
        try:
            return float(v.strip().rstrip("%"))
        except ValueError:
            return np.nan
    return v


def load_raw(ticker: str, name: str) -> pd.DataFrame | None:
    path = os.path.join(DB, ticker, f"{name}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    cols = list(df.columns); cols[0] = "Metric"; df.columns = cols
    df["Metric"] = (df["Metric"].astype(str)
                    .str.replace("\xa0", " ", regex=False)
                    .str.replace(" +", "", regex=False)
                    .str.strip().str.rstrip("+").str.strip())
    return df.dropna(subset=["Metric"])


def to_df(ticker: str, name: str) -> pd.DataFrame | None:
    raw = load_raw(ticker, name)
    if raw is None:
        return None
    raw = raw.set_index("Metric")
    raw = raw.apply(lambda col: col.map(_pct))
    raw = raw.apply(pd.to_numeric, errors="coerce")
    return raw.T


def get_cagr(ticker: str, tbl: str, period: str):
    raw = load_raw(ticker, tbl)
    if raw is None:
        return None
    raw.columns = ["Period", "Value"]
    row = raw[raw["Period"].str.strip() == period]
    if row.empty:
        return None
    return _pct(row["Value"].iloc[0])


@st.cache_data(ttl=3600)
def company_data(ticker: str) -> dict:
    return {n: to_df(ticker, n) for n in [
        "quarterly_pl", "annual_pl", "balance_sheet", "cash_flow",
        "ratios", "shareholding_qtr", "shareholding_annual",
        "sales_growth", "profit_growth", "price_cagr", "roe_summary",
    ]}


@st.cache_data(ttl=3600)
def load_ohlcv(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(OHLCV_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _parse_screener_dates(idx):
    dates = []
    for s in idx:
        try:
            d = pd.to_datetime(str(s).strip(), format="%b %Y")
            dates.append(d + pd.offsets.MonthEnd(0))
        except Exception:
            dates.append(pd.NaT)
    return dates


def compute_pe(ohlcv: pd.DataFrame, apl: pd.DataFrame) -> dict | None:
    """Compute daily P/E and mean/SD bands. Returns dict or None."""
    if ohlcv is None or apl is None or "EPS in Rs" not in apl.columns:
        return None
    apl2 = apl.copy()
    apl2.index = pd.DatetimeIndex(_parse_screener_dates(apl2.index))
    apl2 = apl2[apl2.index.notna()].sort_index()
    eps = apl2["EPS in Rs"].dropna()
    if eps.empty:
        return None
    close = ohlcv["Close"].sort_index()
    start = max(close.index[0], eps.index[0])
    all_dates = pd.date_range(start, close.index[-1], freq="D")
    eps_daily   = eps.reindex(all_dates).ffill()
    close_daily = close.reindex(all_dates).ffill()
    mask = (eps_daily > 0) & close_daily.notna()
    pe = (close_daily[mask] / eps_daily[mask]).replace([np.inf, -np.inf], np.nan).dropna()
    if pe.empty:
        return None
    mean_pe = pe.mean()
    std_pe  = pe.std()
    return {
        "pe":    pe,
        "eps":   eps_daily,
        "close": close_daily,
        "mean":  round(mean_pe, 1),
        "std":   std_pe,
        "bands": {
            "+2SD": round(mean_pe + 2 * std_pe, 1),
            "+1SD": round(mean_pe + std_pe, 1),
            "Mean": round(mean_pe, 1),
            "-1SD": round(max(mean_pe - std_pe, 0), 1),
            "-2SD": round(max(mean_pe - 2 * std_pe, 0), 1),
        },
    }


# ── CHART STYLE ─────────────────────────────────────────────────────
def _style(fig, *, yt="", yt2="", height=400, legend=True, barmode=None):
    layout = dict(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(15,22,32,0.92)", font_size=12,
                        bordercolor="rgba(255,255,255,0.10)", namelength=-1),
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(showgrid=False, zeroline=False, title="",
                   tickfont=dict(size=11), showline=True,
                   linecolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.10)", zeroline=False,
                   title=yt, tickfont=dict(size=11)),
        legend=dict(orientation="h", y=1.06, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)) if legend else dict(visible=False),
        margin=dict(t=40, b=30, l=10, r=10),
    )
    if not legend:
        layout["showlegend"] = False
    if barmode:
        layout["barmode"] = barmode
    fig.update_layout(**layout)
    if yt2:
        fig.update_layout(yaxis2=dict(gridcolor="rgba(0,0,0,0)", zeroline=False,
                                      title=yt2, tickfont=dict(size=11)))
    return fig


# ── KPI CARDS ────────────────────────────────────────────────────────
def _kpi_html(cards: list) -> str:
    html = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px;">'
    for label, value, delta, is_pos in cards:
        dc    = GREEN if is_pos else RED
        arrow = "&#8593;" if is_pos else "&#8595;"
        d_html = (f'<div style="font-size:11px;color:{dc};margin-top:6px;">'
                  f'{arrow} {delta}</div>') if delta else ""
        html += f"""
        <div style="flex:1;min-width:120px;background:rgba(128,128,128,0.06);
                    border:1px solid rgba(128,128,128,0.14);border-radius:10px;
                    padding:16px 12px;text-align:center;">
            <div style="font-size:10px;color:#888;text-transform:uppercase;
                        letter-spacing:0.12em;margin-bottom:6px;">{label}</div>
            <div style="font-size:1.2rem;font-weight:700;line-height:1.2;
                        word-break:break-word;">{value}</div>
            {d_html}
        </div>"""
    return html + "</div>"


def _fmt_cr(v: float) -> str:
    if abs(v) >= 100_000:
        return f"Rs {v/100000:.2f}L Cr"
    return f"Rs {v:,.0f} Cr"


def _delta_pct(v, p):
    if p is None or pd.isna(p) or p == 0:
        return "", True
    d = (v - p) / abs(p) * 100
    return f"{d:.1f}% YoY", d >= 0


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("Fundamental Intelligence")
    st.caption("Single-Stock Deep Dive")
    st.divider()

    sector_opts = ["All Sectors"] + sorted(SECTORS.keys())
    sel_sector  = st.selectbox("Filter by Sector", sector_opts)

    if sel_sector == "All Sectors":
        filtered = COMPANIES
    else:
        filtered = [c for c in COMPANIES if SECTOR_OF.get(c) == sel_sector]
        if not filtered:
            filtered = COMPANIES

    def_idx = filtered.index("RELIANCE") if "RELIANCE" in filtered else 0
    ticker  = st.selectbox(
        "Company",
        filtered,
        format_func=lambda x: f"{x}  —  {NAMES.get(x, x)}",
        index=def_idx,
    )

    st.divider()
    sector_label = SECTOR_OF.get(ticker, "Other")
    st.markdown(f"**Sector** &nbsp; {sector_label}")
    has_ohlcv = os.path.exists(os.path.join(OHLCV_DIR, f"{ticker}.parquet"))
    px_badge  = "Price data: Available" if has_ohlcv else "Price data: Not available"
    st.caption(px_badge)
    st.caption(f"{len(COMPANIES)} companies in database")

# ── Load data ───────────────────────────────────────────────────────
D     = company_data(ticker)
name  = NAMES.get(ticker, ticker)
ohlcv = load_ohlcv(ticker)
pe_data = compute_pe(ohlcv, D["annual_pl"])

# ═══════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════
tab_snap, tab_val, tab_qual, tab_fin, tab_own = st.tabs([
    "  Snapshot  ",
    "  Valuation  ",
    "  Quality  ",
    "  Financials  ",
    "  Ownership  ",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — SNAPSHOT
# ═══════════════════════════════════════════════════════════════════
with tab_snap:
    st.header(name)
    st.caption(f"{ticker}  |  {sector_label}")

    # ── Price Chart ────────────────────────────────────────────────
    if ohlcv is not None:
        period_key = st.radio(
            "Period", list(PERIOD_MAP.keys()),
            index=3, horizontal=True, key="snap_period",
        )
        days   = PERIOD_MAP[period_key]
        cutoff = ohlcv.index[-1] - pd.Timedelta(days=days)
        px_df  = ohlcv[ohlcv.index >= cutoff]

        fig_px = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.78, 0.22], vertical_spacing=0.02,
        )
        fig_px.add_trace(go.Scatter(
            x=px_df.index, y=px_df["Close"], name="Close",
            mode="lines", line=dict(color=BLUE, width=1.5),
            hovertemplate="Rs %{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        vol_colors = [GREEN if c >= o else RED
                      for c, o in zip(px_df["Close"], px_df["Open"])]
        fig_px.add_trace(go.Bar(
            x=px_df.index, y=px_df["Volume"], name="Volume",
            marker_color=vol_colors, opacity=0.6,
            hovertemplate="%{y:,.0f}<extra></extra>",
        ), row=2, col=1)
        fig_px.update_layout(
            height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified", showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            font=dict(family="Inter, sans-serif", size=12),
            xaxis=dict(showgrid=False, zeroline=False, showline=True,
                       linecolor="rgba(128,128,128,0.2)"),
            xaxis2=dict(showgrid=False, zeroline=False, showline=True,
                        linecolor="rgba(128,128,128,0.2)"),
            yaxis=dict(gridcolor="rgba(128,128,128,0.10)", zeroline=False, title="Price (Rs)"),
            yaxis2=dict(gridcolor="rgba(128,128,128,0.10)", zeroline=False, title="Volume"),
            hoverlabel=dict(bgcolor="rgba(15,22,32,0.92)", font_size=12,
                            bordercolor="rgba(255,255,255,0.10)"),
        )
        # 52-week Hi/Lo annotations
        hi52 = ohlcv[ohlcv.index >= ohlcv.index[-1] - pd.Timedelta(days=365)]["High"].max()
        lo52 = ohlcv[ohlcv.index >= ohlcv.index[-1] - pd.Timedelta(days=365)]["Low"].min()
        curr = ohlcv["Close"].iloc[-1]
        fig_px.add_hline(y=hi52, line_dash="dot", line_color="rgba(39,174,96,0.4)",
                         annotation_text=f"52W H  Rs {hi52:,.0f}",
                         annotation_font_size=10, annotation_font_color=GREEN, row=1, col=1)
        fig_px.add_hline(y=lo52, line_dash="dot", line_color="rgba(231,76,60,0.4)",
                         annotation_text=f"52W L  Rs {lo52:,.0f}",
                         annotation_font_size=10, annotation_font_color=RED, row=1, col=1)
        st.plotly_chart(fig_px, use_container_width=True)
    else:
        st.info("Price chart unavailable — OHLCV not fetched for this ticker yet.")

    st.divider()

    # ── KPI Cards ──────────────────────────────────────────────────
    apl = D["annual_pl"]
    rat = D["ratios"]
    cf  = D["cash_flow"]
    cards = []

    if apl is not None and len(apl) >= 2:
        latest, prev = apl.iloc[-1], apl.iloc[-2]
        for col, lbl, fmt in [
            ("Sales",      "Revenue",    "cr"),
            ("Net Profit", "Net Profit", "cr"),
            ("OPM %",      "OPM",        "pct"),
            ("EPS in Rs",  "EPS",        "num"),
        ]:
            v = latest.get(col)
            p = prev.get(col)
            if v is None or pd.isna(v):
                continue
            if fmt == "cr":
                disp = _fmt_cr(v); d, pos = _delta_pct(v, p)
            elif fmt == "pct":
                disp = f"{v:.1f}%"
                d = f"{v-p:+.1f}pp YoY" if p is not None and not pd.isna(p) else ""
                pos = v >= (p or 0)
            else:
                disp = f"Rs {v:.2f}"; d, pos = _delta_pct(v, p)
            cards.append((lbl, disp, d, pos))

    if rat is not None and "ROCE %" in rat.columns:
        rv = rat["ROCE %"].dropna()
        if len(rv) >= 2:
            v, p = rv.iloc[-1], rv.iloc[-2]
            cards.append(("ROCE", f"{v:.1f}%", f"{v-p:+.1f}pp YoY", v >= p))

    if cf is not None and "Free Cash Flow" in cf.columns:
        fcf = cf["Free Cash Flow"].dropna()
        if len(fcf) >= 2:
            v, p = fcf.iloc[-1], fcf.iloc[-2]
            d, pos = _delta_pct(v, p)
            cards.append(("Free Cash Flow", _fmt_cr(v), d, pos))

    if cards:
        st.markdown(_kpi_html(cards), unsafe_allow_html=True)

    st.divider()

    # ── Growth Snapshot ────────────────────────────────────────────
    st.subheader("Growth Snapshot")
    growth_cols = [
        ("Sales CAGR",  "sales_growth"),
        ("Profit CAGR", "profit_growth"),
        ("Price CAGR",  "price_cagr"),
        ("ROE",         "roe_summary"),
    ]
    periods = [("10 Years:", "10Y"), ("5 Years:", "5Y"), ("3 Years:", "3Y")]
    hdr = "".join(
        f'<th style="padding:10px 20px;text-align:center;font-size:11px;'
        f'text-transform:uppercase;letter-spacing:0.10em;color:#888;'
        f'border-bottom:1px solid rgba(128,128,128,0.18);">{lbl}</th>'
        for lbl, _ in growth_cols
    )
    rows_html = ""
    for pk, pl in periods:
        cells = (f'<td style="padding:12px 20px;font-size:12px;color:#666;'
                 f'font-weight:600;letter-spacing:0.05em;">{pl}</td>')
        for _, tbl in growth_cols:
            v = get_cagr(ticker, tbl, pk)
            val = f"{v:.0f}%" if v is not None else "—"
            c = GREEN if (v is not None and v > 0) else (RED if (v is not None and v < 0) else "#888")
            cells += (f'<td style="padding:12px 20px;text-align:center;font-size:20px;'
                      f'font-weight:700;color:{c};">{val}</td>')
        rows_html += f"<tr>{cells}</tr>"
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr><th style="border-bottom:1px solid rgba(128,128,128,0.18);'
        f'width:50px;"></th>{hdr}</tr></thead><tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True,
    )

    # ── Historical Trend Sparklines ────────────────────────────────
    with st.expander("Historical Trends (12-Year)"):
        if apl is not None:
            s1, s2, s3 = st.columns(3)
            for col_obj, col, title, color, fmt in [
                (s1, "Sales",      "Revenue",    BLUE,  "cr"),
                (s2, "Net Profit", "Net Profit", GREEN, "cr"),
            ]:
                if col in apl.columns:
                    ht = "Rs %{y:,.0f} Cr<extra></extra>"
                    fig = go.Figure(go.Bar(
                        x=apl.index, y=apl[col], marker_color=color,
                        opacity=0.85, hovertemplate=ht,
                    ))
                    _style(fig, height=220, legend=False)
                    fig.update_layout(title=dict(text=title, font_size=13),
                                      margin=dict(t=36, b=20))
                    col_obj.plotly_chart(fig, use_container_width=True)
            if rat is not None and "ROCE %" in rat.columns:
                fig_r = go.Figure(go.Scatter(
                    x=rat.index, y=rat["ROCE %"], mode="lines+markers",
                    line=dict(color=ORANGE, width=2), marker=dict(size=5),
                    hovertemplate="%{y:.1f}%<extra></extra>",
                ))
                _style(fig_r, yt="ROCE %", height=220, legend=False)
                fig_r.update_layout(title=dict(text="ROCE %", font_size=13),
                                    margin=dict(t=36, b=20))
                s3.plotly_chart(fig_r, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — VALUATION
# ═══════════════════════════════════════════════════════════════════
with tab_val:
    st.header(f"{name}  —  Valuation")

    if pe_data is None:
        if ohlcv is None:
            st.warning("Price data (OHLCV) not available for this ticker. Run master_ingest.py to fetch it.")
        else:
            st.warning("Could not compute P/E — EPS data missing or all negative.")
    else:
        # ── P/E Summary Cards ──────────────────────────────────────
        current_pe = round(pe_data["pe"].iloc[-1], 1) if not pe_data["pe"].empty else None
        pe_cards = [
            ("Current P/E",  f"{current_pe}x" if current_pe else "—", "", True),
            ("Mean P/E",     f"{pe_data['mean']}x", "", True),
            ("Median P/E",   f"{round(pe_data['pe'].median(), 1)}x", "", True),
            ("Min P/E",      f"{round(pe_data['pe'].min(), 1)}x", "", True),
            ("Max P/E",      f"{round(pe_data['pe'].max(), 1)}x", "", True),
        ]
        if current_pe is not None:
            vs_mean = current_pe - pe_data["mean"]
            pe_cards[0] = (
                "Current P/E",
                f"{current_pe}x",
                f"{vs_mean:+.1f}x vs mean",
                vs_mean <= 0,
            )
        st.markdown(_kpi_html(pe_cards), unsafe_allow_html=True)
        st.divider()

        # ── P/E Band Chart ─────────────────────────────────────────
        st.subheader("P/E Band Chart")
        pe_period = st.radio(
            "History", ["3Y", "5Y", "10Y", "Max"],
            index=2, horizontal=True, key="val_period",
        )
        p_days  = {"3Y": 1095, "5Y": 1825, "10Y": 3650, "Max": 99999}[pe_period]
        cutoff  = pe_data["close"].index[-1] - pd.Timedelta(days=p_days)

        close_s = pe_data["close"][pe_data["close"].index >= cutoff]
        eps_s   = pe_data["eps"][pe_data["eps"].index >= cutoff]

        band_colors = {
            "+2SD": "rgba(231,76,60,0.7)",
            "+1SD": "rgba(243,156,18,0.7)",
            "Mean": "rgba(128,128,128,0.7)",
            "-1SD": "rgba(39,174,96,0.7)",
            "-2SD": "rgba(39,174,96,0.4)",
        }

        fig_pe = go.Figure()
        # Band lines
        for band_label, band_pe in pe_data["bands"].items():
            if band_pe <= 0:
                continue
            band_price = band_pe * eps_s.reindex(close_s.index).ffill()
            fig_pe.add_trace(go.Scatter(
                x=band_price.index, y=band_price,
                name=f"{band_label}  ({band_pe}x)",
                mode="lines",
                line=dict(color=band_colors[band_label], width=1, dash="dot"),
                hovertemplate=f"Rs %{{y:,.0f}} ({band_label})<extra></extra>",
            ))
        # Actual price on top
        fig_pe.add_trace(go.Scatter(
            x=close_s.index, y=close_s,
            name="Price", mode="lines",
            line=dict(color=BLUE, width=2),
            hovertemplate="Rs %{y:,.2f}<extra></extra>",
        ))
        _style(fig_pe, yt="Price (Rs)", height=460)
        st.plotly_chart(fig_pe, use_container_width=True)

        # ── Price vs EPS ───────────────────────────────────────────
        with st.expander("Price vs EPS Overlay"):
            apl = D["annual_pl"]
            if apl is not None and "EPS in Rs" in apl.columns:
                apl2 = apl.copy()
                apl2.index = pd.DatetimeIndex(_parse_screener_dates(apl2.index))
                apl2 = apl2[apl2.index.notna()].sort_index()
                eps_ann = apl2["EPS in Rs"].dropna()

                fig_eps = make_subplots(specs=[[{"secondary_y": True}]])
                fig_eps.add_trace(go.Scatter(
                    x=close_s.index, y=close_s, name="Price",
                    mode="lines", line=dict(color=BLUE, width=1.5),
                    hovertemplate="Rs %{y:,.2f}<extra></extra>",
                ), secondary_y=False)
                fig_eps.add_trace(go.Bar(
                    x=eps_ann.index, y=eps_ann, name="EPS (Rs)",
                    marker_color=GREEN, opacity=0.7,
                    hovertemplate="EPS: Rs %{y:.2f}<extra></extra>",
                ), secondary_y=True)
                _style(fig_eps, yt="Price (Rs)", yt2="EPS (Rs)", height=360)
                fig_eps.update_yaxes(showgrid=False, secondary_y=True)
                st.plotly_chart(fig_eps, use_container_width=True)

        # ── Historical P/E Distribution ────────────────────────────
        with st.expander("P/E Distribution (Histogram)"):
            pe_hist = pe_data["pe"][pe_data["pe"].index >= cutoff]
            fig_hist = go.Figure(go.Histogram(
                x=pe_hist, nbinsx=40,
                marker_color=BLUE, opacity=0.75,
                hovertemplate="P/E: %{x:.1f}  Count: %{y}<extra></extra>",
            ))
            fig_hist.add_vline(x=pe_data["mean"], line_dash="dash",
                               line_color=GREY, annotation_text=f"Mean {pe_data['mean']}x",
                               annotation_font_size=11)
            if current_pe:
                fig_hist.add_vline(x=current_pe, line_dash="solid",
                                   line_color=ORANGE, annotation_text=f"Now {current_pe}x",
                                   annotation_font_size=11)
            _style(fig_hist, yt="Frequency", height=300, legend=False)
            st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — QUALITY
# ═══════════════════════════════════════════════════════════════════
with tab_qual:
    st.header(f"{name}  —  Business Quality")

    apl = D["annual_pl"]
    rat = D["ratios"]
    cf  = D["cash_flow"]
    bs  = D["balance_sheet"]

    # ── Margins & Returns — side by side ────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Margin Trends")
        if apl is not None:
            pl2 = apl.copy()
            if "Net Profit" in pl2.columns and "Sales" in pl2.columns:
                pl2["NPM %"] = (pl2["Net Profit"] / pl2["Sales"] * 100).round(1)
            margin_cols = [(c, col) for c, col in [
                (ORANGE, "OPM %"), (GREEN, "NPM %")
            ] if col in pl2.columns]
            if margin_cols:
                fig_m = go.Figure()
                for color, col in margin_cols:
                    fig_m.add_trace(go.Scatter(
                        x=pl2.index, y=pl2[col], name=col,
                        mode="lines+markers",
                        line=dict(color=color, width=2), marker=dict(size=5),
                        hovertemplate="%{y:.1f}%<extra></extra>",
                    ))
                _style(fig_m, yt="%", height=300)
                st.plotly_chart(fig_m, use_container_width=True)

    with c2:
        st.subheader("Returns (ROCE)")
        if rat is not None and "ROCE %" in rat.columns:
            roce = rat["ROCE %"].dropna()
            fig_r = go.Figure(go.Scatter(
                x=roce.index, y=roce, name="ROCE %",
                mode="lines+markers",
                line=dict(color=BLUE, width=2.5), marker=dict(size=6),
                fill="tozeroy", fillcolor="rgba(76,155,232,0.08)",
                hovertemplate="%{y:.1f}%<extra></extra>",
            ))
            _style(fig_r, yt="ROCE %", height=300, legend=False)
            st.plotly_chart(fig_r, use_container_width=True)

    st.divider()

    # ── FCF & Earnings Quality ─────────────────────────────────────
    st.subheader("Cash Generation")
    fc1, fc2 = st.columns(2)

    with fc1:
        if cf is not None and "Free Cash Flow" in cf.columns:
            fcf    = cf["Free Cash Flow"].dropna()
            colors = [GREEN if v >= 0 else RED for v in fcf]
            fig_f  = go.Figure(go.Bar(
                x=fcf.index, y=fcf, marker_color=colors, name="FCF",
                hovertemplate="Rs %{y:,.0f} Cr<extra></extra>",
            ))
            fig_f.add_hline(y=0, line_dash="dot",
                            line_color="rgba(128,128,128,0.4)", line_width=1)
            _style(fig_f, yt="Rs Crores", height=300, legend=False)
            fig_f.update_layout(title=dict(text="Free Cash Flow", font_size=13))
            st.plotly_chart(fig_f, use_container_width=True)

    with fc2:
        if cf is not None and apl is not None and \
                "Cash from Operating Activity" in cf.columns and "Net Profit" in apl.columns:
            common = cf.index.intersection(apl.index)
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=list(common), y=list(apl.loc[common, "Net Profit"]),
                name="Net Profit", mode="lines+markers",
                line=dict(color=BLUE, width=2), marker=dict(size=5),
                hovertemplate="Rs %{y:,.0f} Cr<extra></extra>",
            ))
            fig_eq.add_trace(go.Scatter(
                x=list(common), y=list(cf.loc[common, "Cash from Operating Activity"]),
                name="Operating CF", mode="lines+markers",
                line=dict(color=GREEN, width=2), marker=dict(size=5),
                hovertemplate="Rs %{y:,.0f} Cr<extra></extra>",
            ))
            _style(fig_eq, yt="Rs Crores", height=300)
            fig_eq.update_layout(title=dict(text="OCF vs Net Profit", font_size=13))
            st.plotly_chart(fig_eq, use_container_width=True)

    # ── Working Capital Detail ─────────────────────────────────────
    with st.expander("Working Capital Detail"):
        if rat is not None:
            wc1, wc2 = st.columns(2)
            wc_cols = [c for c in ["Debtor Days", "Inventory Days", "Days Payable"] if c in rat.columns]
            with wc1:
                if wc_cols:
                    fig_wc = go.Figure()
                    for col, color in zip(wc_cols, [BLUE, ORANGE, RED]):
                        fig_wc.add_trace(go.Scatter(
                            x=rat.index, y=rat[col], name=col,
                            mode="lines+markers",
                            line=dict(width=2, color=color), marker=dict(size=5),
                            hovertemplate="%{y:.0f} days<extra></extra>",
                        ))
                    _style(fig_wc, yt="Days", height=300)
                    fig_wc.update_layout(title=dict(text="Debtor / Inventory / Payable Days", font_size=13))
                    st.plotly_chart(fig_wc, use_container_width=True)

            with wc2:
                if "Cash Conversion Cycle" in rat.columns:
                    ccc    = rat["Cash Conversion Cycle"].dropna()
                    colors = [RED if v > 0 else GREEN for v in ccc]
                    fig_ccc = go.Figure(go.Bar(
                        x=ccc.index, y=ccc, marker_color=colors, name="CCC",
                        hovertemplate="%{y:.0f} days<extra></extra>",
                    ))
                    fig_ccc.add_hline(y=0, line_dash="dot",
                                      line_color="rgba(128,128,128,0.4)", line_width=1)
                    _style(fig_ccc, yt="Days", height=300, legend=False)
                    fig_ccc.update_layout(title=dict(text="Cash Conversion Cycle", font_size=13))
                    st.plotly_chart(fig_ccc, use_container_width=True)

            if "Working Capital Days" in rat.columns:
                wcd    = rat["Working Capital Days"].dropna()
                colors = [RED if v > 0 else GREEN for v in wcd]
                fig_wcd = go.Figure(go.Bar(
                    x=wcd.index, y=wcd, marker_color=colors, name="WC Days",
                    hovertemplate="%{y:.0f} days<extra></extra>",
                ))
                fig_wcd.add_hline(y=0, line_dash="dot",
                                  line_color="rgba(128,128,128,0.4)", line_width=1)
                _style(fig_wcd, yt="Days", height=260, legend=False)
                fig_wcd.update_layout(title=dict(text="Net Working Capital Days", font_size=13))
                st.plotly_chart(fig_wcd, use_container_width=True)

    # ── DuPont Analysis ────────────────────────────────────────────
    with st.expander("DuPont Analysis  (ROE Decomposition)"):
        if apl is not None and bs is not None:
            try:
                common = apl.index.intersection(bs.index)
                dup = pd.DataFrame(index=common)
                if "Net Profit" in apl.columns and "Sales" in apl.columns:
                    dup["Net Profit Margin %"] = (apl.loc[common, "Net Profit"] /
                                                   apl.loc[common, "Sales"] * 100).round(2)
                if "Total Assets" in bs.columns and "Sales" in apl.columns:
                    dup["Asset Turnover"] = (apl.loc[common, "Sales"] /
                                             bs.loc[common, "Total Assets"]).round(3)
                eq_cols = [c for c in ["Equity Capital", "Reserves"] if c in bs.columns]
                if eq_cols and "Total Assets" in bs.columns:
                    equity = bs.loc[common, eq_cols].sum(axis=1).replace(0, np.nan)
                    dup["Equity Multiplier"] = (bs.loc[common, "Total Assets"] / equity).round(2)
                    dup["ROE (DuPont) %"] = (dup["Net Profit Margin %"] / 100 *
                                              dup["Asset Turnover"] *
                                              dup["Equity Multiplier"] * 100).round(2)

                dup = dup.dropna(how="all")
                if not dup.empty:
                    d1, d2, d3, d4 = st.columns(4)
                    for col_obj, col, color in [
                        (d1, "Net Profit Margin %", BLUE),
                        (d2, "Asset Turnover",      ORANGE),
                        (d3, "Equity Multiplier",   PURPLE),
                        (d4, "ROE (DuPont) %",      GREEN),
                    ]:
                        if col not in dup.columns:
                            continue
                        fig_d = go.Figure(go.Scatter(
                            x=dup.index, y=dup[col], mode="lines+markers",
                            line=dict(color=color, width=2), marker=dict(size=5),
                            hovertemplate="%{y:.2f}<extra></extra>",
                        ))
                        _style(fig_d, height=220, legend=False)
                        fig_d.update_layout(title=dict(text=col, font_size=12),
                                            margin=dict(t=36, b=20))
                        col_obj.plotly_chart(fig_d, use_container_width=True)
            except Exception:
                st.caption("DuPont data unavailable for this company.")


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — FINANCIALS
# ═══════════════════════════════════════════════════════════════════
with tab_fin:
    st.header(f"{name}  —  Financial Statements")
    sub_pl, sub_bs, sub_cf = st.tabs(["  P&L  ", "  Balance Sheet  ", "  Cash Flow  "])

    # ── P&L ────────────────────────────────────────────────────────
    with sub_pl:
        apl = D["annual_pl"]
        qpl = D["quarterly_pl"]
        view = st.radio("Period", ["Annual", "Quarterly"], horizontal=True, key="pl_view")
        pl   = apl if view == "Annual" else qpl

        if pl is not None:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            if "Sales" in pl.columns:
                fig.add_trace(go.Bar(
                    x=pl.index, y=pl["Sales"], name="Revenue",
                    marker_color=BLUE, opacity=0.85,
                    hovertemplate="Rs %{y:,.0f} Cr<extra></extra>",
                ), secondary_y=False)
            if "Net Profit" in pl.columns:
                fig.add_trace(go.Bar(
                    x=pl.index, y=pl["Net Profit"], name="Net Profit",
                    marker_color=GREEN, opacity=0.85,
                    hovertemplate="Rs %{y:,.0f} Cr<extra></extra>",
                ), secondary_y=False)
            if "OPM %" in pl.columns:
                fig.add_trace(go.Scatter(
                    x=pl.index, y=pl["OPM %"], name="OPM %",
                    mode="lines+markers", line=dict(color=ORANGE, width=2),
                    marker=dict(size=5),
                    hovertemplate="%{y:.1f}%<extra></extra>",
                ), secondary_y=True)
            _style(fig, yt="Rs Crores", yt2="OPM %", height=420, barmode="group")
            fig.update_yaxes(showgrid=True,  gridcolor="rgba(128,128,128,0.10)", secondary_y=False)
            fig.update_yaxes(showgrid=False, secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # Margins
            pl2 = pl.copy()
            if "Net Profit" in pl2.columns and "Sales" in pl2.columns:
                pl2["NPM %"] = (pl2["Net Profit"] / pl2["Sales"] * 100).round(1)
            mcols = [c for c in ["OPM %", "NPM %"] if c in pl2.columns]
            if mcols:
                with st.expander("Margins"):
                    fig2 = go.Figure()
                    for col, color in zip(mcols, [ORANGE, GREEN]):
                        fig2.add_trace(go.Scatter(
                            x=pl2.index, y=pl2[col], name=col,
                            mode="lines+markers", line=dict(width=2, color=color),
                            marker=dict(size=5),
                            hovertemplate="%{y:.1f}%<extra></extra>",
                        ))
                    _style(fig2, yt="%", height=300)
                    st.plotly_chart(fig2, use_container_width=True)

            # EPS
            if "EPS in Rs" in pl.columns:
                with st.expander("Earnings Per Share"):
                    fig3 = go.Figure(go.Bar(
                        x=pl.index, y=pl["EPS in Rs"], marker_color=BLUE, opacity=0.85,
                        hovertemplate="Rs %{y:.2f}<extra></extra>",
                    ))
                    _style(fig3, yt="Rs per share", height=280, legend=False)
                    st.plotly_chart(fig3, use_container_width=True)

            # YoY Growth (annual only)
            if view == "Annual" and apl is not None and \
                    "Sales" in apl.columns and "Net Profit" in apl.columns:
                with st.expander("YoY Growth"):
                    yoy = pd.DataFrame({
                        "Revenue":    apl["Sales"].pct_change() * 100,
                        "Net Profit": apl["Net Profit"].pct_change() * 100,
                    }).dropna().round(1)
                    fig4 = go.Figure()
                    for col, color in zip(["Revenue", "Net Profit"], [BLUE, GREEN]):
                        fig4.add_trace(go.Bar(
                            x=yoy.index, y=yoy[col], name=col,
                            marker_color=color, opacity=0.85,
                            hovertemplate="%{y:.1f}%<extra></extra>",
                        ))
                    fig4.add_hline(y=0, line_dash="dot",
                                   line_color="rgba(128,128,128,0.4)", line_width=1)
                    _style(fig4, yt="%", height=320, barmode="group")
                    st.plotly_chart(fig4, use_container_width=True)

    # ── Balance Sheet ───────────────────────────────────────────────
    with sub_bs:
        bs = D["balance_sheet"]
        if bs is not None:
            asset_cols = [c for c in ["Fixed Assets", "CWIP", "Investments", "Other Assets"]
                          if c in bs.columns]
            if asset_cols:
                fig = go.Figure()
                for col, color in zip(asset_cols, [BLUE, ORANGE, GREEN, PURPLE]):
                    fig.add_trace(go.Bar(
                        x=bs.index, y=bs[col], name=col, marker_color=color,
                        hovertemplate="Rs %{y:,.0f} Cr<extra></extra>",
                    ))
                _style(fig, yt="Rs Crores", height=380, barmode="stack")
                fig.update_layout(title=dict(text="Asset Composition", font_size=14))
                st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                eq_cols = [c for c in ["Equity Capital", "Reserves"] if c in bs.columns]
                equity  = bs[eq_cols].sum(axis=1)
                fig2    = go.Figure()
                fig2.add_trace(go.Bar(x=bs.index, y=equity, name="Equity + Reserves",
                                      marker_color=GREEN,
                                      hovertemplate="Rs %{y:,.0f} Cr<extra></extra>"))
                if "Borrowings" in bs.columns:
                    fig2.add_trace(go.Bar(x=bs.index, y=bs["Borrowings"], name="Borrowings",
                                          marker_color=RED,
                                          hovertemplate="Rs %{y:,.0f} Cr<extra></extra>"))
                _style(fig2, yt="Rs Crores", height=340, barmode="group")
                fig2.update_layout(title=dict(text="Debt vs Equity", font_size=14))
                st.plotly_chart(fig2, use_container_width=True)

            with c2:
                if "Borrowings" in bs.columns:
                    de = (bs["Borrowings"] / equity.replace(0, np.nan)).round(2)
                    fig3 = go.Figure(go.Scatter(
                        x=de.index, y=de, mode="lines+markers",
                        line=dict(color=RED, width=2), marker=dict(size=5),
                        hovertemplate="%{y:.2f}x<extra></extra>", name="D/E",
                    ))
                    fig3.add_hline(y=1, line_dash="dot",
                                   line_color="rgba(128,128,128,0.4)", line_width=1,
                                   annotation_text="1x", annotation_font_size=11,
                                   annotation_font_color="#888")
                    _style(fig3, yt="D/E Ratio", height=340, legend=False)
                    fig3.update_layout(title=dict(text="Debt / Equity Ratio", font_size=14))
                    st.plotly_chart(fig3, use_container_width=True)

            with st.expander("CWIP & Total Assets"):
                b1, b2 = st.columns(2)
                if "CWIP" in bs.columns:
                    fig4 = go.Figure(go.Bar(x=bs.index, y=bs["CWIP"],
                                            marker_color=ORANGE, name="CWIP",
                                            hovertemplate="Rs %{y:,.0f} Cr<extra></extra>"))
                    _style(fig4, yt="Rs Crores", height=260, legend=False)
                    fig4.update_layout(title=dict(text="Capital Work-in-Progress", font_size=13))
                    b1.plotly_chart(fig4, use_container_width=True)
                if "Total Assets" in bs.columns:
                    fig5 = go.Figure(go.Scatter(
                        x=bs.index, y=bs["Total Assets"], mode="lines",
                        fill="tozeroy", line=dict(color=BLUE, width=2),
                        fillcolor="rgba(76,155,232,0.10)", name="Total Assets",
                        hovertemplate="Rs %{y:,.0f} Cr<extra></extra>",
                    ))
                    _style(fig5, yt="Rs Crores", height=260, legend=False)
                    fig5.update_layout(title=dict(text="Total Assets", font_size=13))
                    b2.plotly_chart(fig5, use_container_width=True)

    # ── Cash Flow ───────────────────────────────────────────────────
    with sub_cf:
        cf  = D["cash_flow"]
        apl = D["annual_pl"]
        if cf is not None:
            fig = go.Figure()
            for col, color, short in [
                ("Cash from Operating Activity", GREEN,  "Operating (OCF)"),
                ("Cash from Investing Activity", RED,    "Investing (ICF)"),
                ("Cash from Financing Activity", BLUE,   "Financing (CFF)"),
            ]:
                if col in cf.columns:
                    fig.add_trace(go.Bar(x=cf.index, y=cf[col], name=short,
                                         marker_color=color, opacity=0.85,
                                         hovertemplate="Rs %{y:,.0f} Cr<extra></extra>"))
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.4)", line_width=1)
            _style(fig, yt="Rs Crores", height=400, barmode="group")
            st.plotly_chart(fig, use_container_width=True)

            cf1, cf2 = st.columns(2)
            with cf1:
                if "Free Cash Flow" in cf.columns:
                    fcf    = cf["Free Cash Flow"].dropna()
                    colors = [GREEN if v >= 0 else RED for v in fcf]
                    fig2   = go.Figure(go.Bar(x=fcf.index, y=fcf, marker_color=colors, name="FCF",
                                              hovertemplate="Rs %{y:,.0f} Cr<extra></extra>"))
                    fig2.add_hline(y=0, line_dash="dot",
                                   line_color="rgba(128,128,128,0.4)", line_width=1)
                    _style(fig2, yt="Rs Crores", height=300, legend=False)
                    fig2.update_layout(title=dict(text="Free Cash Flow", font_size=13))
                    cf1.plotly_chart(fig2, use_container_width=True)

            with cf2:
                col_name = "CFO/OP" if "CFO/OP" in cf.columns else None
                if col_name:
                    cfoop = cf[col_name].dropna()
                    bar_colors = [GREEN if v >= 100 else ORANGE for v in cfoop]
                    fig3  = go.Figure(go.Bar(x=cfoop.index, y=cfoop, marker_color=bar_colors,
                                             name="CFO/OP",
                                             hovertemplate="%{y:.0f}%<extra></extra>"))
                    fig3.add_hline(y=100, line_dash="dot",
                                   line_color="rgba(128,128,128,0.4)", line_width=1,
                                   annotation_text="100%", annotation_font_size=11,
                                   annotation_font_color="#888")
                    _style(fig3, yt="% of Op. Profit", height=300, legend=False)
                    fig3.update_layout(title=dict(text="Cash Conversion Quality (CFO/OP)", font_size=13))
                    cf2.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 5 — OWNERSHIP
# ═══════════════════════════════════════════════════════════════════
with tab_own:
    st.header(f"{name}  —  Ownership & Shareholding")

    shq = D["shareholding_qtr"]
    sha = D["shareholding_annual"]

    view = st.radio("Granularity", ["Quarterly", "Annual"], horizontal=True, key="own_view")
    sh   = shq if view == "Quarterly" else sha

    if sh is not None:
        holders    = [c for c in ["Promoters", "FIIs", "DIIs", "Government", "Public"] if c in sh.columns]
        sh_colors  = [ORANGE, GREEN, BLUE, PURPLE, GREY]

        # ── Stacked area ───────────────────────────────────────────
        fig = go.Figure()
        for col, color in zip(holders, sh_colors):
            fig.add_trace(go.Scatter(
                x=sh.index, y=sh[col], name=col,
                mode="lines", stackgroup="one",
                line=dict(width=0.5, color=color),
                hovertemplate="%{y:.2f}%<extra></extra>",
            ))
        _style(fig, yt="Shareholding %", height=380)
        st.plotly_chart(fig, use_container_width=True)

        # ── FII vs DII  |  Promoter ────────────────────────────────
        o1, o2 = st.columns(2)
        with o1:
            if "FIIs" in sh.columns and "DIIs" in sh.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=sh.index, y=sh["FIIs"], name="FII",
                                          mode="lines+markers",
                                          line=dict(color=GREEN, width=2), marker=dict(size=5),
                                          hovertemplate="%{y:.2f}%<extra></extra>"))
                fig2.add_trace(go.Scatter(x=sh.index, y=sh["DIIs"], name="DII",
                                          mode="lines+markers",
                                          line=dict(color=BLUE, width=2), marker=dict(size=5),
                                          hovertemplate="%{y:.2f}%<extra></extra>"))
                _style(fig2, yt="%", height=300)
                fig2.update_layout(title=dict(text="FII vs DII", font_size=14))
                o1.plotly_chart(fig2, use_container_width=True)

        with o2:
            if "Promoters" in sh.columns:
                fig3 = go.Figure(go.Scatter(
                    x=sh.index, y=sh["Promoters"], name="Promoters",
                    mode="lines+markers", line=dict(color=ORANGE, width=2),
                    marker=dict(size=5), fill="tozeroy",
                    fillcolor="rgba(243,156,18,0.10)",
                    hovertemplate="%{y:.2f}%<extra></extra>",
                ))
                _style(fig3, yt="%", height=300, legend=False)
                fig3.update_layout(title=dict(text="Promoter Holding", font_size=14))
                o2.plotly_chart(fig3, use_container_width=True)

        # ── QoQ / YoY Changes ──────────────────────────────────────
        with st.expander("Period-on-Period Changes"):
            if len(sh) >= 2:
                delta_df = sh[holders].diff().dropna().round(2)
                delta_df.index.name = "Period"
                fig_d = go.Figure()
                for col, color in zip(holders, sh_colors):
                    if col in delta_df.columns:
                        fig_d.add_trace(go.Bar(
                            x=delta_df.index, y=delta_df[col], name=col,
                            marker_color=color, opacity=0.8,
                            hovertemplate="%{y:+.2f}pp<extra></extra>",
                        ))
                fig_d.add_hline(y=0, line_dash="dot",
                                line_color="rgba(128,128,128,0.4)", line_width=1)
                _style(fig_d, yt="Change (pp)", height=300, barmode="group")
                st.plotly_chart(fig_d, use_container_width=True)

        # ── Retail participation ───────────────────────────────────
        with st.expander("Retail Participation (No. of Shareholders)"):
            if "No. of Shareholders" in sh.columns:
                ns = sh["No. of Shareholders"].dropna()
                fig4 = go.Figure(go.Bar(
                    x=ns.index, y=ns, marker_color=TEAL,
                    hovertemplate="%{y:,.0f}<extra></extra>", name="Shareholders",
                ))
                _style(fig4, yt="Count", height=260, legend=False)
                st.plotly_chart(fig4, use_container_width=True)