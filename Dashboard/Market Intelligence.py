"""
Market Intelligence  --  Cross-Equity Analysis
===============================================
Target user : Equity analyst hunting for ideas, comparing peers, monitoring sector rotation
Tabs        : Screener | Peer Compare | Relative Strength | Sector View
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

CHART_CFG = dict(displaylogo=False, modeBarButtonsToRemove=["lasso2d", "select2d"], displayModeBar=True)

st.set_page_config(
    page_title="Market Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
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
def screener_df() -> pd.DataFrame:
    rows = []
    for t in COMPANIES:
        r = {"Ticker": t, "Company": NAMES.get(t, t), "Sector": SECTOR_OF.get(t, "Other")}
        try:
            apl = to_df(t, "annual_pl")
            if apl is not None:
                for src, dst in [
                    ("Sales",      "Revenue (Cr)"),
                    ("Net Profit", "Net Profit (Cr)"),
                    ("OPM %",      "OPM %"),
                    ("EPS in Rs",  "EPS (Rs)"),
                ]:
                    if src in apl.columns:
                        r[dst] = round(float(apl[src].dropna().iloc[-1]), 1)
                if "Net Profit" in apl.columns and "Sales" in apl.columns:
                    rev = float(apl["Sales"].dropna().iloc[-1])
                    pat = float(apl["Net Profit"].dropna().iloc[-1])
                    r["NPM %"] = round(pat / rev * 100, 1) if rev else None
            cf = to_df(t, "cash_flow")
            if cf is not None and "Free Cash Flow" in cf.columns:
                r["FCF (Cr)"] = round(float(cf["Free Cash Flow"].dropna().iloc[-1]), 0)
            rat = to_df(t, "ratios")
            if rat is not None and "ROCE %" in rat.columns:
                r["ROCE %"] = round(float(rat["ROCE %"].dropna().iloc[-1]), 1)
            r["Sales CAGR 5Y (%)"]  = get_cagr(t, "sales_growth",  "5 Years:")
            r["Profit CAGR 5Y (%)"] = get_cagr(t, "profit_growth", "5 Years:")
            r["Price CAGR 3Y (%)"]  = get_cagr(t, "price_cagr",    "3 Years:")
            r["ROE (last yr)"]      = get_cagr(t, "roe_summary",   "Last Year:")
        except Exception:
            pass
        rows.append(r)
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def relative_strength_df() -> pd.DataFrame:
    periods = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
    rows = []
    for ticker in COMPANIES:
        path = os.path.join(OHLCV_DIR, f"{ticker}.parquet")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            close = df["Close"].dropna().sort_index()
            if close.empty:
                continue
            row = {
                "Ticker":  ticker,
                "Company": NAMES.get(ticker, ticker),
                "Sector":  SECTOR_OF.get(ticker, "Other"),
            }
            current = close.iloc[-1]
            for label, n_bars in periods.items():
                if len(close) > n_bars:
                    past = close.iloc[-n_bars - 1]
                    row[label] = round((current - past) / past * 100, 1)
                else:
                    row[label] = np.nan
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def load_ohlcv_norm(tickers: tuple, period_days: int) -> pd.DataFrame:
    """Load OHLCV for multiple tickers, normalise to 100 at start of period."""
    result = {}
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=period_days)
    for ticker in tickers:
        path = os.path.join(OHLCV_DIR, f"{ticker}.parquet")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            close = df["Close"].dropna().sort_index()
            close = close[close.index >= cutoff]
            if close.empty:
                continue
            result[NAMES.get(ticker, ticker)] = (close / close.iloc[0] * 100).round(2)
        except Exception:
            continue
    if not result:
        return pd.DataFrame()
    return pd.DataFrame(result)


# ── CHART STYLE ─────────────────────────────────────────────────────
def _style(fig, *, yt="", height=420, legend=True, barmode=None):
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
    return fig


# ═══════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════
st.title("Market Intelligence")
st.caption(f"Cross-equity analysis  |  {len(COMPANIES)} companies")

# ═══════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════
tab_scr, tab_peer, tab_rs, tab_sec = st.tabs([
    "  Screener  ",
    "  Peer Compare  ",
    "  Relative Strength  ",
    "  Sector View  ",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — SCREENER
# ═══════════════════════════════════════════════════════════════════
with tab_scr:
    st.subheader("Fundamental Screener")
    with st.spinner("Loading all companies..."):
        sdf = screener_df()

    METRIC_COLS = [c for c in [
        "Revenue (Cr)", "Net Profit (Cr)", "OPM %", "NPM %", "ROCE %",
        "FCF (Cr)", "EPS (Rs)", "Sales CAGR 5Y (%)", "Profit CAGR 5Y (%)",
        "Price CAGR 3Y (%)", "ROE (last yr)",
    ] if c in sdf.columns]

    # ── Filter row ─────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    scr_sector = f1.selectbox("Sector", ["All Sectors"] + sorted(SECTORS.keys()), key="scr_sec")
    sort_by    = f2.selectbox("Sort by", METRIC_COLS, key="scr_sort")
    min_val    = f3.number_input(f"Min {sort_by}", value=None, key="scr_min")
    asc        = f4.checkbox("Ascending", value=False, key="scr_asc")

    display = sdf.copy()
    if scr_sector != "All Sectors":
        display = display[display["Sector"] == scr_sector]
    if min_val is not None and sort_by in display.columns:
        display = display[display[sort_by] >= min_val]
    display = display.sort_values(sort_by, ascending=asc).reset_index(drop=True)

    st.caption(f"{len(display)} companies shown")
    st.dataframe(
        display[["Company", "Ticker", "Sector"] + METRIC_COLS],
        use_container_width=True, height=480, hide_index=True,
    )

    # ── Bar comparison ─────────────────────────────────────────────
    with st.expander("Bar Comparison"):
        metric_bar = st.selectbox("Metric", METRIC_COLS, key="bar_metric")
        bar_data   = display[["Company", metric_bar]].dropna().sort_values(metric_bar)
        bar_colors = [GREEN if v >= 0 else RED for v in bar_data[metric_bar]]
        bar_h      = max(300, len(bar_data) * 26)
        fig_b = go.Figure(go.Bar(
            x=bar_data[metric_bar], y=bar_data["Company"],
            orientation="h", marker_color=bar_colors,
            hovertemplate="%{y}  %{x:.1f}<extra></extra>",
        ))
        fig_b.update_layout(
            height=bar_h, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            hovermode="closest", margin=dict(l=200, t=10, b=30, r=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.10)",
                       zeroline=False, title=metric_bar, tickfont_size=11),
            yaxis=dict(showgrid=False, zeroline=False, tickfont_size=11),
            hoverlabel=dict(bgcolor="rgba(15,22,32,0.92)", font_size=12,
                            bordercolor="rgba(255,255,255,0.10)"),
        )
        st.plotly_chart(fig_b, use_container_width=True, config=CHART_CFG, key="scr_bar")

    # ── Scatter ────────────────────────────────────────────────────
    with st.expander("Scatter Plot"):
        s1, s2 = st.columns(2)
        x_ax = s1.selectbox("X axis", METRIC_COLS,
                             index=METRIC_COLS.index("Sales CAGR 5Y (%)") if "Sales CAGR 5Y (%)" in METRIC_COLS else 0,
                             key="sc_x")
        y_ax = s2.selectbox("Y axis", METRIC_COLS,
                             index=METRIC_COLS.index("ROCE %") if "ROCE %" in METRIC_COLS else 1,
                             key="sc_y")
        sc_data = display[["Company", "Sector", x_ax, y_ax]].dropna()
        fig_s   = px.scatter(
            sc_data, x=x_ax, y=y_ax, text="Company",
            color="Sector" if scr_sector == "All Sectors" else None,
            hover_name="Company",
            hover_data={x_ax: ":.1f", y_ax: ":.1f", "Company": False},
        )
        if scr_sector != "All Sectors":
            fig_s.update_traces(marker_color=BLUE)
        fig_s.update_traces(textposition="top center", marker_size=9)
        fig_s.update_layout(
            height=520, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            hovermode="closest",
            xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.10)",
                       zeroline=False, title=x_ax, tickfont_size=11),
            yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.10)",
                       zeroline=False, title=y_ax, tickfont_size=11),
            hoverlabel=dict(bgcolor="rgba(15,22,32,0.92)", font_size=12,
                            bordercolor="rgba(255,255,255,0.10)"),
            margin=dict(t=20, b=40, l=60, r=10),
        )
        st.plotly_chart(fig_s, use_container_width=True, config=CHART_CFG, key="scr_scatter")


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — PEER COMPARE
# ═══════════════════════════════════════════════════════════════════
with tab_peer:
    st.subheader("Peer Comparison")

    # ── Controls ───────────────────────────────────────────────────
    pc1, pc2, pc3 = st.columns([3, 2, 2])
    peer_sector  = pc1.selectbox("Sector (quick filter)", ["All"] + sorted(SECTORS.keys()), key="pc_sec")
    peer_pool    = COMPANIES if peer_sector == "All" else \
                   [c for c in COMPANIES if SECTOR_OF.get(c) == peer_sector] or COMPANIES
    default_peers = peer_pool[:min(5, len(peer_pool))]
    sel_peers    = pc1.multiselect(
        "Select companies (max 8)",
        peer_pool,
        default=default_peers,
        format_func=lambda x: f"{x} — {NAMES.get(x, x)}",
        max_selections=8,
        key="pc_tickers",
    )
    peer_period  = pc2.radio("Price Period", ["3M", "6M", "1Y", "3Y"], index=2, key="pc_period")
    peer_view    = pc3.radio("View", ["Price Trend", "Fundamental Metrics"], key="pc_view")

    if not sel_peers:
        st.info("Select at least two companies above.")
    else:
        if peer_view == "Price Trend":
            # ── Normalised price chart ─────────────────────────────
            p_days = {"3M": 91, "6M": 182, "1Y": 365, "3Y": 1095}[peer_period]
            norm_df = load_ohlcv_norm(tuple(sel_peers), p_days)

            if norm_df.empty:
                st.warning("No OHLCV data available for selected companies.")
            else:
                fig_norm = go.Figure()
                palette  = [BLUE, GREEN, ORANGE, RED, PURPLE, TEAL, GREY, "#E67E22"]
                for i, col in enumerate(norm_df.columns):
                    fig_norm.add_trace(go.Scatter(
                        x=norm_df.index, y=norm_df[col], name=col,
                        mode="lines", line=dict(width=2, color=palette[i % len(palette)]),
                        hovertemplate="%{y:.1f}<extra></extra>",
                    ))
                fig_norm.add_hline(y=100, line_dash="dot",
                                   line_color="rgba(128,128,128,0.4)", line_width=1)
                _style(fig_norm, yt="Indexed to 100", height=460)
                fig_norm.update_layout(title=dict(
                    text=f"Relative Price Performance (normalised, last {peer_period})",
                    font_size=14,
                ))
                st.plotly_chart(fig_norm, use_container_width=True, config=CHART_CFG, key="peer_norm")

                # Returns table
                if not norm_df.empty:
                    latest = norm_df.iloc[-1] - 100
                    ret_df = latest.reset_index()
                    ret_df.columns = ["Company", f"Return ({peer_period}) %"]
                    ret_df = ret_df.sort_values(f"Return ({peer_period}) %", ascending=False)
                    st.dataframe(ret_df, use_container_width=True, hide_index=True)

        else:
            # ── Fundamental metrics side-by-side ───────────────────
            with st.spinner("Loading fundamentals..."):
                sdf = screener_df()
            peer_data = sdf[sdf["Ticker"].isin(sel_peers)].set_index("Company")

            METRIC_COLS = [c for c in [
                "Revenue (Cr)", "Net Profit (Cr)", "OPM %", "NPM %", "ROCE %",
                "FCF (Cr)", "EPS (Rs)", "Sales CAGR 5Y (%)", "Profit CAGR 5Y (%)", "ROE (last yr)",
            ] if c in peer_data.columns]

            sel_metric = st.selectbox("Metric to compare", METRIC_COLS, key="pc_metric")
            bar_data   = peer_data[[sel_metric]].dropna().sort_values(sel_metric)
            bar_colors = [GREEN if v >= 0 else RED for v in bar_data[sel_metric]]
            fig_pb = go.Figure(go.Bar(
                x=bar_data[sel_metric], y=bar_data.index,
                orientation="h", marker_color=bar_colors,
                hovertemplate="%{y}  %{x:.1f}<extra></extra>",
            ))
            fig_pb.update_layout(
                height=max(300, len(bar_data) * 50),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=200, t=10, b=30, r=10),
                xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.10)",
                           zeroline=False, title=sel_metric, tickfont_size=11),
                yaxis=dict(showgrid=False, zeroline=False, tickfont_size=11),
                hoverlabel=dict(bgcolor="rgba(15,22,32,0.92)", font_size=12,
                                bordercolor="rgba(255,255,255,0.10)"),
            )
            st.plotly_chart(fig_pb, use_container_width=True, config=CHART_CFG, key="peer_bar")

            # All metrics table
            with st.expander("Full Metrics Table"):
                disp = peer_data[METRIC_COLS].T
                st.dataframe(disp.style.format("{:.1f}", na_rep="—"),
                             use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — RELATIVE STRENGTH
# ═══════════════════════════════════════════════════════════════════
with tab_rs:
    st.subheader("Relative Strength  —  Price Return Heatmap")

    with st.spinner("Calculating returns across all tickers..."):
        rs_df = relative_strength_df()

    if rs_df.empty:
        st.warning("No OHLCV data found. Run fetch_ohlcv.py first.")
    else:
        # ── Controls ───────────────────────────────────────────────
        r1, r2, r3 = st.columns([2, 2, 2])
        rs_sector  = r1.selectbox("Sector filter", ["All Sectors"] + sorted(SECTORS.keys()), key="rs_sec")
        rs_sort_by = r2.selectbox("Sort within sector by", ["1M", "3M", "6M", "1Y", "1W"], key="rs_sort")
        show_n     = r3.slider("Max companies shown", 20, len(rs_df), min(60, len(rs_df)), key="rs_n")

        display_rs = rs_df.copy()
        if rs_sector != "All Sectors":
            display_rs = display_rs[display_rs["Sector"] == rs_sector]

        display_rs = (display_rs
                      .sort_values(["Sector", rs_sort_by], ascending=[True, False])
                      .head(show_n)
                      .reset_index(drop=True))

        period_cols = [c for c in ["1W", "1M", "3M", "6M", "1Y"] if c in display_rs.columns]
        heat_data   = display_rs[period_cols].values
        y_labels    = display_rs["Ticker"] + " (" + display_rs["Sector"].str[:8] + ")"

        # Max abs value for symmetric colorscale
        abs_max = max(np.nanmax(np.abs(heat_data)), 1)

        fig_heat = go.Figure(go.Heatmap(
            z=heat_data,
            x=period_cols,
            y=y_labels,
            colorscale=[
                [0.0, "rgba(192,57,43,0.9)"],
                [0.35, "rgba(231,76,60,0.5)"],
                [0.5, "rgba(40,40,40,0.2)"],
                [0.65, "rgba(39,174,96,0.5)"],
                [1.0, "rgba(30,132,73,0.9)"],
            ],
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            text=[[f"{v:+.1f}%" if not np.isnan(v) else "—" for v in row] for row in heat_data],
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>Period: %{x}<br>Return: %{text}<extra></extra>",
            showscale=True,
            colorbar=dict(title="Return %", tickfont=dict(size=10)),
        ))
        fig_heat.update_layout(
            height=max(400, len(display_rs) * 24 + 80),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=160, t=40, b=40, r=80),
            xaxis=dict(side="top", tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
            font=dict(family="Inter, sans-serif", size=12),
        )
        st.plotly_chart(fig_heat, use_container_width=True, config=CHART_CFG, key="rs_heat")

        # ── Top / Bottom movers ────────────────────────────────────
        with st.expander("Top & Bottom Movers"):
            tb_period = st.radio("Period", period_cols, horizontal=True, key="tb_period")
            top_n  = 10
            sorted_rs = display_rs[["Ticker", "Company", "Sector", tb_period]].dropna().sort_values(tb_period, ascending=False)
            t1, t2 = st.columns(2)
            with t1:
                st.markdown("**Top performers**")
                top = sorted_rs.head(top_n)
                fig_top = go.Figure(go.Bar(
                    x=top[tb_period], y=top["Company"],
                    orientation="h", marker_color=GREEN, opacity=0.85,
                    hovertemplate="%{y}  %{x:+.1f}%<extra></extra>",
                ))
                fig_top.update_layout(
                    height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=160, t=10, b=20, r=10),
                    xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.10)",
                               zeroline=False, title=f"Return {tb_period} (%)", tickfont_size=11),
                    yaxis=dict(showgrid=False, zeroline=False, tickfont_size=11),
                )
                t1.plotly_chart(fig_top, use_container_width=True)
            with t2:
                st.markdown("**Bottom performers**")
                bot = sorted_rs.tail(top_n).sort_values(tb_period)
                fig_bot = go.Figure(go.Bar(
                    x=bot[tb_period], y=bot["Company"],
                    orientation="h", marker_color=RED, opacity=0.85,
                    hovertemplate="%{y}  %{x:+.1f}%<extra></extra>",
                ))
                fig_bot.update_layout(
                    height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=160, t=10, b=20, r=10),
                    xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.10)",
                               zeroline=False, title=f"Return {tb_period} (%)", tickfont_size=11),
                    yaxis=dict(showgrid=False, zeroline=False, tickfont_size=11),
                )
                t2.plotly_chart(fig_bot, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — SECTOR VIEW
# ═══════════════════════════════════════════════════════════════════
with tab_sec:
    st.subheader("Sector View  —  Aggregated Fundamentals")

    with st.spinner("Aggregating sector data..."):
        sdf = screener_df()

    FUND_METRICS = [c for c in [
        "OPM %", "NPM %", "ROCE %", "ROE (last yr)",
        "Sales CAGR 5Y (%)", "Profit CAGR 5Y (%)", "Price CAGR 3Y (%)",
    ] if c in sdf.columns]

    # Aggregate by sector (median)
    sec_agg = (sdf.groupby("Sector")[FUND_METRICS]
               .median().round(1)
               .reset_index()
               .sort_values(FUND_METRICS[0] if FUND_METRICS else "Sector"))

    # ── Sector metric selector ─────────────────────────────────────
    sv1, sv2 = st.columns([3, 1])
    sec_metric = sv1.selectbox("Metric (median across sector)", FUND_METRICS, key="sv_metric")
    sec_sort   = sv2.radio("Sort", ["High first", "Low first"], key="sv_sort")

    sec_sorted = sec_agg.sort_values(sec_metric, ascending=(sec_sort == "Low first"))
    bar_colors = [GREEN if v >= 0 else RED for v in sec_sorted[sec_metric]]

    fig_sv = go.Figure(go.Bar(
        x=sec_sorted[sec_metric], y=sec_sorted["Sector"],
        orientation="h", marker_color=bar_colors, opacity=0.85,
        text=[f"{v:.1f}" for v in sec_sorted[sec_metric]],
        textposition="outside",
        hovertemplate="%{y}  %{x:.1f}<extra></extra>",
    ))
    fig_sv.update_layout(
        height=500, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=180, t=10, b=40, r=60),
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.10)",
                   zeroline=False, title=sec_metric, tickfont_size=11),
        yaxis=dict(showgrid=False, zeroline=False, tickfont_size=11),
        hoverlabel=dict(bgcolor="rgba(15,22,32,0.92)", font_size=12,
                        bordercolor="rgba(255,255,255,0.10)"),
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_sv, use_container_width=True, config=CHART_CFG, key="sec_bar")

    st.divider()

    # ── Full sector heatmap ────────────────────────────────────────
    st.markdown("**Sector Heatmap — all metrics**")
    heat_sec = sec_agg.set_index("Sector")[FUND_METRICS]
    z_vals   = heat_sec.values.astype(float)

    fig_sh = go.Figure(go.Heatmap(
        z=z_vals,
        x=FUND_METRICS,
        y=heat_sec.index.tolist(),
        colorscale=[
            [0.0, "rgba(192,57,43,0.9)"],
            [0.4, "rgba(231,76,60,0.4)"],
            [0.5, "rgba(40,40,40,0.15)"],
            [0.6, "rgba(39,174,96,0.4)"],
            [1.0, "rgba(30,132,73,0.9)"],
        ],
        zmid=np.nanmedian(z_vals),
        text=[[f"{v:.1f}" if not np.isnan(v) else "—" for v in row] for row in z_vals],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Value", tickfont=dict(size=10)),
    ))
    fig_sh.update_layout(
        height=max(350, len(heat_sec) * 30 + 80),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=160, t=60, b=60, r=80),
        xaxis=dict(side="top", tickfont=dict(size=11), tickangle=-20),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_sh, use_container_width=True, config=CHART_CFG, key="sec_heat")

    # ── Drill into one sector ──────────────────────────────────────
    with st.expander("Drill into a Sector"):
        drill_sec = st.selectbox("Choose sector", sorted(SECTORS.keys()), key="drill_sec")
        drill_df  = sdf[sdf["Sector"] == drill_sec].sort_values(
            "ROCE %" if "ROCE %" in sdf.columns else sdf.columns[3], ascending=False
        )
        DRILL_COLS = [c for c in [
            "Company", "Revenue (Cr)", "OPM %", "ROCE %", "FCF (Cr)",
            "Sales CAGR 5Y (%)", "Profit CAGR 5Y (%)", "ROE (last yr)",
        ] if c in drill_df.columns]
        st.dataframe(drill_df[DRILL_COLS], use_container_width=True,
                     height=300, hide_index=True)