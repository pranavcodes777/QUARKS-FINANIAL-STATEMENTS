import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

# ── CONFIG ───────────────────────────────────────────────────────────────────

DB = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Database"))

COMPANIES = sorted([d for d in os.listdir(DB) if os.path.isdir(os.path.join(DB, d))])

NAMES = {
    "ADANIPORTS": "Adani Ports",       "ASIANPAINT": "Asian Paints",
    "AXISBANK": "Axis Bank",           "BAJFINANCE": "Bajaj Finance",
    "BAJAJFINSV": "Bajaj Finserv",     "BHARTIARTL": "Bharti Airtel",
    "HCLTECH": "HCL Technologies",     "HDFCBANK": "HDFC Bank",
    "HINDUNILVR": "Hindustan Unilever","ICICIBANK": "ICICI Bank",
    "INDUSINDBK": "IndusInd Bank",     "INFY": "Infosys",
    "ITC": "ITC",                      "JSWSTEEL": "JSW Steel",
    "KOTAKBANK": "Kotak Mahindra Bank","LT": "Larsen & Toubro",
    "M&M": "Mahindra & Mahindra",      "MARUTI": "Maruti Suzuki",
    "NTPC": "NTPC",                    "POWERGRID": "Power Grid",
    "RELIANCE": "Reliance Industries", "SBIN": "State Bank of India",
    "SUNPHARMA": "Sun Pharma",         "TCS": "TCS",
    "TATAMOTORS": "Tata Motors",       "TATASTEEL": "Tata Steel",
    "TECHM": "Tech Mahindra",          "TITAN": "Titan",
    "ULTRACEMCO": "UltraTech Cement",  "WIPRO": "Wipro",
}

BLUE   = "#4C9BE8"
GREEN  = "#27AE60"
RED    = "#E74C3C"
ORANGE = "#F39C12"
PURPLE = "#8E44AD"

st.set_page_config(
    page_title="Sensex Fundamentals",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DATA HELPERS ──────────────────────────────────────────────────────────────

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
    cols = list(df.columns)
    cols[0] = "Metric"
    df.columns = cols
    df["Metric"] = (
        df["Metric"].astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.replace(" +", "", regex=False)
        .str.strip()
        .str.rstrip("+")
        .str.strip()
    )
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
    return {
        n: to_df(ticker, n)
        for n in [
            "quarterly_pl", "annual_pl", "balance_sheet", "cash_flow",
            "ratios", "shareholding_qtr", "shareholding_annual",
            "sales_growth", "profit_growth", "price_cagr", "roe_summary",
        ]
    }


@st.cache_data(ttl=3600)
def screener_df() -> pd.DataFrame:
    rows = []
    for t in COMPANIES:
        r: dict = {"Ticker": t, "Company": NAMES.get(t, t)}
        try:
            apl = to_df(t, "annual_pl")
            if apl is not None:
                for src, dst in [
                    ("Sales", "Revenue (Cr)"), ("Net Profit", "Net Profit (Cr)"),
                    ("OPM %", "OPM %"), ("EPS in Rs", "EPS (Rs)"),
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


# ── CHART STYLE HELPER ────────────────────────────────────────────────────────

def _style(fig, *, yt="", yt2="", height=400, legend=True, barmode=None):
    """Apply consistent professional styling to any Plotly figure."""
    layout = dict(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(15,22,32,0.92)",
            font_size=12,
            bordercolor="rgba(255,255,255,0.10)",
            namelength=-1,
        ),
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title="",
            tickfont=dict(size=11),
            showline=True,
            linecolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.10)",
            zeroline=False,
            title=yt,
            tickfont=dict(size=11),
            showline=False,
        ),
        legend=dict(
            orientation="h", y=1.06, x=0,
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ) if legend else dict(visible=False),
        margin=dict(t=40, b=30, l=10, r=10),
    )
    if not legend:
        layout["showlegend"] = False
    if barmode:
        layout["barmode"] = barmode
    fig.update_layout(**layout)
    if yt2:
        fig.update_layout(
            yaxis2=dict(
                gridcolor="rgba(0,0,0,0)",
                zeroline=False,
                title=yt2,
                tickfont=dict(size=11),
                showline=False,
            )
        )
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Sensex Fundamentals")
    ticker = st.selectbox(
        "Select Company",
        COMPANIES,
        format_func=lambda x: f"{x}  —  {NAMES.get(x, x)}",
        index=COMPANIES.index("RELIANCE") if "RELIANCE" in COMPANIES else 0,
    )
    st.caption(f"Screener.in  |  {len(COMPANIES)} companies")

D    = company_data(ticker)
name = NAMES.get(ticker, ticker)

# ── TABS ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Scorecard", "P&L", "Balance Sheet",
    "Cash Flow", "Efficiency", "Ownership", "Sensex Screener",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header(name)

    apl = D["annual_pl"]
    cf  = D["cash_flow"]
    rat = D["ratios"]

    def _fmt_cr(v):
        if abs(v) >= 100000:
            return f"₹{v/100000:.2f}L Cr"
        return f"₹{v:,.0f} Cr"

    def _delta_pct(v, p):
        if p is None or pd.isna(p) or p == 0:
            return "", True
        d = (v - p) / abs(p) * 100
        return f"{d:.1f}% YoY", d >= 0

    kpis = []
    if apl is not None and len(apl) >= 2:
        latest, prev = apl.iloc[-1], apl.iloc[-2]
        for col, label, fmt in [
            ("Sales",      "Revenue",    "cr"),
            ("Net Profit", "Net Profit", "cr"),
            ("EPS in Rs",  "EPS",        "num"),
            ("OPM %",      "OPM",        "pct"),
        ]:
            v = latest.get(col)
            p = prev.get(col)
            if v is None or pd.isna(v):
                continue
            if fmt == "cr":
                disp = _fmt_cr(v)
                d, pos = _delta_pct(v, p)
            elif fmt == "pct":
                disp = f"{v:.1f}%"
                d = f"{v-p:+.1f}pp YoY" if p is not None and not pd.isna(p) else ""
                pos = (v >= p) if (p is not None and not pd.isna(p)) else True
            else:
                disp = f"₹{v:.2f}"
                d, pos = _delta_pct(v, p)
            kpis.append((label, disp, d, pos))

    if rat is not None and "ROCE %" in rat.columns:
        rv = rat["ROCE %"].dropna()
        if len(rv) >= 2:
            v, p = rv.iloc[-1], rv.iloc[-2]
            kpis.append(("ROCE", f"{v:.1f}%", f"{v-p:+.1f}pp YoY", v >= p))

    if cf is not None and "Free Cash Flow" in cf.columns:
        fcf = cf["Free Cash Flow"].dropna()
        if len(fcf) >= 2:
            v, p = fcf.iloc[-1], fcf.iloc[-2]
            d, pos = _delta_pct(v, p)
            kpis.append(("Free Cash Flow", _fmt_cr(v), d, pos))

    cards_html = ""
    for label, value, delta, is_pos in kpis:
        dc    = "#27AE60" if is_pos else "#E74C3C"
        arrow = "↑" if is_pos else "↓"
        delta_html = (
            f'<div style="font-size:11px;color:{dc};margin-top:8px;letter-spacing:0.02em;">'
            f'{arrow} {delta}</div>'
        ) if delta else ""
        cards_html += f"""
        <div style="flex:1;min-width:0;background:rgba(128,128,128,0.06);
                    border:1px solid rgba(128,128,128,0.14);border-radius:12px;
                    padding:20px 14px;text-align:center;">
            <div style="font-size:10px;color:#888;text-transform:uppercase;
                        letter-spacing:0.12em;margin-bottom:10px;">{label}</div>
            <div style="font-size:1.3rem;font-weight:700;line-height:1.25;
                        word-break:break-word;">{value}</div>
            {delta_html}
        </div>"""

    st.markdown(
        f'<div style="display:flex;gap:12px;margin-bottom:4px;">{cards_html}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Growth Snapshot Table
    st.subheader("Growth Snapshot")

    growth_cols = [
        ("Sales CAGR",  "sales_growth"),
        ("Profit CAGR", "profit_growth"),
        ("Price CAGR",  "price_cagr"),
        ("ROE",         "roe_summary"),
    ]
    periods = [("10 Years:", "10Y"), ("5 Years:", "5Y"), ("3 Years:", "3Y")]

    header_cells = "".join(
        f'<th style="padding:10px 24px;text-align:center;font-size:11px;'
        f'text-transform:uppercase;letter-spacing:0.10em;color:#888;'
        f'border-bottom:1px solid rgba(128,128,128,0.18);">{lbl}</th>'
        for lbl, _ in growth_cols
    )
    header_row = (
        f'<tr><th style="padding:10px 24px;border-bottom:1px solid '
        f'rgba(128,128,128,0.18);width:60px;"></th>{header_cells}</tr>'
    )
    body_rows = ""
    for period_key, period_label in periods:
        cells = (
            f'<td style="padding:14px 24px;font-size:12px;color:#666;'
            f'font-weight:600;letter-spacing:0.05em;">{period_label}</td>'
        )
        for _, tbl in growth_cols:
            v = get_cagr(ticker, tbl, period_key)
            val = f"{v:.0f}%" if v is not None else "—"
            c = "#27AE60" if (v is not None and v > 0) else ("#E74C3C" if (v is not None and v < 0) else "#888")
            cells += (
                f'<td style="padding:14px 24px;text-align:center;font-size:22px;'
                f'font-weight:700;color:{c};">{val}</td>'
            )
        body_rows += f"<tr>{cells}</tr>"

    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead>{header_row}</thead><tbody>{body_rows}</tbody></table>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Trend sparklines
    if apl is not None:
        st.subheader("12-Year Trend")
        s1, s2, s3 = st.columns(3)

        def _spark(col_obj, df, col, title, color, fmt="cr"):
            if col not in df.columns:
                return
            ht = "₹%{y:,.0f} Cr<extra></extra>" if fmt == "cr" else "%{y:.1f}%<extra></extra>"
            fig = go.Figure(go.Bar(
                x=df.index, y=df[col],
                marker_color=color, opacity=0.85,
                hovertemplate=ht,
            ))
            _style(fig, height=240, legend=False)
            fig.update_layout(title=dict(text=title, font_size=13), margin=dict(t=36, b=20))
            col_obj.plotly_chart(fig, use_container_width=True)

        _spark(s1, apl, "Sales",      "Revenue", BLUE)
        _spark(s2, apl, "Net Profit", "Net Profit", GREEN)
        if rat is not None and "ROCE %" in rat.columns:
            fig_r = go.Figure(go.Scatter(
                x=rat.index, y=rat["ROCE %"],
                mode="lines+markers",
                line=dict(color=BLUE, width=2),
                marker=dict(size=5),
                hovertemplate="%{y:.1f}%<extra></extra>",
            ))
            _style(fig_r, yt="ROCE %", height=240, legend=False)
            fig_r.update_layout(title=dict(text="ROCE %", font_size=13), margin=dict(t=36, b=20))
            s3.plotly_chart(fig_r, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — P&L
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header(f"{name} — P&L")
    apl = D["annual_pl"]
    qpl = D["quarterly_pl"]

    view = st.radio("Period", ["Annual", "Quarterly"], horizontal=True)
    pl   = apl if view == "Annual" else qpl

    if pl is not None:
        # Revenue + Net Profit bars + OPM line
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "Sales" in pl.columns:
            fig.add_trace(go.Bar(
                x=pl.index, y=pl["Sales"], name="Revenue",
                marker_color=BLUE, opacity=0.85,
                hovertemplate="₹%{y:,.0f} Cr<extra></extra>",
            ), secondary_y=False)
        if "Net Profit" in pl.columns:
            fig.add_trace(go.Bar(
                x=pl.index, y=pl["Net Profit"], name="Net Profit",
                marker_color=GREEN, opacity=0.85,
                hovertemplate="₹%{y:,.0f} Cr<extra></extra>",
            ), secondary_y=False)
        if "OPM %" in pl.columns:
            fig.add_trace(go.Scatter(
                x=pl.index, y=pl["OPM %"], name="OPM %",
                mode="lines+markers", line=dict(color=ORANGE, width=2),
                marker=dict(size=5),
                hovertemplate="%{y:.1f}%<extra></extra>",
            ), secondary_y=True)
        _style(fig, yt="₹ Crores", yt2="OPM %", height=420, barmode="group")
        fig.update_yaxes(showgrid=True,  gridcolor="rgba(128,128,128,0.10)", secondary_y=False)
        fig.update_yaxes(showgrid=False, secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        # Margins
        pl2 = pl.copy()
        if "Net Profit" in pl2.columns and "Sales" in pl2.columns:
            pl2["NPM %"] = (pl2["Net Profit"] / pl2["Sales"] * 100).round(1)
        margin_cols = [c for c in ["OPM %", "NPM %"] if c in pl2.columns]
        if margin_cols:
            st.subheader("Margins")
            fig2 = go.Figure()
            for col, color, label in zip(margin_cols, [ORANGE, GREEN], ["OPM %", "NPM %"]):
                fig2.add_trace(go.Scatter(
                    x=pl2.index, y=pl2[col], name=label,
                    mode="lines+markers", line=dict(width=2, color=color),
                    marker=dict(size=5),
                    hovertemplate="%{y:.1f}%<extra></extra>",
                ))
            _style(fig2, yt="%", height=300)
            st.plotly_chart(fig2, use_container_width=True)

        # EPS
        if "EPS in Rs" in pl.columns:
            st.subheader("Earnings Per Share")
            fig3 = go.Figure(go.Bar(
                x=pl.index, y=pl["EPS in Rs"],
                marker_color=BLUE, opacity=0.85,
                hovertemplate="₹%{y:.2f}<extra></extra>",
            ))
            _style(fig3, yt="₹ per share", height=280, legend=False)
            st.plotly_chart(fig3, use_container_width=True)

        # YoY growth (annual only)
        if view == "Annual" and "Sales" in apl.columns and "Net Profit" in apl.columns:
            st.subheader("YoY Growth")
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
            fig4.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.4)", line_width=1)
            _style(fig4, yt="%", height=320, barmode="group")
            st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BALANCE SHEET
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header(f"{name} — Balance Sheet")
    bs = D["balance_sheet"]

    if bs is not None:
        # Asset composition stacked
        st.subheader("Asset Composition")
        asset_cols = [c for c in ["Fixed Assets", "CWIP", "Investments", "Other Assets"]
                      if c in bs.columns]
        if asset_cols:
            fig = go.Figure()
            for col, color in zip(asset_cols, [BLUE, ORANGE, GREEN, PURPLE]):
                fig.add_trace(go.Bar(
                    x=bs.index, y=bs[col], name=col, marker_color=color,
                    hovertemplate="₹%{y:,.0f} Cr<extra></extra>",
                ))
            _style(fig, yt="₹ Crores", height=380, barmode="stack")
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Debt vs Equity")
            eq_cols = [c for c in ["Equity Capital", "Reserves"] if c in bs.columns]
            equity  = bs[eq_cols].sum(axis=1)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=bs.index, y=equity, name="Equity + Reserves",
                marker_color=GREEN,
                hovertemplate="₹%{y:,.0f} Cr<extra></extra>",
            ))
            if "Borrowings" in bs.columns:
                fig2.add_trace(go.Bar(
                    x=bs.index, y=bs["Borrowings"], name="Borrowings",
                    marker_color=RED,
                    hovertemplate="₹%{y:,.0f} Cr<extra></extra>",
                ))
            _style(fig2, yt="₹ Crores", height=340, barmode="group")
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.subheader("Debt / Equity Ratio")
            if "Borrowings" in bs.columns:
                de = (bs["Borrowings"] / equity.replace(0, np.nan)).round(2)
                fig3 = go.Figure(go.Scatter(
                    x=de.index, y=de, mode="lines+markers",
                    line=dict(color=RED, width=2), marker=dict(size=5),
                    hovertemplate="%{y:.2f}x<extra></extra>",
                    name="D/E",
                ))
                fig3.add_hline(
                    y=1, line_dash="dot",
                    line_color="rgba(128,128,128,0.4)", line_width=1,
                    annotation_text="1x", annotation_font_size=11,
                    annotation_font_color="#888",
                )
                _style(fig3, yt="D/E Ratio", height=340, legend=False)
                st.plotly_chart(fig3, use_container_width=True)

        # CWIP
        if "CWIP" in bs.columns:
            st.subheader("Capital Work-in-Progress")
            fig4 = go.Figure(go.Bar(
                x=bs.index, y=bs["CWIP"], marker_color=ORANGE,
                hovertemplate="₹%{y:,.0f} Cr<extra></extra>",
                name="CWIP",
            ))
            _style(fig4, yt="₹ Crores", height=280, legend=False)
            st.plotly_chart(fig4, use_container_width=True)

        # Total Assets
        if "Total Assets" in bs.columns:
            st.subheader("Total Assets")
            fig5 = go.Figure(go.Scatter(
                x=bs.index, y=bs["Total Assets"],
                mode="lines", fill="tozeroy",
                line=dict(color=BLUE, width=2),
                fillcolor="rgba(76,155,232,0.12)",
                hovertemplate="₹%{y:,.0f} Cr<extra></extra>",
                name="Total Assets",
            ))
            _style(fig5, yt="₹ Crores", height=260, legend=False)
            st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CASH FLOW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header(f"{name} — Cash Flow")
    cf  = D["cash_flow"]
    apl = D["annual_pl"]

    if cf is not None:
        # OCF / ICF / CFF grouped bars
        fig = go.Figure()
        for col, color, short in [
            ("Cash from Operating Activity", GREEN,  "Operating (OCF)"),
            ("Cash from Investing Activity", RED,    "Investing (ICF)"),
            ("Cash from Financing Activity", BLUE,   "Financing (CFF)"),
        ]:
            if col in cf.columns:
                fig.add_trace(go.Bar(
                    x=cf.index, y=cf[col], name=short,
                    marker_color=color, opacity=0.85,
                    hovertemplate="₹%{y:,.0f} Cr<extra></extra>",
                ))
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.4)", line_width=1)
        _style(fig, yt="₹ Crores", height=400, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Free Cash Flow")
            if "Free Cash Flow" in cf.columns:
                fcf    = cf["Free Cash Flow"].dropna()
                colors = [GREEN if v >= 0 else RED for v in fcf]
                fig2   = go.Figure(go.Bar(
                    x=fcf.index, y=fcf, marker_color=colors,
                    hovertemplate="₹%{y:,.0f} Cr<extra></extra>",
                    name="FCF",
                ))
                fig2.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.4)", line_width=1)
                _style(fig2, yt="₹ Crores", height=320, legend=False)
                st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.subheader("Cash Conversion Quality (OCF / Op. Profit)")
            if "CFO/OP" in cf.columns:
                cfoop = cf["CFO/OP"].dropna()
                bar_colors = [GREEN if v >= 100 else ORANGE for v in cfoop]
                fig3  = go.Figure(go.Bar(
                    x=cfoop.index, y=cfoop, marker_color=bar_colors,
                    hovertemplate="%{y:.0f}%<extra></extra>",
                    name="CFO/OP",
                ))
                fig3.add_hline(
                    y=100, line_dash="dot",
                    line_color="rgba(128,128,128,0.4)", line_width=1,
                    annotation_text="100%", annotation_font_size=11,
                    annotation_font_color="#888",
                )
                _style(fig3, yt="% of Operating Profit", height=320, legend=False)
                st.plotly_chart(fig3, use_container_width=True)

        # Earnings quality
        if apl is not None and "Net Profit" in apl.columns and \
                "Cash from Operating Activity" in cf.columns:
            st.subheader("Earnings Quality — OCF vs Net Profit")
            common = cf.index.intersection(apl.index)
            fig4   = go.Figure()
            fig4.add_trace(go.Scatter(
                x=list(common), y=list(apl.loc[common, "Net Profit"]),
                name="Net Profit", mode="lines+markers",
                line=dict(color=BLUE, width=2), marker=dict(size=5),
                hovertemplate="₹%{y:,.0f} Cr<extra></extra>",
            ))
            fig4.add_trace(go.Scatter(
                x=list(common),
                y=list(cf.loc[common, "Cash from Operating Activity"]),
                name="Operating Cash Flow", mode="lines+markers",
                line=dict(color=GREEN, width=2), marker=dict(size=5),
                hovertemplate="₹%{y:,.0f} Cr<extra></extra>",
            ))
            _style(fig4, yt="₹ Crores", height=340)
            st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EFFICIENCY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header(f"{name} — Efficiency & Returns")
    rat = D["ratios"]

    if rat is not None:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("ROCE %")
            if "ROCE %" in rat.columns:
                roce = rat["ROCE %"].dropna()
                fig = go.Figure(go.Scatter(
                    x=roce.index, y=roce,
                    mode="lines+markers",
                    line=dict(color=BLUE, width=2.5),
                    marker=dict(size=6),
                    fill="tozeroy", fillcolor="rgba(76,155,232,0.08)",
                    hovertemplate="%{y:.1f}%<extra></extra>",
                    name="ROCE",
                ))
                _style(fig, yt="Return on Capital Employed (%)", height=320, legend=False)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Cash Conversion Cycle")
            if "Cash Conversion Cycle" in rat.columns:
                ccc    = rat["Cash Conversion Cycle"].dropna()
                colors = [RED if v > 0 else GREEN for v in ccc]
                fig2   = go.Figure(go.Bar(
                    x=ccc.index, y=ccc, marker_color=colors,
                    hovertemplate="%{y:.0f} days<extra></extra>",
                    name="CCC",
                ))
                fig2.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.4)", line_width=1)
                _style(fig2, yt="Days", height=320, legend=False)
                st.plotly_chart(fig2, use_container_width=True)

        # Working capital breakdown — all three on shared axis
        st.subheader("Working Capital Breakdown")
        wc_cols = [c for c in ["Debtor Days", "Inventory Days", "Days Payable"]
                   if c in rat.columns]
        if wc_cols:
            fig3 = go.Figure()
            for col, color in zip(wc_cols, [BLUE, ORANGE, RED]):
                fig3.add_trace(go.Scatter(
                    x=rat.index, y=rat[col], name=col,
                    mode="lines+markers",
                    line=dict(width=2, color=color),
                    marker=dict(size=6),
                    hovertemplate="%{y:.0f} days<extra></extra>",
                ))
            _style(fig3, yt="Days", height=360)
            fig3.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig3, use_container_width=True)

        # Net Working Capital Days
        if "Working Capital Days" in rat.columns:
            st.subheader("Net Working Capital Days")
            wcd    = rat["Working Capital Days"].dropna()
            colors = [RED if v > 0 else GREEN for v in wcd]
            fig4   = go.Figure(go.Bar(
                x=wcd.index, y=wcd, marker_color=colors,
                hovertemplate="%{y:.0f} days<extra></extra>",
                name="WC Days",
            ))
            fig4.add_hline(y=0, line_dash="dot", line_color="rgba(128,128,128,0.4)", line_width=1)
            _style(fig4, yt="Days", height=280, legend=False)
            st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — OWNERSHIP
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header(f"{name} — Ownership")
    shq = D["shareholding_qtr"]
    sha = D["shareholding_annual"]

    view = st.radio("Granularity", ["Quarterly", "Annual"], horizontal=True)
    sh   = shq if view == "Quarterly" else sha

    if sh is not None:
        holders = [c for c in ["Promoters", "FIIs", "DIIs", "Government", "Public"]
                   if c in sh.columns]

        # Stacked area shareholding
        fig = go.Figure()
        sh_colors = [BLUE, GREEN, ORANGE, RED, PURPLE]
        for col, color in zip(holders, sh_colors):
            fig.add_trace(go.Scatter(
                x=sh.index, y=sh[col], name=col,
                mode="lines", stackgroup="one",
                line=dict(width=0.5, color=color),
                fillcolor=color.replace(")", ",0.5)").replace("rgb", "rgba") if color.startswith("rgb") else color,
                hovertemplate="%{y:.2f}%<extra></extra>",
            ))
        _style(fig, yt="Shareholding %", height=400)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("FII vs DII")
            if "FIIs" in sh.columns and "DIIs" in sh.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=sh.index, y=sh["FIIs"], name="FII",
                    mode="lines+markers", line=dict(color=GREEN, width=2),
                    marker=dict(size=5),
                    hovertemplate="%{y:.2f}%<extra></extra>",
                ))
                fig2.add_trace(go.Scatter(
                    x=sh.index, y=sh["DIIs"], name="DII",
                    mode="lines+markers", line=dict(color=BLUE, width=2),
                    marker=dict(size=5),
                    hovertemplate="%{y:.2f}%<extra></extra>",
                ))
                _style(fig2, yt="Shareholding %", height=320)
                st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.subheader("Promoter Holding")
            if "Promoters" in sh.columns:
                fig3 = go.Figure(go.Scatter(
                    x=sh.index, y=sh["Promoters"],
                    mode="lines+markers",
                    line=dict(color=ORANGE, width=2),
                    marker=dict(size=5),
                    fill="tozeroy", fillcolor="rgba(243,156,18,0.10)",
                    hovertemplate="%{y:.2f}%<extra></extra>",
                    name="Promoters",
                ))
                _style(fig3, yt="Shareholding %", height=320, legend=False)
                st.plotly_chart(fig3, use_container_width=True)

        if "No. of Shareholders" in sh.columns:
            st.subheader("Number of Shareholders")
            ns = sh["No. of Shareholders"].dropna()
            fig4 = go.Figure(go.Bar(
                x=ns.index, y=ns, marker_color=BLUE,
                hovertemplate="%{y:,.0f}<extra></extra>",
                name="Shareholders",
            ))
            _style(fig4, yt="Count", height=280, legend=False)
            st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SENSEX SCREENER
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("Sensex 30 — Screener")

    with st.spinner("Loading all companies..."):
        sdf = screener_df()

    METRIC_COLS = [c for c in [
        "Revenue (Cr)", "Net Profit (Cr)", "OPM %", "NPM %", "ROCE %",
        "FCF (Cr)", "EPS (Rs)", "Sales CAGR 5Y (%)", "Profit CAGR 5Y (%)",
        "Price CAGR 3Y (%)", "ROE (last yr)",
    ] if c in sdf.columns]

    st.subheader("All Companies")
    col_sort, col_asc = st.columns([3, 1])
    sort_by = col_sort.selectbox("Sort by", METRIC_COLS)
    asc     = col_asc.checkbox("Ascending", value=False)

    display = sdf.sort_values(sort_by, ascending=asc).reset_index(drop=True)
    st.dataframe(
        display[["Company", "Ticker"] + METRIC_COLS],
        use_container_width=True, height=560, hide_index=True,
    )

    st.divider()

    # Bar comparison
    metric_bar = st.selectbox("Compare by", METRIC_COLS, key="bar_metric")
    bar_data   = sdf[["Company", metric_bar]].dropna().sort_values(metric_bar)
    bar_colors = [GREEN if v >= 0 else RED for v in bar_data[metric_bar]]

    fig_b = go.Figure(go.Bar(
        x=bar_data[metric_bar],
        y=bar_data["Company"],
        orientation="h",
        marker_color=bar_colors,
        hovertemplate="%{y}  —  %{x:.1f}<extra></extra>",
    ))
    fig_b.update_layout(
        height=700,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        margin=dict(l=180, t=10, b=30, r=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.10)",
                   zeroline=False, title=metric_bar, tickfont_size=11),
        yaxis=dict(showgrid=False, zeroline=False, title="", tickfont_size=11),
        hoverlabel=dict(bgcolor="rgba(15,22,32,0.92)", font_size=12,
                        bordercolor="rgba(255,255,255,0.10)"),
    )
    st.plotly_chart(fig_b, use_container_width=True)

    st.divider()

    # Scatter
    st.subheader("Scatter — Growth vs Returns")
    s1, s2 = st.columns(2)
    x_ax = s1.selectbox(
        "X axis", METRIC_COLS,
        index=METRIC_COLS.index("Sales CAGR 5Y (%)") if "Sales CAGR 5Y (%)" in METRIC_COLS else 0,
        key="sc_x",
    )
    y_ax = s2.selectbox(
        "Y axis", METRIC_COLS,
        index=METRIC_COLS.index("ROCE %") if "ROCE %" in METRIC_COLS else 1,
        key="sc_y",
    )

    sc_data = sdf[["Company", x_ax, y_ax]].dropna()
    fig_s   = px.scatter(
        sc_data, x=x_ax, y=y_ax, text="Company",
        hover_name="Company",
        hover_data={x_ax: ":.1f", y_ax: ":.1f", "Company": False},
    )
    fig_s.update_traces(
        textposition="top center", marker_size=9,
        marker_color=BLUE,
    )
    fig_s.update_layout(
        height=520,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.10)",
                   zeroline=False, title=x_ax, tickfont_size=11),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.10)",
                   zeroline=False, title=y_ax, tickfont_size=11),
        hoverlabel=dict(bgcolor="rgba(15,22,32,0.92)", font_size=12,
                        bordercolor="rgba(255,255,255,0.10)"),
        margin=dict(t=20, b=40, l=60, r=10),
    )
    st.plotly_chart(fig_s, use_container_width=True)
