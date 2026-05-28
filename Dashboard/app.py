import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

# ── CONFIG ───────────────────────────────────────────────────────────────────

DB = r"E:\Quarks&Quants\Fundamental\Financial Statements\Database"

COMPANIES = sorted([d for d in os.listdir(DB) if os.path.isdir(os.path.join(DB, d))])

NAMES = {
    "ADANIPORTS": "Adani Ports",      "ASIANPAINT": "Asian Paints",
    "AXISBANK": "Axis Bank",          "BAJFINANCE": "Bajaj Finance",
    "BAJAJFINSV": "Bajaj Finserv",    "BHARTIARTL": "Bharti Airtel",
    "HCLTECH": "HCL Technologies",    "HDFCBANK": "HDFC Bank",
    "HINDUNILVR": "Hindustan Unilever","ICICIBANK": "ICICI Bank",
    "INDUSINDBK": "IndusInd Bank",    "INFY": "Infosys",
    "ITC": "ITC",                     "JSWSTEEL": "JSW Steel",
    "KOTAKBANK": "Kotak Mahindra Bank","LT": "Larsen & Toubro",
    "M&M": "Mahindra & Mahindra",     "MARUTI": "Maruti Suzuki",
    "NTPC": "NTPC",                   "POWERGRID": "Power Grid",
    "RELIANCE": "Reliance Industries","SBIN": "State Bank of India",
    "SUNPHARMA": "Sun Pharma",        "TCS": "TCS",
    "TATAMOTORS": "Tata Motors",      "TATASTEEL": "Tata Steel",
    "TECHM": "Tech Mahindra",         "TITAN": "Titan",
    "ULTRACEMCO": "UltraTech Cement", "WIPRO": "Wipro",
}

BLUE   = "#1f77b4"
GREEN  = "#2ca02c"
RED    = "#d62728"
ORANGE = "#ff7f0e"

st.set_page_config(
    page_title="Sensex Fundamentals",
    page_icon="📊",
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
    """Returns transposed DataFrame: index = date periods, columns = metrics."""
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
                    ("OPM %", "OPM %"), ("EPS in Rs", "EPS (₹)"),
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


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Sensex Fundamentals")
    ticker = st.selectbox(
        "Select Company",
        COMPANIES,
        format_func=lambda x: f"{x}  —  {NAMES.get(x, x)}",
        index=COMPANIES.index("RELIANCE") if "RELIANCE" in COMPANIES else 0,
    )
    st.caption(f"Screener.in data  |  {len(COMPANIES)} companies")

D    = company_data(ticker)
name = NAMES.get(ticker, ticker)

# ── TABS ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🏠 Scorecard",
    "📈 P&L",
    "🏦 Balance Sheet",
    "💵 Cash Flow",
    "⚙️ Efficiency",
    "👥 Ownership",
    "🔍 Sensex Screener",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header(f"{name}")

    apl = D["annual_pl"]
    cf  = D["cash_flow"]
    rat = D["ratios"]

    if apl is not None and len(apl) >= 2:
        latest = apl.iloc[-1]
        prev   = apl.iloc[-2]

        def _metric(col, label, suffix="", divisor=1):
            v = latest.get(col)
            p = prev.get(col)
            if v is None or pd.isna(v):
                return
            disp = f"₹{v/divisor:,.0f} Cr" if suffix == "cr" else (
                   f"{v:.1f}%" if suffix == "pct" else f"{v:.2f}")
            delta = f"{((v-p)/abs(p)*100):.1f}% YoY" if (p and not pd.isna(p) and p != 0) else None
            st.metric(label, disp, delta)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: _metric("Sales",      "Revenue",    "cr")
        with c2: _metric("Net Profit", "Net Profit", "cr")
        with c3: _metric("EPS in Rs",  "EPS (₹)")
        with c4: _metric("OPM %",      "OPM %",      "pct")
        with c5:
            if rat is not None and "ROCE %" in rat.columns:
                rv = rat["ROCE %"].dropna()
                if len(rv) >= 2:
                    st.metric("ROCE %", f"{rv.iloc[-1]:.1f}%",
                              f"{rv.iloc[-1] - rv.iloc[-2]:+.1f}pp")
        with c6:
            if cf is not None and "Free Cash Flow" in cf.columns:
                fcf = cf["Free Cash Flow"].dropna()
                if len(fcf) >= 2:
                    st.metric("FCF (Cr)",
                              f"₹{fcf.iloc[-1]:,.0f}",
                              f"{((fcf.iloc[-1]-fcf.iloc[-2])/abs(fcf.iloc[-2])*100):.1f}% YoY"
                              if fcf.iloc[-2] != 0 else None)

    st.divider()

    # ── Growth CAGR cards
    st.subheader("Growth Snapshot")
    periods = [("10 Years:", "10Y"), ("5 Years:", "5Y"), ("3 Years:", "3Y")]
    gc1, gc2, gc3, gc4 = st.columns(4)

    def _cagr_col(col, tbl, label):
        col.markdown(f"**{label}**")
        for period, pl in periods:
            v = get_cagr(ticker, tbl, period)
            col.metric(pl, f"{v:.0f}%" if v is not None else "–")

    with gc1: _cagr_col(gc1, "sales_growth",  "📦 Sales CAGR")
    with gc2: _cagr_col(gc2, "profit_growth", "💰 Profit CAGR")
    with gc3: _cagr_col(gc3, "price_cagr",    "📉 Price CAGR")
    with gc4:
        gc4.markdown("**🔄 ROE**")
        for period, pl in periods:
            v = get_cagr(ticker, "roe_summary", period)
            gc4.metric(pl, f"{v:.0f}%" if v is not None else "–")

    st.divider()

    # ── 12-year sparklines
    if apl is not None:
        st.subheader("12-Year Trend")
        s1, s2, s3 = st.columns(3)

        def _spark(col_obj, df, col, title, color):
            if col in df.columns:
                fig = px.bar(x=df.index, y=df[col], title=title,
                             color_discrete_sequence=[color])
                fig.update_layout(height=240, showlegend=False,
                                  margin=dict(t=30, b=0, l=0, r=0))
                col_obj.plotly_chart(fig, use_container_width=True)

        _spark(s1, apl, "Sales",      "Revenue (₹ Cr)", BLUE)
        _spark(s2, apl, "Net Profit", "Net Profit (₹ Cr)", GREEN)
        if rat is not None and "ROCE %" in rat.columns:
            fig_r = px.line(x=rat.index, y=rat["ROCE %"],
                            title="ROCE %", markers=True)
            fig_r.update_layout(height=240, margin=dict(t=30, b=0, l=0, r=0))
            s3.plotly_chart(fig_r, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — P&L
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header(f"{name} — P&L Analysis")
    apl = D["annual_pl"]
    qpl = D["quarterly_pl"]

    view = st.radio("Period", ["Annual", "Quarterly"], horizontal=True)
    pl   = apl if view == "Annual" else qpl

    if pl is not None:
        st.subheader(f"{'Annual' if view == 'Annual' else 'Quarterly'} Revenue & Profit")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "Sales" in pl.columns:
            fig.add_trace(go.Bar(x=pl.index, y=pl["Sales"], name="Revenue",
                                 marker_color=BLUE, opacity=0.85), secondary_y=False)
        if "Net Profit" in pl.columns:
            fig.add_trace(go.Bar(x=pl.index, y=pl["Net Profit"], name="Net Profit",
                                 marker_color=GREEN, opacity=0.85), secondary_y=False)
        if "OPM %" in pl.columns:
            fig.add_trace(go.Scatter(x=pl.index, y=pl["OPM %"], name="OPM %",
                                     mode="lines+markers",
                                     line=dict(color=ORANGE, width=2)),
                          secondary_y=True)
        fig.update_layout(height=420, barmode="group", hovermode="x unified",
                          legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(title_text="₹ Crores", secondary_y=False)
        fig.update_yaxes(title_text="OPM %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        # Margins
        st.subheader("Margins")
        pl2 = pl.copy()
        if "Net Profit" in pl2.columns and "Sales" in pl2.columns:
            pl2["NPM %"] = (pl2["Net Profit"] / pl2["Sales"] * 100).round(1)
        margin_cols = [c for c in ["OPM %", "NPM %"] if c in pl2.columns]
        if margin_cols:
            fig2 = go.Figure()
            for col, color in zip(margin_cols, [ORANGE, GREEN]):
                fig2.add_trace(go.Scatter(x=pl2.index, y=pl2[col], name=col,
                                          mode="lines+markers", line=dict(width=2, color=color)))
            fig2.update_layout(height=300, hovermode="x unified", yaxis_title="%",
                               legend=dict(orientation="h"))
            st.plotly_chart(fig2, use_container_width=True)

        # EPS
        if "EPS in Rs" in pl.columns:
            st.subheader("EPS (₹)")
            fig3 = px.bar(x=pl.index, y=pl["EPS in Rs"],
                          color_discrete_sequence=[BLUE])
            fig3.update_layout(height=280, yaxis_title="₹")
            st.plotly_chart(fig3, use_container_width=True)

        # YoY growth
        if view == "Annual" and "Sales" in apl.columns and "Net Profit" in apl.columns:
            st.subheader("YoY Growth %")
            yoy = pd.DataFrame({
                "Revenue Growth": apl["Sales"].pct_change() * 100,
                "Profit Growth":  apl["Net Profit"].pct_change() * 100,
            }).dropna().round(1)
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=yoy.index, y=yoy["Revenue Growth"],
                                  name="Revenue", marker_color=BLUE, opacity=0.8))
            fig4.add_trace(go.Bar(x=yoy.index, y=yoy["Profit Growth"],
                                  name="Net Profit", marker_color=GREEN, opacity=0.8))
            fig4.add_hline(y=0, line_dash="dash", line_color="black")
            fig4.update_layout(height=320, barmode="group", hovermode="x unified",
                               yaxis_title="%", legend=dict(orientation="h"))
            st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BALANCE SHEET
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header(f"{name} — Balance Sheet")
    bs = D["balance_sheet"]

    if bs is not None:
        # Asset composition stacked bar
        st.subheader("Asset Composition")
        asset_cols = [c for c in ["Fixed Assets", "CWIP", "Investments", "Other Assets"]
                      if c in bs.columns]
        if asset_cols:
            fig = go.Figure()
            for col, color in zip(asset_cols, [BLUE, ORANGE, GREEN, "purple"]):
                fig.add_trace(go.Bar(x=bs.index, y=bs[col], name=col, marker_color=color))
            fig.update_layout(barmode="stack", height=380, hovermode="x unified",
                              yaxis_title="₹ Crores", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        # Debt vs Equity
        with c1:
            st.subheader("Debt vs Equity")
            eq_cols = [c for c in ["Equity Capital", "Reserves"] if c in bs.columns]
            equity  = bs[eq_cols].sum(axis=1)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=bs.index, y=equity,
                                  name="Equity + Reserves", marker_color=GREEN))
            if "Borrowings" in bs.columns:
                fig2.add_trace(go.Bar(x=bs.index, y=bs["Borrowings"],
                                      name="Borrowings", marker_color=RED))
            fig2.update_layout(height=340, barmode="group", hovermode="x unified",
                               yaxis_title="₹ Crores", legend=dict(orientation="h"))
            st.plotly_chart(fig2, use_container_width=True)

        # D/E ratio
        with c2:
            st.subheader("Debt / Equity Ratio")
            if "Borrowings" in bs.columns:
                de = (bs["Borrowings"] / equity.replace(0, np.nan)).round(2)
                fig3 = px.line(x=de.index, y=de, markers=True,
                               color_discrete_sequence=[RED])
                fig3.add_hline(y=1, line_dash="dash", line_color="gray",
                               annotation_text="D/E = 1")
                fig3.update_layout(height=340, yaxis_title="D/E Ratio")
                st.plotly_chart(fig3, use_container_width=True)

        # CWIP — capex signal
        if "CWIP" in bs.columns:
            st.subheader("Capital Work-in-Progress (Capex Signal)")
            fig4 = px.bar(x=bs.index, y=bs["CWIP"],
                          color_discrete_sequence=[ORANGE])
            fig4.update_layout(height=280, yaxis_title="₹ Crores")
            st.plotly_chart(fig4, use_container_width=True)

        # Total Assets growth
        if "Total Assets" in bs.columns:
            st.subheader("Total Assets Growth")
            fig5 = px.area(x=bs.index, y=bs["Total Assets"],
                           color_discrete_sequence=[BLUE])
            fig5.update_layout(height=260, yaxis_title="₹ Crores")
            st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CASH FLOW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header(f"{name} — Cash Flow")
    cf  = D["cash_flow"]
    apl = D["annual_pl"]

    if cf is not None:
        st.subheader("Cash Flow Components")
        fig = go.Figure()
        for col, color, short in [
            ("Cash from Operating Activity", GREEN,  "OCF"),
            ("Cash from Investing Activity", RED,    "ICF"),
            ("Cash from Financing Activity", BLUE,   "CFF"),
        ]:
            if col in cf.columns:
                fig.add_trace(go.Bar(x=cf.index, y=cf[col],
                                     name=short, marker_color=color, opacity=0.85))
        fig.update_layout(height=400, barmode="group", hovermode="x unified",
                          yaxis_title="₹ Crores", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Free Cash Flow")
            if "Free Cash Flow" in cf.columns:
                fcf    = cf["Free Cash Flow"].dropna()
                colors = [GREEN if v >= 0 else RED for v in fcf]
                fig2   = go.Figure(go.Bar(x=fcf.index, y=fcf, marker_color=colors))
                fig2.add_hline(y=0, line_dash="dash", line_color="black")
                fig2.update_layout(height=320, yaxis_title="₹ Crores",
                                   hovermode="x unified")
                st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.subheader("CFO / Operating Profit %")
            if "CFO/OP" in cf.columns:
                cfoop = cf["CFO/OP"].dropna()
                fig3  = px.bar(x=cfoop.index, y=cfoop,
                               color_discrete_sequence=[BLUE])
                fig3.add_hline(y=100, line_dash="dash", line_color="green",
                               annotation_text="100%")
                fig3.update_layout(height=320, yaxis_title="%")
                st.plotly_chart(fig3, use_container_width=True)

        # Earnings quality
        if apl is not None and "Net Profit" in apl.columns and \
                "Cash from Operating Activity" in cf.columns:
            st.subheader("Earnings Quality — OCF vs Net Profit")
            common = cf.index.intersection(apl.index)
            fig4   = go.Figure()
            fig4.add_trace(go.Scatter(
                x=list(common), y=list(apl.loc[common, "Net Profit"]),
                name="Net Profit", mode="lines+markers", line=dict(color=BLUE)))
            fig4.add_trace(go.Scatter(
                x=list(common),
                y=list(cf.loc[common, "Cash from Operating Activity"]),
                name="OCF", mode="lines+markers", line=dict(color=GREEN)))
            fig4.update_layout(height=340, hovermode="x unified",
                               yaxis_title="₹ Crores",
                               legend=dict(orientation="h"))
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
            if "ROCE %" in rat.columns:
                fig = px.line(x=rat.index, y=rat["ROCE %"], title="ROCE %",
                              markers=True, color_discrete_sequence=[BLUE])
                fig.update_layout(height=320, yaxis_title="%")
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "Cash Conversion Cycle" in rat.columns:
                ccc    = rat["Cash Conversion Cycle"].dropna()
                colors = [RED if v > 0 else GREEN for v in ccc]
                fig2   = go.Figure(go.Bar(x=ccc.index, y=ccc, marker_color=colors))
                fig2.add_hline(y=0, line_dash="dash")
                fig2.update_layout(height=320,
                                   title="Cash Conversion Cycle (days)",
                                   yaxis_title="Days")
                st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Working Capital Breakdown (Days)")
        wc_cols = [c for c in ["Debtor Days", "Inventory Days", "Days Payable"]
                   if c in rat.columns]
        if wc_cols:
            fig3 = go.Figure()
            for col, color in zip(wc_cols, [BLUE, ORANGE, RED]):
                fig3.add_trace(go.Scatter(x=rat.index, y=rat[col], name=col,
                                          mode="lines+markers",
                                          line=dict(width=2, color=color)))
            fig3.update_layout(height=360, hovermode="x unified", yaxis_title="Days",
                               legend=dict(orientation="h"))
            st.plotly_chart(fig3, use_container_width=True)

        if "Working Capital Days" in rat.columns:
            wcd    = rat["Working Capital Days"].dropna()
            colors = [RED if v > 0 else GREEN for v in wcd]
            fig4   = go.Figure(go.Bar(x=wcd.index, y=wcd, marker_color=colors))
            fig4.add_hline(y=0, line_dash="dash")
            fig4.update_layout(height=280, title="Net Working Capital Days",
                               yaxis_title="Days")
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

        st.subheader("Shareholding Composition")
        fig = go.Figure()
        sh_colors = [BLUE, GREEN, ORANGE, RED, "purple"]
        for col, color in zip(holders, sh_colors):
            fig.add_trace(go.Scatter(
                x=sh.index, y=sh[col], name=col,
                mode="lines", stackgroup="one", line=dict(width=0.5),
                fillcolor=color))
        fig.update_layout(height=400, hovermode="x unified", yaxis_title="%",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("FII vs DII Shift")
            if "FIIs" in sh.columns and "DIIs" in sh.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=sh.index, y=sh["FIIs"],
                                          name="FII", mode="lines+markers",
                                          line=dict(color=GREEN, width=2)))
                fig2.add_trace(go.Scatter(x=sh.index, y=sh["DIIs"],
                                          name="DII", mode="lines+markers",
                                          line=dict(color=BLUE, width=2)))
                fig2.update_layout(height=320, hovermode="x unified", yaxis_title="%",
                                   legend=dict(orientation="h"))
                st.plotly_chart(fig2, use_container_width=True)

        with c2:
            st.subheader("Promoter Holding %")
            if "Promoters" in sh.columns:
                fig3 = px.area(x=sh.index, y=sh["Promoters"],
                               color_discrete_sequence=[ORANGE])
                fig3.update_layout(height=320, yaxis_title="%")
                st.plotly_chart(fig3, use_container_width=True)

        if "No. of Shareholders" in sh.columns:
            st.subheader("Number of Shareholders")
            ns = sh["No. of Shareholders"].dropna()
            fig4 = px.bar(x=ns.index, y=ns, color_discrete_sequence=[BLUE])
            fig4.update_layout(height=280, yaxis_title="Count")
            st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SENSEX SCREENER
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("Sensex 30 — Screener")

    with st.spinner("Loading all 30 companies..."):
        sdf = screener_df()

    METRIC_COLS = [c for c in [
        "Revenue (Cr)", "Net Profit (Cr)", "OPM %", "NPM %", "ROCE %",
        "FCF (Cr)", "EPS (₹)", "Sales CAGR 5Y (%)", "Profit CAGR 5Y (%)",
        "Price CAGR 3Y (%)", "ROE (last yr)",
    ] if c in sdf.columns]

    # ── Sortable table
    st.subheader("All Companies — Key Metrics")
    col_sort, col_asc = st.columns([3, 1])
    sort_by  = col_sort.selectbox("Sort by", METRIC_COLS)
    asc      = col_asc.checkbox("Ascending", value=False)

    display = sdf.sort_values(sort_by, ascending=asc).reset_index(drop=True)

    st.dataframe(
        display[["Company", "Ticker"] + METRIC_COLS],
        use_container_width=True,
        height=600,
        hide_index=True,
    )

    st.divider()

    # ── Bar chart
    st.subheader("Compare All Companies")
    metric_bar = st.selectbox("Metric", METRIC_COLS, key="bar_metric")
    bar_data   = sdf[["Company", metric_bar]].dropna().sort_values(metric_bar)
    bar_colors = [GREEN if v >= 0 else RED for v in bar_data[metric_bar]]

    fig_b = go.Figure(go.Bar(
        x=bar_data[metric_bar],
        y=bar_data["Company"],
        orientation="h",
        marker_color=bar_colors,
    ))
    fig_b.update_layout(height=700, margin=dict(l=200), xaxis_title=metric_bar)
    st.plotly_chart(fig_b, use_container_width=True)

    st.divider()

    # ── Scatter: any X vs any Y
    st.subheader("Scatter — Any Metric vs Any Metric")
    s1, s2 = st.columns(2)
    x_ax = s1.selectbox("X axis", METRIC_COLS,
                         index=METRIC_COLS.index("Sales CAGR 5Y (%)") if "Sales CAGR 5Y (%)" in METRIC_COLS else 0,
                         key="sc_x")
    y_ax = s2.selectbox("Y axis", METRIC_COLS,
                         index=METRIC_COLS.index("ROCE %") if "ROCE %" in METRIC_COLS else 1,
                         key="sc_y")

    sc_data = sdf[["Company", x_ax, y_ax]].dropna()
    fig_s   = px.scatter(sc_data, x=x_ax, y=y_ax, text="Company",
                         hover_data=["Company"])
    fig_s.update_traces(textposition="top center", marker_size=9)
    fig_s.update_layout(height=520)
    st.plotly_chart(fig_s, use_container_width=True)
