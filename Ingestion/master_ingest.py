"""
master_ingest.py
----------------
Add one or more equities to the database in one shot.
Fetches:
  1. Fundamental data  (screener.in)  -> Database/<TICKER>/*.parquet
  2. OHLCV price data  (Yahoo Finance) -> Database/OHLCV/<TICKER>.parquet

Usage
-----
# Add a single ticker (screener slug):
    python master_ingest.py HDFCBANK

# Add several tickers at once:
    python master_ingest.py HDFCBANK INFY RELIANCE

# Re-run with no args -- refreshes ALL existing tickers in the database:
    python master_ingest.py
"""

import os
import sys
import time
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

# =========================================================
# CONFIG
# =========================================================

DATABASE_DIR  = r"E:\Quarks&Quants\Fundamental\Financial Statements\Database"
OHLCV_DIR     = os.path.join(DATABASE_DIR, "OHLCV")
DELAY_SECONDS = 3   # polite crawl delay between screener.in requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# screener.in table index -> parquet filename
TABLE_MAP = {
    0:  "quarterly_pl",
    1:  "annual_pl",
    2:  "sales_growth",
    3:  "profit_growth",
    4:  "price_cagr",
    5:  "roe_summary",
    6:  "balance_sheet",
    7:  "cash_flow",
    8:  "ratios",
    11: "shareholding_qtr",
    12: "shareholding_annual",
}

# Screener slug -> Yahoo Finance NSE ticker overrides
# Only needed when the slug differs from "<SLUG>.NS"
TICKER_OVERRIDES = {
    "BAJAJ-AUTO":   "BAJAJ-AUTO.NS",
    "M&M":          "M&M.NS",
    "TATACONSUMER": "TATACONSUM.NS",    # Yahoo Finance uses TATACONSUM
    "TATAMOTORS":   "TATAMOTORS.NS",
    "JINDALSTEL":   "JINDALSTEL.NS",
    "NAUKRI":       "NAUKRI.NS",
    "DMART":        "DMART.NS",
    "LICI":         "LICI.NS",
    "IRCTC":        "IRCTC.NS",
}


# =========================================================
# HELPERS — FUNDAMENTALS
# =========================================================

def fetch_screener_tables(session, ticker: str, consolidated: bool):
    suffix = "consolidated/" if consolidated else ""
    url = f"https://www.screener.in/company/{ticker}/{suffix}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text)), url


def save_fundamentals(tables: list, ticker: str) -> list[str]:
    ticker_dir = os.path.join(DATABASE_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    saved = []
    for idx, name in TABLE_MAP.items():
        if idx < len(tables):
            tables[idx].to_parquet(os.path.join(ticker_dir, f"{name}.parquet"), index=False)
            saved.append(name)
    return saved


def ingest_fundamentals(session, ticker: str) -> tuple[bool, str]:
    """Try consolidated first, fall back to standalone. Returns (success, message)."""
    for consolidated in (True, False):
        try:
            tables, _ = fetch_screener_tables(session, ticker, consolidated)
            if len(tables) < 9 and consolidated:
                continue   # thin consolidated page — retry standalone
            saved = save_fundamentals(tables, ticker)
            label = "consolidated" if consolidated else "standalone"
            return True, f"{len(saved)} tables ({label})"
        except requests.HTTPError as e:
            if e.response.status_code == 404 and consolidated:
                continue   # no consolidated page, try standalone
            return False, str(e)
        except Exception as e:
            return False, str(e)
    return False, "both consolidated and standalone failed"


# =========================================================
# HELPERS — OHLCV
# =========================================================

def get_yf_ticker(screener_slug: str) -> str:
    return TICKER_OVERRIDES.get(screener_slug, f"{screener_slug}.NS")


def ingest_ohlcv(ticker: str) -> tuple[bool, str]:
    yf_ticker = get_yf_ticker(ticker)
    try:
        df = yf.Ticker(yf_ticker).history(period="max", interval="1d", auto_adjust=True)
        if df.empty:
            return False, f"no data returned from Yahoo Finance for {yf_ticker}"
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        os.makedirs(OHLCV_DIR, exist_ok=True)
        df.to_parquet(os.path.join(OHLCV_DIR, f"{ticker}.parquet"))
        return True, f"{len(df)} rows  ({df.index[0].date()} -> {df.index[-1].date()})"
    except Exception as e:
        return False, str(e)


# =========================================================
# DISCOVER EXISTING TICKERS
# =========================================================

def discover_existing_tickers() -> list[str]:
    return sorted(
        e for e in os.listdir(DATABASE_DIR)
        if os.path.isdir(os.path.join(DATABASE_DIR, e)) and e != "OHLCV"
    )


# =========================================================
# MAIN
# =========================================================

def run(tickers: list[str]):
    total   = len(tickers)
    results = {"success": [], "failed_fundamentals": [], "failed_ohlcv": []}
    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"\nIngesting {total} ticker(s)\n")
    print("=" * 65)

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i:02d}/{total}] {ticker}")

        # --- Fundamentals ---
        print(f"  Fundamentals  ", end="")
        ok, msg = ingest_fundamentals(session, ticker)
        if ok:
            print(f"OK — {msg}")
        else:
            print(f"FAILED — {msg}")
            results["failed_fundamentals"].append((ticker, msg))

        # --- OHLCV ---
        print(f"  OHLCV         ", end="")
        ok, msg = ingest_ohlcv(ticker)
        if ok:
            print(f"OK — {msg}")
        else:
            print(f"FAILED — {msg}")
            results["failed_ohlcv"].append((ticker, msg))

        if ok or True:   # always count as attempted
            results["success"].append(ticker)

        if i < total:
            time.sleep(DELAY_SECONDS)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("INGESTION COMPLETE")
    print("=" * 65)
    ff = results["failed_fundamentals"]
    fo = results["failed_ohlcv"]
    print(f"  Tickers processed : {total}")
    print(f"  Fundamentals failed : {len(ff)}")
    print(f"  OHLCV failed        : {len(fo)}")

    if ff:
        print("\nFailed fundamentals:")
        for t, e in ff:
            print(f"  - {t}: {e}")
    if fo:
        print("\nFailed OHLCV (check Yahoo Finance slug / TICKER_OVERRIDES):")
        for t, e in fo:
            print(f"  - {t} -> {get_yf_ticker(t)}: {e}")

    print(f"\nDatabase : {DATABASE_DIR}")
    print(f"OHLCV    : {OHLCV_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tickers_to_run = [t.upper() for t in sys.argv[1:]]
        print(f"Tickers from args: {tickers_to_run}")
    else:
        tickers_to_run = discover_existing_tickers()
        print(f"No args — refreshing all {len(tickers_to_run)} existing tickers.")

    run(tickers_to_run)