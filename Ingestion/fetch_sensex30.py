import pandas as pd
import requests
import os
import time
from io import StringIO

# =========================================================
# CONFIG
# =========================================================

OUTPUT_DIR = r"E:\Quarks&Quants\Fundamental\Financial Statements\Database"

DELAY_SECONDS = 3  # polite crawl delay between companies

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Sensex 30 — screener.in tickers + consolidated flag
# consolidated=False for companies that are single-entity (no subsidiaries)
SENSEX_30 = [
    ("ADANIPORTS",  True),
    ("ASIANPAINT",  True),
    ("AXISBANK",    True),
    ("BAJFINANCE",  True),
    ("BAJAJFINSV",  True),
    ("BHARTIARTL",  True),
    ("HCLTECH",     True),
    ("HDFCBANK",    True),
    ("HINDUNILVR",  True),
    ("ICICIBANK",   True),
    ("INDUSINDBK",  True),
    ("INFY",        True),
    ("ITC",         True),
    ("JSWSTEEL",    True),
    ("KOTAKBANK",   True),
    ("LT",          True),
    ("M&M",         True),
    ("MARUTI",      True),
    ("NTPC",        True),
    ("POWERGRID",   True),
    ("RELIANCE",    True),
    ("SBIN",        True),
    ("SUNPHARMA",   True),
    ("TCS",         True),
    ("TMCV",        True),   # Tata Motors Ltd (screener slug; NSE: TATAMOTORS)
    ("TATASTEEL",   True),
    ("TECHM",       True),
    ("TITAN",       True),
    ("ULTRACEMCO",  True),
    ("WIPRO",       True),
]

# Screener.in table layout (standardised across companies):
#   0  -> Quarterly P&L         (quarterly, ~12 quarters)
#   1  -> Annual P&L            (annual, ~12 years)
#   2  -> Compounded Sales Growth
#   3  -> Compounded Profit Growth
#   4  -> Stock Price CAGR
#   5  -> Return on Equity summary
#   6  -> Balance Sheet         (annual)
#   7  -> Cash Flow Statement   (annual)
#   8  -> Key Ratios            (annual)
#   9  -> Segment data          (paywalled — skipped)
#   10 -> Segment data          (paywalled — skipped)
#   11 -> Shareholding Pattern  (quarterly)
#   12 -> Shareholding Pattern  (annual / historical)
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

# =========================================================
# HELPERS
# =========================================================

def fetch_tables(session, ticker, consolidated):
    suffix = "consolidated/" if consolidated else ""
    url = f"https://www.screener.in/company/{ticker}/{suffix}"
    response = session.get(url, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    return tables, url


def save_company(tables, ticker):
    ticker_dir = os.path.join(OUTPUT_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    saved = []
    for idx, name in TABLE_MAP.items():
        if idx < len(tables):
            path = os.path.join(ticker_dir, f"{name}.parquet")
            tables[idx].to_parquet(path, index=False)
            saved.append(name)
    return saved

# =========================================================
# MAIN LOOP
# =========================================================

session = requests.Session()
session.headers.update(HEADERS)

results = {"success": [], "failed": []}
total = len(SENSEX_30)

print(f"Starting ingestion — {total} companies\n")
print("=" * 55)

for i, (ticker, consolidated) in enumerate(SENSEX_30, 1):
    print(f"\n[{i:02d}/{total}] {ticker}", end="  ")

    try:
        tables, url = fetch_tables(session, ticker, consolidated)

        # fallback: if consolidated returned < 9 tables, try standalone
        if len(tables) < 9 and consolidated:
            print(f"(consolidated thin, retrying standalone)", end="  ")
            tables, url = fetch_tables(session, ticker, consolidated=False)

        saved = save_company(tables, ticker)
        print(f"OK — {len(tables)} tables found, {len(saved)} saved")
        results["success"].append(ticker)

    except requests.HTTPError as e:
        if e.response.status_code == 404 and consolidated:
            # consolidated page doesn't exist — retry standalone
            print(f"(consolidated 404, retrying standalone)", end="  ")
            try:
                tables, url = fetch_tables(session, ticker, consolidated=False)
                saved = save_company(tables, ticker)
                print(f"OK — {len(tables)} tables found, {len(saved)} saved")
                results["success"].append(ticker)
            except Exception as e2:
                print(f"FAILED — {e2}")
                results["failed"].append((ticker, str(e2)))
        else:
            print(f"FAILED — {e}")
            results["failed"].append((ticker, str(e)))

    except Exception as e:
        print(f"FAILED — {e}")
        results["failed"].append((ticker, str(e)))

    if i < total:
        time.sleep(DELAY_SECONDS)

# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 55)
print("INGESTION COMPLETE")
print("=" * 55)
print(f"  Success : {len(results['success'])} / {total}")
print(f"  Failed  : {len(results['failed'])}")

if results["failed"]:
    print("\nFailed companies:")
    for ticker, err in results["failed"]:
        print(f"  - {ticker}: {err}")

print(f"\nDatabase:\n  {OUTPUT_DIR}")
print("\nFolder structure:")
print("  DATABASE/")
print("    <TICKER>/")
print("      quarterly_pl.parquet")
print("      annual_pl.parquet")
print("      balance_sheet.parquet")
print("      cash_flow.parquet")
print("      ratios.parquet")
print("      shareholding_qtr.parquet")
print("      shareholding_annual.parquet")
print("      sales_growth.parquet")
print("      profit_growth.parquet")
print("      price_cagr.parquet")
print("      roe_summary.parquet")
