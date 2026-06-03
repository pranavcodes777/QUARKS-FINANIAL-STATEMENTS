import os
import time
import yfinance as yf
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

DATABASE_DIR  = r"E:\Quarks&Quants\Fundamental\Financial Statements\Database"
OHLCV_DIR     = os.path.join(DATABASE_DIR, "OHLCV")
DELAY_SECONDS = 1

# Screener slug -> Yahoo Finance NSE ticker overrides
# Only entries that differ from "<SLUG>.NS" need to be listed here.
TICKER_OVERRIDES = {
    "BAJAJ-AUTO":   "BAJAJ-AUTO.NS",
    "M&M":          "M&M.NS",
    "TATACONSUMER": "TATACONSUM.NS",   # Yahoo Finance uses TATACONSUM
    "TATAMOTORS":   "TATAMOTORS.NS",
    "ZOMATO":       "ZOMATO.NS",
    "JINDALSTEL":   "JINDALSTEL.NS",
    "NAUKRI":       "NAUKRI.NS",
    "DMART":        "DMART.NS",
    "LICI":         "LICI.NS",
    "IRCTC":        "IRCTC.NS",
}


def get_yf_ticker(screener_slug: str) -> str:
    return TICKER_OVERRIDES.get(screener_slug, f"{screener_slug}.NS")


def fetch_ohlcv(yf_ticker: str) -> pd.DataFrame:
    ticker_obj = yf.Ticker(yf_ticker)
    df = ticker_obj.history(period="max", interval="1d", auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {yf_ticker}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df


def discover_tickers() -> list[str]:
    entries = os.listdir(DATABASE_DIR)
    return sorted(
        e for e in entries
        if os.path.isdir(os.path.join(DATABASE_DIR, e)) and e != "OHLCV"
    )


# =========================================================
# MAIN
# =========================================================

os.makedirs(OHLCV_DIR, exist_ok=True)

tickers = discover_tickers()
total   = len(tickers)
results = {"success": [], "failed": []}

print(f"Fetching OHLCV for {total} equities -> {OHLCV_DIR}\n")
print("=" * 60)

for i, slug in enumerate(tickers, 1):
    yf_ticker = get_yf_ticker(slug)
    out_path  = os.path.join(OHLCV_DIR, f"{slug}.parquet")
    print(f"[{i:02d}/{total}] {slug:20s} ({yf_ticker})", end="  ")

    try:
        df = fetch_ohlcv(yf_ticker)
        df.to_parquet(out_path)
        start = df.index[0].date()
        end   = df.index[-1].date()
        print(f"OK -- {len(df)} rows  ({start} to {end})")
        results["success"].append(slug)
    except Exception as e:
        print(f"FAILED -- {e}")
        results["failed"].append((slug, str(e)))

    if i < total:
        time.sleep(DELAY_SECONDS)

# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("OHLCV DOWNLOAD COMPLETE")
print("=" * 60)
print(f"  Success : {len(results['success'])} / {total}")
print(f"  Failed  : {len(results['failed'])}")

if results["failed"]:
    print("\nFailed tickers (check Yahoo Finance slug):")
    for slug, err in results["failed"]:
        print(f"  - {slug} -> {get_yf_ticker(slug)}: {err}")

print(f"\nFiles saved to: {OHLCV_DIR}")