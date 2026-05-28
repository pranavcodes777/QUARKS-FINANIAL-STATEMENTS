import pandas as pd
import requests
import os
from io import StringIO

# =========================================================
# CONFIG
# =========================================================

url = "https://www.screener.in/company/RELIANCE/consolidated/"

output_folder = r"E:\Quarks&Quants\Fundamental\Financial Statements\Database"

os.makedirs(output_folder, exist_ok=True)

# =========================================================
# FETCH ALL TABLES
# =========================================================

print("Fetching financial statements from Screener.in...")

tables = pd.read_html(url)

print(f"Total tables found: {len(tables)}")

# =========================================================
# TABLE EXTRACTION
# Screener.in consolidated page table layout:
#   0  -> Quarterly P&L         (quarterly, ~13 quarters)
#   1  -> Annual P&L            (annual, ~12 years)
#   2  -> Compounded Sales Growth (summary)
#   3  -> Compounded Profit Growth (summary)
#   4  -> Stock Price CAGR      (summary)
#   5  -> Return on Equity      (summary)
#   6  -> Balance Sheet         (annual, ~12 years)
#   7  -> Cash Flow Statement   (annual, ~12 years)
#   8  -> Key Ratios            (annual, ~12 years)
#   9  -> Segment Data          (paywalled - skipped)
#   10 -> Segment Data          (paywalled - skipped)
#   11 -> Shareholding Pattern  (quarterly, ~12 quarters)
#   12 -> Shareholding Pattern  (annual/historical)
# =========================================================

quarterly_pl        = tables[0]
annual_pl           = tables[1]
sales_growth        = tables[2]
profit_growth       = tables[3]
price_cagr          = tables[4]
roe_summary         = tables[5]
balance_sheet       = tables[6]
cash_flow           = tables[7]
ratios              = tables[8]
shareholding_qtr    = tables[11]
shareholding_annual = tables[12]

# =========================================================
# SAVE FILES
# =========================================================

files = {
    "reliance_quarterly_pl.csv":        quarterly_pl,
    "reliance_annual_pl.csv":           annual_pl,
    "reliance_sales_growth.csv":        sales_growth,
    "reliance_profit_growth.csv":       profit_growth,
    "reliance_price_cagr.csv":          price_cagr,
    "reliance_roe_summary.csv":         roe_summary,
    "reliance_balance_sheet.csv":       balance_sheet,
    "reliance_cash_flow.csv":           cash_flow,
    "reliance_ratios.csv":              ratios,
    "reliance_shareholding_qtr.csv":    shareholding_qtr,
    "reliance_shareholding_annual.csv": shareholding_annual,
}

for filename, df in files.items():
    path = os.path.join(output_folder, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {filename}  {df.shape}")

# =========================================================
# DONE
# =========================================================

print("\n===================================")
print("ALL DATA SAVED SUCCESSFULLY")
print("===================================")
print(f"\nLocation:\n{output_folder}")
