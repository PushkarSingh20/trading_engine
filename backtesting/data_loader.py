import os
import yfinance as yf


# ==========================
# CONFIG
# ==========================

TICKER = "RELIANCE.NS"

PERIOD = "730d"

INTERVAL = "1h"

OUTPUT_FILE = "backtesting/historical_data.csv"


# ==========================
# CREATE DIRECTORY
# ==========================

os.makedirs("backtesting", exist_ok=True)


# ==========================
# DOWNLOAD DATA
# ==========================

print("\nDownloading historical data...\n")

df = yf.download(
    TICKER,
    period=PERIOD,
    interval=INTERVAL,
    auto_adjust=True
)


# ==========================
# VALIDATION
# ==========================

if len(df) == 0:

    print("No data downloaded.")

    exit()


# ==========================
# SAVE DATA
# ==========================
OUTPUT_FILE = "backtesting/historical_data.csv"
# RESET INDEX
# =========================
# SAVE DATA
# =========================

df = df.reset_index()

print("\nFinal Columns:")
print(df.columns)

df.to_csv(
    OUTPUT_FILE,
    index=False
)
print(df.head())

print("\n==========================")
print("Historical data saved")
print(f"Rows: {len(df)}")
print(f"File: {OUTPUT_FILE}")
print("==========================")