import csv
import os
from datetime import datetime


TRADES_FILE = "trades.csv"


def initialize_trade_log():

    if not os.path.exists(TRADES_FILE):

        with open(TRADES_FILE, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                "timestamp",
                "side",
                "entry_price",
                "exit_price",
                "pnl"
            ])


def log_trade(

    side,
    entry,
    exit_price,
    pnl
):

    with open(TRADES_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            str(datetime.now()),
            side,
            entry,
            exit_price,
            round(pnl, 2)
        ])