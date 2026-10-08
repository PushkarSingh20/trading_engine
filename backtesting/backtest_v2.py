"""
backtesting/backtest_v2.py — Strategy V2 Backtesting Engine

Runs a bar-by-bar simulation of Strategy V2 (EMA5/EMA9 Trend Reversal)
over a given OHLCV CSV file.

This is a SEPARATE engine from backtesting/backtest.py (Strategy V1).
Strategy V1 logic, parameters, and outputs are NOT modified here.

DATA BLOCKER NOTE
-----------------
Strategy V2 requires 3-minute candle data.  No such data currently exists
in this repository (only 1h RELIANCE.NS data).  This engine will refuse to
run on 1h data and print a clear error message.  To run this engine, you
must supply a CSV with genuine 3-minute bars.

Usage
-----
    python -m backtesting.backtest_v2 \\
        --data backtesting/data_3min.csv \\
        --output analytics/v2/ \\
        [--ema-fast 5] \\
        [--ema-slow 9] \\
        [--swing-lookback 30] \\
        [--swing-strength 3] \\
        [--min-rr 1.0] \\
        [--sl-buffer 0.0] \\
        [--sl-min-dist 0.50] \\
        [--warmup 50] \\
        [--capital 100000] \\
        [--risk-per-trade 0.01] \\
        [--no-ema9-confirm]  # disable EMA9 confirmation

Output files (in --output directory)
-------------------------------------
    v2_trades.csv      : per-trade log with pattern, EMAs, entry/exit detail
    v2_summary.csv     : aggregate performance metrics
"""

import argparse
import os
import sys
import math

import pandas as pd
import numpy as np

# V2 strategy module
from strategies.strategy_v2_ema_reversal import (
    generate_signals,
    calculate_position_size,
)

# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_INTERVAL_MINUTES = 3   # Refuse to run on non-3-minute data
INTERVAL_TOLERANCE_SECONDS = 30  # Allow a small tolerance in bar spacing


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data(df: pd.DataFrame, data_path: str) -> None:
    """
    Validate the input DataFrame.  Raises SystemExit with a clear message
    if the data does not appear to be 3-minute bars.

    This is NOT a filter — it is a hard blocker to prevent silent errors.
    """
    required_cols = {"Open", "High", "Low", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        print(
            f"\n[BLOCKER] Missing required columns: {missing}\n"
            f"File: {data_path}\n"
        )
        sys.exit(1)

    if len(df) < 2:
        print(
            f"\n[BLOCKER] Insufficient rows ({len(df)}) in {data_path}.\n"
            "Need at least 2 bars to compute intervals.\n"
        )
        sys.exit(1)

    # Check bar interval via DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        print(
            "\n[WARNING] DataFrame index is not a DatetimeIndex. "
            "Cannot verify bar interval. Proceeding with caution.\n"
        )
        return

    median_interval_seconds = (
        df.index.to_series().diff().dropna().median().total_seconds()
    )
    expected_seconds = EXPECTED_INTERVAL_MINUTES * 60
    tolerance = INTERVAL_TOLERANCE_SECONDS

    if abs(median_interval_seconds - expected_seconds) > tolerance:
        median_minutes = median_interval_seconds / 60
        print(
            f"\n[BLOCKER] DATA TIMEFRAME MISMATCH\n"
            f"{'='*60}\n"
            f"Strategy V2 requires 3-minute candles.\n"
            f"Detected bar interval: ~{median_minutes:.1f} minutes "
            f"({median_interval_seconds:.0f} seconds).\n"
            f"File: {data_path}\n\n"
            f"The existing backtesting/historical_data.csv contains only\n"
            f"1-hour (60-minute) bars and CANNOT be used for Strategy V2.\n\n"
            f"ACTION REQUIRED: Download or provide genuine 3-minute OHLCV\n"
            f"data for RELIANCE.NS and supply it via --data.\n"
            f"{'='*60}\n"
        )
        sys.exit(1)


def load_data(data_path: str) -> pd.DataFrame:
    """Load and prepare OHLCV data from CSV."""
    df = pd.read_csv(data_path, header=0)

    # Try to parse datetime column
    for col in ["Datetime", "Date", "timestamp", "time", "index"]:
        if col in df.columns:
            try:
                df.index = pd.to_datetime(df[col])
                df = df.drop(columns=[col])
            except Exception:
                pass
            break
    else:
        # Last resort: try first column
        try:
            df.index = pd.to_datetime(df.iloc[:, 0])
            df = df.iloc[:, 1:]
        except Exception:
            pass

    df = df.sort_index()

    # Flatten multi-level columns if present (yfinance style)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    df: pd.DataFrame,
    ema_fast_period: int = 5,
    ema_slow_period: int = 9,
    require_ema9: bool = True,
    swing_lookback: int = 30,
    swing_strength: int = 3,
    min_rr_ratio: float = 1.0,
    sl_buffer_fraction: float = 0.0,
    sl_min_distance: float = 0.50,
    warmup_bars: int = 50,
    capital: float = 100_000,
    risk_per_trade: float = 0.01,
) -> tuple:
    """
    Run V2 bar-by-bar simulation.

    Returns (trades_df, summary_dict).

    Simulation rules
    ----------------
    - One position at a time (no pyramiding).
    - Entry: open of bar N+1 after reversal candle N.         [RULE]
    - Exit: checked bar-by-bar on close:
        * If close crosses stop_loss → exit at close (conservative fill).
        * If close crosses target → exit at close (conservative fill).
        * End of data → exit at last close (forced liquidation).
    - PnL: LONG  = (exit_price - entry_price) * size
            SHORT = (entry_price - exit_price) * size
    - No costs applied here (use cost_model.py for cost scenarios).
    """
    # --- Generate all potential signals ---
    signals = generate_signals(
        df,
        ema_fast_period=ema_fast_period,
        ema_slow_period=ema_slow_period,
        require_ema9=require_ema9,
        swing_lookback=swing_lookback,
        swing_strength=swing_strength,
        min_rr_ratio=min_rr_ratio,
        sl_buffer_fraction=sl_buffer_fraction,
        sl_min_distance=sl_min_distance,
        warmup_bars=warmup_bars,
    )

    valid_signals = signals[signals["valid"]].reset_index(drop=True)

    opens = df["Open"].reset_index(drop=True)
    highs = df["High"].reset_index(drop=True)
    lows = df["Low"].reset_index(drop=True)
    closes = df["Close"].reset_index(drop=True)

    times = None
    if isinstance(df.index, pd.DatetimeIndex):
        times = df.index

    trades = []
    current_capital = capital
    in_position = False
    pos_direction = None
    pos_entry_bar = None
    pos_entry_price = None
    pos_stop = None
    pos_target = None
    pos_size = None
    pos_pattern = None
    pos_ema5 = None
    pos_ema9 = None
    pos_signal_bar = None
    sig_ptr = 0  # pointer into valid_signals

    n = len(df)

    for i in range(n):
        if in_position:
            c = float(closes.iloc[i])
            h = float(highs.iloc[i])
            l = float(lows.iloc[i])
            exit_price = None
            exit_reason = None

            if pos_direction == "LONG":
                if l <= pos_stop:
                    exit_price = float(closes.iloc[i])  # conservative fill
                    exit_reason = "STOP_LOSS"
                elif h >= pos_target:
                    exit_price = float(closes.iloc[i])
                    exit_reason = "TARGET"
            else:  # SHORT
                if h >= pos_stop:
                    exit_price = float(closes.iloc[i])
                    exit_reason = "STOP_LOSS"
                elif l <= pos_target:
                    exit_price = float(closes.iloc[i])
                    exit_reason = "TARGET"

            # Force-exit at end of data
            if exit_price is None and i == n - 1:
                exit_price = float(closes.iloc[i])
                exit_reason = "EOD_LIQUIDATION"

            if exit_price is not None:
                if pos_direction == "LONG":
                    pnl = (exit_price - pos_entry_price) * pos_size
                else:
                    pnl = (pos_entry_price - exit_price) * pos_size

                current_capital += pnl

                entry_time = (
                    str(times[pos_entry_bar]) if times is not None else str(pos_entry_bar)
                )
                exit_time = (
                    str(times[i]) if times is not None else str(i)
                )

                trades.append({
                    "signal_bar": pos_signal_bar,
                    "entry_bar": pos_entry_bar,
                    "exit_bar": i,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "direction": pos_direction,
                    "pattern": pos_pattern,
                    "entry_price": pos_entry_price,
                    "exit_price": exit_price,
                    "stop_loss": pos_stop,
                    "target": pos_target,
                    "size": pos_size,
                    "pnl": round(pnl, 2),
                    "exit_reason": exit_reason,
                    "ema5_at_signal": pos_ema5,
                    "ema9_at_signal": pos_ema9,
                    "capital_after": round(current_capital, 2),
                })
                in_position = False

        # Check if a new signal triggers at bar i (entry on bar i's open)
        if not in_position:
            while (
                sig_ptr < len(valid_signals)
                and valid_signals.loc[sig_ptr, "entry_bar"] < i
            ):
                sig_ptr += 1  # skip signals whose entry bar has passed

            if (
                sig_ptr < len(valid_signals)
                and valid_signals.loc[sig_ptr, "entry_bar"] == i
            ):
                sig = valid_signals.loc[sig_ptr]
                entry_price = float(opens.iloc[i])  # use actual open, not signal open
                direction = sig["direction"]
                sl = sig["stop_loss"]
                target = sig["target"]

                size = calculate_position_size(
                    current_capital, risk_per_trade, entry_price, sl
                )

                if size > 0:
                    in_position = True
                    pos_direction = direction
                    pos_entry_bar = i
                    pos_entry_price = entry_price
                    pos_stop = sl
                    pos_target = target
                    pos_size = size
                    pos_pattern = sig["pattern"]
                    pos_ema5 = sig["ema5"]
                    pos_ema9 = sig["ema9"]
                    pos_signal_bar = sig["signal_bar"]

                sig_ptr += 1

    trades_df = pd.DataFrame(trades)
    summary = _compute_summary(trades_df, capital, current_capital)
    return trades_df, summary


# ============================================================
# SUMMARY METRICS
# ============================================================

def _compute_summary(trades_df: pd.DataFrame, initial_capital: float, final_capital: float) -> dict:
    if len(trades_df) == 0:
        return {
            "total_trades": 0,
            "win_rate": None,
            "total_pnl": 0.0,
            "profit_factor": None,
            "expectancy": None,
            "max_drawdown": None,
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "sample_adequacy": "VERY_LOW_SAMPLE",
        }

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    n = len(trades_df)
    win_rate = len(wins) / n
    total_pnl = trades_df["pnl"].sum()
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = total_pnl / n

    # Drawdown
    cumulative = trades_df["pnl"].cumsum() + initial_capital
    rolling_max = cumulative.cummax()
    drawdown = rolling_max - cumulative
    max_drawdown = drawdown.max()

    # Sample adequacy flags
    if n < 10:
        adequacy = "VERY_LOW_SAMPLE"
    elif n < 30:
        adequacy = "LOW_SAMPLE"
    else:
        adequacy = "ADEQUATE_FOR_BASIC_REVIEW"

    return {
        "total_trades": n,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "profit_factor": round(profit_factor, 4),
        "expectancy": round(expectancy, 2),
        "max_drawdown": round(max_drawdown, 2),
        "initial_capital": initial_capital,
        "final_capital": round(final_capital, 2),
        "sample_adequacy": adequacy,
    }


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Strategy V2 (EMA5/EMA9 Reversal) Backtesting Engine"
    )
    p.add_argument("--data", required=True, help="Path to 3-minute OHLCV CSV")
    p.add_argument("--output", default="analytics/v2/", help="Output directory")
    p.add_argument("--ema-fast", type=int, default=5)
    p.add_argument("--ema-slow", type=int, default=9)
    p.add_argument("--swing-lookback", type=int, default=30)
    p.add_argument("--swing-strength", type=int, default=3)
    p.add_argument("--min-rr", type=float, default=1.0)
    p.add_argument("--sl-buffer", type=float, default=0.0)
    p.add_argument("--sl-min-dist", type=float, default=0.50)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--capital", type=float, default=100_000)
    p.add_argument("--risk-per-trade", type=float, default=0.01)
    p.add_argument(
        "--no-ema9-confirm",
        action="store_true",
        default=False,
        help="Disable EMA9 confirmation (use EMA5 only)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n[V2 Backtest] Loading data: {args.data}")
    df = load_data(args.data)

    print(f"[V2 Backtest] Rows loaded: {len(df)}")
    validate_data(df, args.data)

    print("[V2 Backtest] Running simulation...")
    trades_df, summary = run_backtest(
        df,
        ema_fast_period=args.ema_fast,
        ema_slow_period=args.ema_slow,
        require_ema9=not args.no_ema9_confirm,
        swing_lookback=args.swing_lookback,
        swing_strength=args.swing_strength,
        min_rr_ratio=args.min_rr,
        sl_buffer_fraction=args.sl_buffer,
        sl_min_distance=args.sl_min_dist,
        warmup_bars=args.warmup,
        capital=args.capital,
        risk_per_trade=args.risk_per_trade,
    )

    os.makedirs(args.output, exist_ok=True)
    trades_path = os.path.join(args.output, "v2_trades.csv")
    summary_path = os.path.join(args.output, "v2_summary.csv")

    trades_df.to_csv(trades_path, index=False)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_path, index=False)

    print("\n[V2 Backtest] Results")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")
    print(f"\nTrades written to : {trades_path}")
    print(f"Summary written to: {summary_path}\n")


if __name__ == "__main__":
    main()
