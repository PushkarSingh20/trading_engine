"""
walkforward_runner.py
=====================
Stage 12 — Walk-Forward Robustness Validation CLI Runner

Executes the walk-forward evaluation using WalkForwardEngine:
  1. Configurable parameters (n_splits, expanding/rolling mode, warmup).
  2. Runs backtests across chronological OOS windows in isolated processes.
  3. Prepares consolidated metrics and pooled OOS trades.
  4. Triggers WalkForwardReport for comprehensive quantitative analysis.

Usage:
  python -m backtesting.walkforward_runner
"""

import os
import argparse
import pandas as pd

from backtesting.walkforward import WalkForwardEngine
from backtesting.walkforward_report import WalkForwardReport


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_FILE = os.path.join(
    BASE_DIR,
    "historical_data.csv"
)

DEFAULT_WINDOW_DIR = os.path.join(
    BASE_DIR,
    "walkforward_data",
    "windows"
)

DEFAULT_ANALYTICS_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "analytics",
    "walkforward"
)


def run_walk_forward(
    data_file=DEFAULT_DATA_FILE,
    window_dir=DEFAULT_WINDOW_DIR,
    analytics_dir=DEFAULT_ANALYTICS_DIR,
    n_splits=5,
    mode="expanding",
    train_bars=None,
    oos_bars=None,
    oos_fraction=0.10,
    warmup_bars=250
):
    print("\n" + "=" * 65)
    print(" STAGE 12: WALK-FORWARD ROBUSTNESS VALIDATION RUNNER")
    print("=" * 65)
    print(f"  Data File     : {data_file}")
    print(f"  Windows       : {n_splits}")
    print(f"  Training Mode : {mode.upper()}")
    print(f"  OOS Fraction  : {oos_fraction * 100:.0f}%")
    print(f"  Warm-up Bars  : {warmup_bars}")
    print("=" * 65)

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} rows.")

    engine = WalkForwardEngine(
        dataframe=df,
        datetime_column="Datetime",
        n_splits=n_splits,
        mode=mode,
        train_bars=train_bars,
        oos_bars=oos_bars,
        oos_fraction=oos_fraction,
        warmup_bars=warmup_bars
    )

    consolidated_df, combined_trades_df = engine.run(
        window_data_dir=window_dir,
        analytics_dir=analytics_dir
    )

    # Generate and print the Stage 12 report
    report = WalkForwardReport(analytics_dir=analytics_dir)
    if report.load():
        report.print_report()
        report.save()

    return consolidated_df, combined_trades_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Stage 12 Walk-Forward Robustness Validation"
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_FILE,
        help="Path to historical CSV data"
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of OOS splits/windows (default: 5)"
    )
    parser.add_argument(
        "--mode",
        choices=["expanding", "rolling"],
        default="expanding",
        help="Training window mode: expanding (default) or rolling"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=250,
        help="Number of warm-up bars for indicator initialization (default: 250)"
    )
    parser.add_argument(
        "--oos-fraction",
        type=float,
        default=0.10,
        help="Fraction of dataset per OOS window (default: 0.10)"
    )

    args = parser.parse_args()

    run_walk_forward(
        data_file=args.data,
        n_splits=args.splits,
        mode=args.mode,
        warmup_bars=args.warmup,
        oos_fraction=args.oos_fraction
    )