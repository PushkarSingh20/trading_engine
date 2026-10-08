"""
walkforward.py
==============
Stage 12 — Walk-Forward Robustness Validation Engine

Provides WalkForwardEngine:
  The authoritative walk-forward orchestration engine that:
    1. Generates chronological expanding or rolling walk-forward windows
       via DatasetSplitter.
    2. Enforces warm-up handling so indicators (EMA200, HMA, ATR, ADX, VROC)
       have valid historical context without contaminating OOS trades.
    3. Executes each OOS window in complete subprocess isolation (zero capital
       or state leakage).
    4. Evaluates sample adequacy using exact Stage 12 review criteria:
         - < 10 trades  : VERY_LOW_SAMPLE
         - 10–29 trades : LOW_SAMPLE
         - >= 30 trades : ADEQUATE_FOR_BASIC_REVIEW
    5. Aggregates combined chronological OOS trade records to evaluate
       pooled out-of-sample performance without averaging Sharpe ratios.
"""

import os
import sys
import subprocess
import pandas as pd

from backtesting.dataset_splitter import DatasetSplitter
from backtesting.performance import PerformanceAnalyzer


def classify_sample_adequacy(trade_count):
    """
    Classifies window trade sample size according to Stage 12 standards.
    These are evidence-quality labels, NOT guarantees of statistical significance.
    """
    if trade_count < 10:
        return "VERY_LOW_SAMPLE"
    elif trade_count < 30:
        return "LOW_SAMPLE"
    else:
        return "ADEQUATE_FOR_BASIC_REVIEW"


class WalkForwardEngine:

    def __init__(
        self,
        dataframe,
        datetime_column="Datetime",
        n_splits=5,
        mode="expanding",
        train_bars=None,
        oos_bars=None,
        oos_fraction=0.10,
        warmup_bars=250
    ):
        self.dataframe = dataframe.copy()
        self.datetime_column = datetime_column
        self.n_splits = n_splits
        self.mode = mode
        self.train_bars = train_bars
        self.oos_bars = oos_bars
        self.oos_fraction = oos_fraction
        self.warmup_bars = warmup_bars

        self.splitter = DatasetSplitter(
            self.dataframe,
            datetime_column=self.datetime_column
        )

        self.windows = None

    # ==========================================
    # WINDOW GENERATION
    # ==========================================
    def generate_windows(self):
        self.windows = self.splitter.walk_forward_windows(
            n_splits=self.n_splits,
            mode=self.mode,
            train_bars=self.train_bars,
            oos_bars=self.oos_bars,
            oos_fraction=self.oos_fraction,
            warmup_bars=self.warmup_bars
        )
        return self.windows

    def print_plan(self):
        return self.splitter.print_walk_forward_summary(
            n_splits=self.n_splits,
            mode=self.mode,
            train_bars=self.train_bars,
            oos_bars=self.oos_bars,
            oos_fraction=self.oos_fraction,
            warmup_bars=self.warmup_bars
        )

    # ==========================================
    # EXECUTION RUNNER
    # ==========================================
    def run(
        self,
        window_data_dir,
        analytics_dir
    ):
        """
        Runs independent backtests on all generated OOS windows.
        """
        os.makedirs(window_data_dir, exist_ok=True)
        os.makedirs(analytics_dir, exist_ok=True)

        if self.windows is None:
            self.generate_windows()

        print("\n" + "=" * 65)
        print(" EXECUTING WALK-FORWARD VALIDATION")
        print("=" * 65)
        print(f"  Windows          : {self.n_splits}")
        print(f"  Training Mode    : {self.mode.upper()}")
        print(f"  Warm-up Bars     : {self.warmup_bars} (excluded from trading)")
        print(f"  Output Directory : {analytics_dir}")
        print("=" * 65)

        window_summaries = []
        all_oos_trades = []

        for w in self.windows:
            wnum = w["window"]

            # Save split data files with is_warmup column
            oos_csv_path = os.path.join(
                window_data_dir,
                f"window_{wnum:02d}_oos.csv"
            )
            train_csv_path = os.path.join(
                window_data_dir,
                f"window_{wnum:02d}_train.csv"
            )

            w["oos_df"].to_csv(oos_csv_path, index=False)
            w["train_df"].to_csv(train_csv_path, index=False)

            summary_csv_path = os.path.join(
                analytics_dir,
                f"window_{wnum:02d}_oos_summary.csv"
            )
            trades_csv_path = os.path.join(
                analytics_dir,
                f"window_{wnum:02d}_oos_trades.csv"
            )

            print(f"\n  [Window {wnum}/{self.n_splits}]")
            print(
                f"  Train Period : {str(w['train_start'])[:19]} → "
                f"{str(w['train_end'])[:19]} ({w['train_rows']:,} bars)"
            )
            print(
                f"  OOS Period   : {str(w['oos_start'])[:19]} → "
                f"{str(w['oos_end'])[:19]} ({w['oos_rows']:,} bars)"
            )
            print(f"  Warmup Bars  : {w['warmup_bars']} (prepended for indicators)")

            # Execute backtest as isolated subprocess (fresh portfolio, 0 state leakage)
            cmd = [
                sys.executable,
                "-m",
                "backtesting.backtest",
                "--data", oos_csv_path,
                "--summary", summary_csv_path,
                "--trades", trades_csv_path,
                "--warmup", str(w["warmup_bars"])
            ]

            subprocess.run(cmd, check=True)

            # Collect results
            sum_df = pd.read_csv(summary_csv_path)
            row = sum_df.iloc[0].to_dict()

            trade_count = int(row.get("total_trades", 0))
            sample_flag = classify_sample_adequacy(trade_count)

            row["window"] = wnum
            row["mode"] = self.mode
            row["train_start"] = str(w["train_start"])[:19]
            row["train_end"] = str(w["train_end"])[:19]
            row["oos_start"] = str(w["oos_start"])[:19]
            row["oos_end"] = str(w["oos_end"])[:19]
            row["train_rows"] = w["train_rows"]
            row["oos_rows"] = w["oos_rows"]
            row["warmup_bars"] = w["warmup_bars"]
            row["sample_flag"] = sample_flag

            window_summaries.append(row)

            # Collect trades if any
            if os.path.exists(trades_csv_path) and os.path.getsize(trades_csv_path) > 10:
                t_df = pd.read_csv(trades_csv_path)
                if len(t_df) > 0:
                    t_df["window"] = wnum
                    all_oos_trades.append(t_df)

            print(
                f"  Window {wnum} Result: {trade_count} trades | "
                f"Win Rate: {row.get('win_rate', 0):.1f}% | "
                f"PnL: {row.get('realized_pnl', 0):.2f} | "
                f"Flag: {sample_flag}"
            )

        # Consolidate window summaries
        consolidated_df = pd.DataFrame(window_summaries)
        consolidated_path = os.path.join(
            analytics_dir,
            "walkforward_consolidated.csv"
        )
        consolidated_df.to_csv(consolidated_path, index=False)

        # Combine all OOS trades across windows
        if len(all_oos_trades) > 0:
            combined_trades_df = pd.concat(all_oos_trades, ignore_index=True)
        else:
            combined_trades_df = pd.DataFrame(columns=[
                "datetime", "position", "size", "entry_price", "exit_price",
                "stop_loss", "target_price", "pnl", "vroc", "adx", "atr",
                "regime", "exit_reason", "window"
            ])

        combined_trades_path = os.path.join(
            analytics_dir,
            "combined_oos_trades.csv"
        )
        combined_trades_df.to_csv(combined_trades_path, index=False)

        print("\n" + "=" * 65)
        print(" WALK-FORWARD RUN COMPLETE")
        print("=" * 65)
        print(f"  Windows Run         : {len(window_summaries)}")
        print(f"  Total Pooled Trades : {len(combined_trades_df)}")
        print(f"  Consolidated CSV    : {consolidated_path}")
        print(f"  Combined Trades CSV : {combined_trades_path}")

        return consolidated_df, combined_trades_df

    # ==========================================
    # LEGACY COMPATIBILITY METHODS
    # ==========================================
    def get_train_data(self):
        return self.splitter.static_split()["train"]

    def get_validation_data(self):
        return self.splitter.static_split()["validation"]

    def get_test_data(self):
        return self.splitter.static_split()["test"]

    def print_summary(self):
        self.splitter.summary()


if __name__ == "__main__":
    DATA_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "historical_data.csv"
    )
    df = pd.read_csv(DATA_FILE)
    engine = WalkForwardEngine(df, n_splits=5, mode="expanding")
    engine.print_plan()