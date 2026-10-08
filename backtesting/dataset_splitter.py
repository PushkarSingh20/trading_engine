"""
dataset_splitter.py
===================
Stage 12 — Walk-Forward Robustness Validation

Provides:
  1. walk_forward_windows():
     Generates sequential walk-forward splits with configurable parameters:
       - n_splits: Number of chronological out-of-sample (OOS) windows
       - mode: "expanding" (default) or "rolling" training window
       - train_bars: Fixed training window length (for rolling mode)
       - oos_bars: Fixed OOS window length (overrides oos_fraction if provided)
       - oos_fraction: Fraction of full dataset per OOS window (default: 0.10)
       - warmup_bars: Historical context bars prepended to each window for
         indicator initialization (EMA200, HMA, ATR, ADX, VROC).
         Warm-up bars are tagged with `is_warmup=True` and NEVER traded.

  2. static_split():
     Legacy 60/20/20 train/val/test split kept strictly for backward compatibility.
"""

import pandas as pd


class DatasetSplitter:

    def __init__(
        self,
        df,
        datetime_column="Datetime"
    ):
        self.datetime_column = datetime_column
        self.df = df.copy()

        # Clean column names
        self.df.columns = [
            str(col).strip().split(".")[0]
            for col in self.df.columns
        ]

        # Convert Datetime, coerce invalid headers (e.g. multi-index yfinance ticker rows)
        self.df[self.datetime_column] = pd.to_datetime(
            self.df[self.datetime_column],
            errors="coerce"
        )
        self.df = self.df.dropna(subset=[self.datetime_column])

        # Clean OHLCV columns if present
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df = self.df.dropna()

        # Sort strictly chronologically
        self.df = self.df.sort_values(
            self.datetime_column
        ).reset_index(drop=True)

    # =============================================
    # WALK-FORWARD WINDOW GENERATOR (STAGE 12)
    # =============================================
    def walk_forward_windows(
        self,
        n_splits=5,
        mode="expanding",
        train_bars=None,
        oos_bars=None,
        oos_fraction=0.10,
        warmup_bars=250
    ):
        """
        Generates chronological walk-forward windows.

        Parameters:
          n_splits     : Number of sequential OOS windows (default: 5)
          mode         : "expanding" (default) or "rolling" training window
          train_bars   : Window length for training in bars (used in rolling mode;
                         defaults to initial available training rows)
          oos_bars     : Explicit number of bars per OOS window (optional)
          oos_fraction : Fraction of total data per OOS window if oos_bars is None
          warmup_bars  : Number of bars prepended for indicator warm-up (default: 250)

        Returns:
          List of window dictionaries containing metadata and prepared DataFrames.
        """
        if mode not in ["expanding", "rolling"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'expanding' or 'rolling'.")

        total_rows = len(self.df)

        # Determine OOS size
        if oos_bars is not None:
            actual_oos_size = int(oos_bars)
        else:
            actual_oos_size = int(total_rows * oos_fraction)

        if actual_oos_size < 1:
            raise ValueError(
                f"OOS window size must be >= 1 bar (got {actual_oos_size} from total {total_rows})."
            )

        total_oos_rows = actual_oos_size * n_splits
        initial_train_rows = total_rows - total_oos_rows

        if initial_train_rows <= warmup_bars:
            raise ValueError(
                f"Insufficient data for {n_splits} windows.\n"
                f"  Total rows      : {total_rows}\n"
                f"  OOS size/window : {actual_oos_size}\n"
                f"  Total OOS rows  : {total_oos_rows}\n"
                f"  Initial train   : {initial_train_rows}\n"
                f"  Warmup required : {warmup_bars}\n"
                f"Reduce n_splits or oos_fraction."
            )

        actual_train_bars = train_bars if train_bars is not None else initial_train_rows

        windows = []

        for i in range(n_splits):
            oos_start_idx = initial_train_rows + (i * actual_oos_size)
            oos_end_idx = oos_start_idx + actual_oos_size

            # Determine training indices
            train_end_idx = oos_start_idx  # Exclusive

            if mode == "expanding":
                train_start_idx = 0
            else:  # rolling
                train_start_idx = max(0, train_end_idx - actual_train_bars)

            # ----------------------------------------------------
            # OOS DataFrame preparation with Warm-Up Prefix
            # ----------------------------------------------------
            oos_warmup_start = max(0, oos_start_idx - warmup_bars)
            oos_df_raw = self.df.iloc[oos_warmup_start:oos_end_idx].copy()

            # Mark warm-up bars clearly
            # Indices strictly prior to oos_start_idx are warmup bars
            oos_df_raw["is_warmup"] = oos_df_raw.index < oos_start_idx
            oos_df_raw = oos_df_raw.reset_index(drop=True)

            # ----------------------------------------------------
            # Train DataFrame preparation
            # ----------------------------------------------------
            if mode == "rolling" and train_start_idx > 0:
                train_warmup_start = max(0, train_start_idx - warmup_bars)
                train_df_raw = self.df.iloc[train_warmup_start:train_end_idx].copy()
                train_df_raw["is_warmup"] = train_df_raw.index < train_start_idx
            else:
                train_df_raw = self.df.iloc[train_start_idx:train_end_idx].copy()
                train_df_raw["is_warmup"] = False

            train_df_raw = train_df_raw.reset_index(drop=True)

            # Tradeable live data slices for date metadata
            train_live_df = self.df.iloc[train_start_idx:train_end_idx]
            oos_live_df = self.df.iloc[oos_start_idx:oos_end_idx]

            window = {
                "window": i + 1,
                "mode": mode,
                "train_start": train_live_df[self.datetime_column].iloc[0],
                "train_end": train_live_df[self.datetime_column].iloc[-1],
                "oos_start": oos_live_df[self.datetime_column].iloc[0],
                "oos_end": oos_live_df[self.datetime_column].iloc[-1],
                "train_rows": len(train_live_df),
                "oos_rows": len(oos_live_df),
                "warmup_bars": warmup_bars,
                "train_df": train_df_raw,
                "oos_df": oos_df_raw,
            }

            windows.append(window)

        return windows

    # =============================================
    # PRINT WALK-FORWARD SUMMARY
    # =============================================
    def print_walk_forward_summary(
        self,
        n_splits=5,
        mode="expanding",
        train_bars=None,
        oos_bars=None,
        oos_fraction=0.10,
        warmup_bars=250
    ):
        windows = self.walk_forward_windows(
            n_splits=n_splits,
            mode=mode,
            train_bars=train_bars,
            oos_bars=oos_bars,
            oos_fraction=oos_fraction,
            warmup_bars=warmup_bars
        )

        total_rows = len(self.df)
        oos_size = windows[0]["oos_rows"]

        print("\n" + "=" * 65)
        print(f" WALK-FORWARD WINDOW PLAN ({mode.upper()} TRAINING)")
        print("=" * 65)
        print(f" Total rows    : {total_rows:,}")
        print(f" OOS per window: {oos_size:,} bars")
        print(f" Warm-up bars  : {warmup_bars}")
        print(f" Windows       : {n_splits}")
        print("=" * 65)

        for w in windows:
            print(f"\n  Window {w['window']}/{n_splits} [{mode}]")
            print(
                f"    TRAIN : {str(w['train_start'])[:19]} → "
                f"{str(w['train_end'])[:19]} ({w['train_rows']:,} bars)"
            )
            print(
                f"    OOS   : {str(w['oos_start'])[:19]} → "
                f"{str(w['oos_end'])[:19]} ({w['oos_rows']:,} bars)"
            )

        print("\n" + "=" * 65)
        return windows

    # =============================================
    # STATIC SPLIT (LEGACY 60/20/20)
    # =============================================
    def static_split(self):
        total_rows = len(self.df)
        train_end = int(total_rows * 0.60)
        validation_end = int(total_rows * 0.80)

        train_df = self.df.iloc[:train_end].copy()
        validation_df = self.df.iloc[train_end:validation_end].copy()
        test_df = self.df.iloc[validation_end:].copy()

        return {
            "train": train_df,
            "validation": validation_df,
            "test": test_df
        }

    def split(self):
        """Legacy alias to static_split()."""
        return self.static_split()

    def summary(self):
        """Legacy summary for static split."""
        data = self.static_split()
        print("\n==============================")
        print("DATASET SPLIT (STATIC 60/20/20)")
        print("==============================")
        for name, dataset in data.items():
            print(f"\n{name.upper()}")
            print("Rows :", len(dataset))
            print("Start:", dataset[self.datetime_column].min())
            print("End  :", dataset[self.datetime_column].max())
        print("==============================")
        return data