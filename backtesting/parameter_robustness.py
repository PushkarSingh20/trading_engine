"""
parameter_robustness.py
=======================
Stage 13 — Parameter Robustness Testing Framework

Performs 1D sensitivity sweeps and 2D interaction grid scans across
walk-forward evaluation windows (Windows 1–4; Window 5 quarantined).

Evaluates:
  - Parameter plateau flatness vs fragile spikes
  - Trade frequency recovery (sample adequacy)
  - Expectancy and profit factor stability

Output CSV naming convention (Stage 13 specification):
  1D: adx_1d.csv, vroc_1d.csv, atr_1d.csv, target_1d.csv,
      lookback_1d.csv, quality_score_1d.csv
  2D: adx_vroc_2d.csv, atr_target_2d.csv

Usage:
  python -m backtesting.parameter_robustness --mode 1d --param adx_entry
  python -m backtesting.parameter_robustness --mode 1d --param vroc_thresh
  python -m backtesting.parameter_robustness --mode 2d --param1 atr_mult --param2 target_mult
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np

from backtesting.dataset_splitter import DatasetSplitter
from backtesting.performance import PerformanceAnalyzer
from backtesting.walkforward import classify_sample_adequacy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "historical_data.csv")
ROBUSTNESS_DIR = os.path.join(os.path.dirname(BASE_DIR), "analytics", "robustness")
TEMP_DATA_DIR = os.path.join(BASE_DIR, "walkforward_data", "windows")

os.makedirs(ROBUSTNESS_DIR, exist_ok=True)
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# Baseline defaults strictly matching Stage 12 frozen configuration
BASELINE_PARAMS = {
    "adx_entry": 50.0,
    "vroc_thresh": 100.0,
    "atr_mult": 2.0,
    "target_mult": 2.0,
    "lookback": 5,
    "quality_score": 3,
}

# Parameter grid ranges for Stage 13 exploration (as specified in Stage 13 prompt)
PARAM_GRIDS = {
    "adx_entry":    [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
    "vroc_thresh":  [40.0, 60.0, 80.0, 100.0, 120.0, 150.0],
    "atr_mult":     [1.5, 2.0, 2.5, 3.0],
    "target_mult":  [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    "lookback":     [3, 5, 7, 10, 12, 15],
    "quality_score": [2, 3, 4],
}

# Maps internal param_name -> canonical output CSV stem (Stage 13 naming spec)
OUTPUT_STEM_MAP_1D = {
    "adx_entry":     "adx_1d",
    "vroc_thresh":   "vroc_1d",
    "atr_mult":      "atr_1d",
    "target_mult":   "target_1d",
    "lookback":      "lookback_1d",
    "quality_score": "quality_score_1d",
}

OUTPUT_STEM_MAP_2D = {
    ("adx_entry", "vroc_thresh"): "adx_vroc_2d",
    ("atr_mult",  "target_mult"): "atr_target_2d",
}


def _resolve_1d_output(param_name):
    stem = OUTPUT_STEM_MAP_1D.get(param_name, f"{param_name}_1d")
    return os.path.join(ROBUSTNESS_DIR, f"{stem}.csv")


def _resolve_2d_output(param1_name, param2_name):
    stem = OUTPUT_STEM_MAP_2D.get(
        (param1_name, param2_name),
        f"{param1_name}_{param2_name}_2d"
    )
    return os.path.join(ROBUSTNESS_DIR, f"{stem}.csv")


def prepare_evaluation_windows(data_file=DATA_FILE, max_windows=4, warmup_bars=250):
    """
    Generates walk-forward windows and returns only the training/evaluation
    windows up to max_windows (Windows 1-4). Window 5 is quarantined.
    """
    df = pd.read_csv(data_file)
    splitter = DatasetSplitter(df, "Datetime")
    all_windows = splitter.walk_forward_windows(
        n_splits=5,
        mode="expanding",
        oos_fraction=0.10,
        warmup_bars=warmup_bars
    )

    # Strictly select Windows 1 to max_windows (Window 5 quarantined)
    eval_windows = all_windows[:max_windows]

    # Ensure window CSVs exist
    for w in eval_windows:
        wnum = w["window"]
        oos_path = os.path.join(TEMP_DATA_DIR, f"window_{wnum:02d}_oos.csv")
        w["oos_df"].to_csv(oos_path, index=False)
        w["oos_path"] = oos_path

    return eval_windows


def evaluate_parameter_set(params, eval_windows, temp_output_dir=ROBUSTNESS_DIR):
    """
    Runs isolated backtest subprocesses for a given parameter set across
    the evaluation windows, then computes pooled metrics.
    """
    window_trades = []
    window_summaries = []

    for w in eval_windows:
        wnum = w["window"]
        sum_path = os.path.join(temp_output_dir, f"_temp_w{wnum}_summary.csv")
        trd_path = os.path.join(temp_output_dir, f"_temp_w{wnum}_trades.csv")

        cmd = [
            sys.executable,
            "-m", "backtesting.backtest",
            "--data", w["oos_path"],
            "--summary", sum_path,
            "--trades", trd_path,
            "--warmup", str(w["warmup_bars"]),
            "--adx-entry",    str(params.get("adx_entry",    BASELINE_PARAMS["adx_entry"])),
            "--vroc-thresh",  str(params.get("vroc_thresh",  BASELINE_PARAMS["vroc_thresh"])),
            "--atr-mult",     str(params.get("atr_mult",     BASELINE_PARAMS["atr_mult"])),
            "--target-mult",  str(params.get("target_mult",  BASELINE_PARAMS["target_mult"])),
            "--lookback",     str(params.get("lookback",     BASELINE_PARAMS["lookback"])),
            "--quality-score",str(params.get("quality_score",BASELINE_PARAMS["quality_score"])),
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(sum_path):
            s_df = pd.read_csv(sum_path)
            if len(s_df) > 0:
                window_summaries.append(s_df.iloc[0].to_dict())

        if os.path.exists(trd_path) and os.path.getsize(trd_path) > 10:
            t_df = pd.read_csv(trd_path)
            if len(t_df) > 0:
                t_df["window"] = wnum
                window_trades.append(t_df)

    # Aggregate pooled trade metrics
    if len(window_trades) > 0:
        pooled_trades = pd.concat(window_trades, ignore_index=True)
        analyzer = PerformanceAnalyzer(pooled_trades)

        total_trades = len(pooled_trades)
        wins   = len(pooled_trades[pooled_trades["pnl"] > 0])
        losses = len(pooled_trades[pooled_trades["pnl"] <= 0])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        pnl    = pooled_trades["pnl"].sum()
        pf     = analyzer.profit_factor()
        exp    = analyzer.expectancy()
        mdd    = analyzer.max_drawdown()
        sharpe = analyzer.sharpe_ratio()
    else:
        total_trades = 0
        wins = 0
        losses = 0
        win_rate = 0.0
        pnl = 0.0
        pf = 0.0
        exp = 0.0
        mdd = 0.0
        sharpe = 0.0

    pos_windows = sum(1 for s in window_summaries if s.get("realized_pnl", 0) > 0)
    total_eval_windows = len(eval_windows)
    pos_window_pct = (pos_windows / total_eval_windows * 100) if total_eval_windows > 0 else 0.0

    sample_flag = classify_sample_adequacy(total_trades)

    result = {
        **params,
        "total_trades":    total_trades,
        "wins":            wins,
        "losses":          losses,
        "win_rate":        round(win_rate, 2),
        "realized_pnl":   round(pnl, 2),
        "profit_factor":   pf,
        "expectancy":      round(exp, 2),
        "max_drawdown":    round(mdd, 2),
        "sharpe_ratio":    round(sharpe, 4),
        "sample_flag":     sample_flag,
        "positive_windows": f"{pos_windows}/{total_eval_windows}",
        "win_window_pct":   round(pos_window_pct, 1),
    }

    return result


def run_1d_sensitivity(param_name, values=None, eval_windows=None):
    if values is None:
        values = PARAM_GRIDS.get(param_name, [BASELINE_PARAMS[param_name]])
    if eval_windows is None:
        eval_windows = prepare_evaluation_windows()

    out_file = _resolve_1d_output(param_name)

    print("\n" + "=" * 75)
    print(f" 1D PARAMETER SENSITIVITY SCAN: {param_name.upper()}")
    print(f" Evaluation Scope: Windows 1-{len(eval_windows)} (Window 5 Quarantined)")
    print(f" Values: {values}")
    print("=" * 75)

    results = []
    for val in values:
        current_params = BASELINE_PARAMS.copy()
        current_params[param_name] = val
        print(f"  Scanning {param_name} = {val}...", end="", flush=True)
        res = evaluate_parameter_set(current_params, eval_windows)
        results.append(res)
        print(
            f" -> Trades: {res['total_trades']:<3} | "
            f"Win%: {res['win_rate']:<5.1f} | "
            f"PnL: ${res['realized_pnl']:<8.2f} | "
            f"PF: {res['profit_factor']} | "
            f"Flag: {res['sample_flag']}"
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv(out_file, index=False)
    print(f"\nSaved 1D results to: {out_file}")

    return df_results


def run_2d_grid(param1_name, param2_name, values1=None, values2=None, eval_windows=None):
    if values1 is None:
        values1 = PARAM_GRIDS.get(param1_name, [BASELINE_PARAMS[param1_name]])
    if values2 is None:
        values2 = PARAM_GRIDS.get(param2_name, [BASELINE_PARAMS[param2_name]])
    if eval_windows is None:
        eval_windows = prepare_evaluation_windows()

    out_file = _resolve_2d_output(param1_name, param2_name)

    print("\n" + "=" * 75)
    print(f" 2D PARAMETER INTERACTION GRID: {param1_name.upper()} x {param2_name.upper()}")
    print(f" Grid Size: {len(values1)} x {len(values2)} = {len(values1)*len(values2)} combinations")
    print(f" Evaluation Scope: Windows 1-{len(eval_windows)} (Window 5 Quarantined)")
    print("=" * 75)

    results = []
    for v1 in values1:
        for v2 in values2:
            current_params = BASELINE_PARAMS.copy()
            current_params[param1_name] = v1
            current_params[param2_name] = v2
            print(f"  Testing {param1_name}={v1}, {param2_name}={v2}...", end="", flush=True)
            res = evaluate_parameter_set(current_params, eval_windows)
            results.append(res)
            print(
                f" -> Trades: {res['total_trades']:<3} | "
                f"Win%: {res['win_rate']:<5.1f} | "
                f"PnL: ${res['realized_pnl']:<8.2f} | "
                f"PF: {res['profit_factor']}"
            )

    df_results = pd.DataFrame(results)
    df_results.to_csv(out_file, index=False)
    print(f"\nSaved 2D grid results to: {out_file}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 13 Parameter Robustness Testing")
    parser.add_argument("--mode", choices=["1d", "2d", "all"], default="1d",
                        help="Scan mode: 1d, 2d, or all")
    parser.add_argument("--param",  default="adx_entry",
                        help="Parameter name for 1D scan")
    parser.add_argument("--param1", default="adx_entry",
                        help="First parameter for 2D grid")
    parser.add_argument("--param2", default="vroc_thresh",
                        help="Second parameter for 2D grid")

    args = parser.parse_args()
    eval_windows = prepare_evaluation_windows(max_windows=4)

    if args.mode == "1d":
        run_1d_sensitivity(args.param, eval_windows=eval_windows)
    elif args.mode == "2d":
        run_2d_grid(args.param1, args.param2, eval_windows=eval_windows)
    elif args.mode == "all":
        for p in ["adx_entry", "vroc_thresh", "atr_mult", "target_mult", "lookback", "quality_score"]:
            run_1d_sensitivity(p, eval_windows=eval_windows)
        run_2d_grid("adx_entry", "vroc_thresh", eval_windows=eval_windows)
        run_2d_grid("atr_mult",  "target_mult",  eval_windows=eval_windows)
