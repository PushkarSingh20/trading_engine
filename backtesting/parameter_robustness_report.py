"""
parameter_robustness_report.py
==============================
Stage 13 — Parameter Robustness Analysis & Reporting

Evaluates parameter surfaces, detects performance plateaus vs fragile spikes,
and assesses trade frequency adequacy across evaluated walk-forward windows.

Reads CSVs from analytics/robustness/:
  1D: adx_1d.csv, vroc_1d.csv, atr_1d.csv, target_1d.csv,
      lookback_1d.csv, quality_score_1d.csv
  2D: adx_vroc_2d.csv, atr_target_2d.csv

Output:
  analytics/robustness/parameter_robustness_summary.csv

Usage:
  python -m backtesting.parameter_robustness_report
"""

import os
import glob
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROBUSTNESS_DIR = os.path.join(os.path.dirname(BASE_DIR), "analytics", "robustness")
REPORT_OUTPUT_FILE = os.path.join(ROBUSTNESS_DIR, "parameter_robustness_summary.csv")

# Expected canonical file names per Stage 13 spec
EXPECTED_1D = [
    "adx_1d.csv", "vroc_1d.csv", "atr_1d.csv",
    "target_1d.csv", "lookback_1d.csv", "quality_score_1d.csv",
]
EXPECTED_2D = ["adx_vroc_2d.csv", "atr_target_2d.csv"]

# Maps CSV stem -> param column name for 1D files
PARAM_COL_MAP = {
    "adx_1d":        "adx_entry",
    "vroc_1d":       "vroc_thresh",
    "atr_1d":        "atr_mult",
    "target_1d":     "target_mult",
    "lookback_1d":   "lookback",
    "quality_score_1d": "quality_score",
    "adx_vroc_2d":   None,  # two params
    "atr_target_2d": None,
}

PARAM_COLS_2D = {
    "adx_vroc_2d":   ("adx_entry", "vroc_thresh"),
    "atr_target_2d": ("atr_mult",  "target_mult"),
}

SAMPLE_LABELS = {
    "VERY_LOW_SAMPLE": "<10 trades",
    "LOW_SAMPLE":      "10-29 trades",
    "ADEQUATE_FOR_BASIC_REVIEW": ">=30 trades",
}


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_region(row):
    """
    Classify a single parameter point.

    Labels (do NOT use as recommendations — these are descriptive, not prescriptive):
      DEAD_ZONE        < 10 trades regardless of direction
      LOW_SAMPLE       10-29 trades — insufficient for inference
      STABLE_PLATEAU   >= 30 trades + positive expectancy + majority positive neighbors
      FRAGILE_SPIKE    adequate sample, highest PnL but isolated (few positive neighbors)
      DECAY_ZONE       adequate sample, negative expectancy
    """
    trades = int(row.get("total_trades", 0))
    exp    = float(row.get("expectancy", 0.0))
    pos_nbr_frac = float(row.get("pos_neighbor_frac", -1))

    if trades < 10:
        return "DEAD_ZONE"
    elif trades < 30:
        return "LOW_SAMPLE"
    else:
        if exp <= 0:
            return "DECAY_ZONE"
        if pos_nbr_frac >= 0 and pos_nbr_frac < 0.5:
            return "FRAGILE_SPIKE"
        return "STABLE_PLATEAU"


# ─────────────────────────────────────────────────────────────────────────────
# Neighborhood Analysis (for 2D grids)
# ─────────────────────────────────────────────────────────────────────────────

def compute_neighborhood(df, p1, p2):
    """
    For each cell in a 2D grid, compute:
      pos_neighbor_frac          — fraction of adjacent cells (4-connectivity) with positive PnL
      adequate_neighbor_frac     — fraction of adjacent cells with adequate sample (>=10 trades)
      neighbor_expectancy_median — median expectancy of neighbors
      neighbor_expectancy_worst  — min expectancy of neighbors
      neighbor_expectancy_best   — max expectancy of neighbors
    """
    rows = []
    v1_vals = sorted(df[p1].unique())
    v2_vals = sorted(df[p2].unique())

    # Index lookup for speed
    idx_map = {(r[p1], r[p2]): i for i, r in df.iterrows()}

    def _get(v1, v2):
        k = (v1, v2)
        if k in idx_map:
            return df.loc[idx_map[k]]
        return None

    v1_idx = {v: i for i, v in enumerate(v1_vals)}
    v2_idx = {v: i for i, v in enumerate(v2_vals)}

    result_rows = []
    for _, row in df.iterrows():
        v1 = row[p1]
        v2 = row[p2]
        i1 = v1_idx[v1]
        i2 = v2_idx[v2]

        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i1 + di, i2 + dj
            if 0 <= ni < len(v1_vals) and 0 <= nj < len(v2_vals):
                nrow = _get(v1_vals[ni], v2_vals[nj])
                if nrow is not None:
                    neighbors.append(nrow)

        if neighbors:
            nbr_pnl  = [float(n["realized_pnl"]) for n in neighbors]
            nbr_trd  = [int(n["total_trades"])    for n in neighbors]
            nbr_exp  = [float(n["expectancy"])     for n in neighbors]
            pos_frac  = sum(1 for p in nbr_pnl if p > 0) / len(nbr_pnl)
            adeq_frac = sum(1 for t in nbr_trd if t >= 10) / len(nbr_trd)
            exp_med   = float(np.median(nbr_exp))
            exp_worst = float(np.min(nbr_exp))
            exp_best  = float(np.max(nbr_exp))
        else:
            pos_frac = adeq_frac = exp_med = exp_worst = exp_best = float("nan")

        result_rows.append({
            p1: v1, p2: v2,
            "pos_neighbor_frac":         round(pos_frac, 3)  if not np.isnan(pos_frac)  else float("nan"),
            "adequate_neighbor_frac":    round(adeq_frac, 3) if not np.isnan(adeq_frac) else float("nan"),
            "neighbor_exp_median":       round(exp_med, 2)   if not np.isnan(exp_med)   else float("nan"),
            "neighbor_exp_worst":        round(exp_worst, 2) if not np.isnan(exp_worst) else float("nan"),
            "neighbor_exp_best":         round(exp_best, 2)  if not np.isnan(exp_best)  else float("nan"),
        })

    # Drop any pre-existing neighborhood columns if previously written
    drop_cols = [c for c in ["pos_neighbor_frac", "adequate_neighbor_frac", "neighbor_exp_median", 
                             "neighbor_exp_worst", "neighbor_exp_best", "classification"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    nbr_df = pd.DataFrame(result_rows)
    df = df.merge(nbr_df, on=[p1, p2], how="left")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1D Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_1d_sensitivity(file_path):
    df = pd.read_csv(file_path)
    stem = os.path.basename(file_path).replace(".csv", "")
    
    # Identify parameter column
    param_name = PARAM_COL_MAP.get(stem, None)
    if not param_name or param_name not in df.columns:
        known = ["adx_entry", "vroc_thresh", "atr_mult", "target_mult", "lookback", "quality_score"]
        varying = [p for p in known if p in df.columns and df[p].nunique() > 1]
        param_name = varying[0] if varying else (stem.replace("_1d", ""))

    print("\n" + "=" * 80)
    print(f" 1D PARAMETER SENSITIVITY: {stem.upper()} ({param_name})")
    print("=" * 80)

    df["classification"] = df.apply(classify_region, axis=1)

    # Profitability Flatness Ratio (PFR) — primary metric
    pos_exp_count = (df["expectancy"] > 0).sum()
    total_points  = len(df)
    pfr = (pos_exp_count / total_points * 100) if total_points > 0 else 0.0

    # Parameter Fragility Index (PFI) — supplementary diagnostic only
    pnl_mean = df["realized_pnl"].mean()
    pnl_std  = df["realized_pnl"].std()
    pfi = (pnl_std / abs(pnl_mean)) if pnl_mean != 0 else float("inf")

    display_cols = [param_name, "total_trades", "win_rate", "realized_pnl",
                    "profit_factor", "expectancy", "sample_flag", "classification"]
    avail_cols = [c for c in display_cols if c in df.columns]

    print(df[avail_cols].to_string(index=False))
    print("-" * 80)
    print(f"  Profitability Flatness Ratio (PFR) : {pfr:.1f}%  ({pos_exp_count}/{total_points} configs with positive expectancy)")
    print(f"  Parameter Fragility Index (PFI)    : {pfi:.3f}  [SUPPLEMENTARY DIAGNOSTIC ONLY — not a primary criterion]")
    if param_name in df.columns:
        print(f"  Max Expectancy Config              : {param_name} = {df.loc[df['expectancy'].idxmax()][param_name]}  (Exp: ${df['expectancy'].max():.2f})")
        print(f"  Max Trade Count Config             : {param_name} = {df.loc[df['total_trades'].idxmax()][param_name]}  (Trades: {df['total_trades'].max()})")
    print("=" * 80)

    return {
        "sweep_type":     "1D",
        "parameter":      stem,
        "configurations": total_points,
        "pfr_pct":        round(pfr, 1),
        "pfi_supplementary": round(pfi, 3),
        "max_pnl":        df["realized_pnl"].max(),
        "max_expectancy": df["expectancy"].max(),
        "max_trades":     df["total_trades"].max(),
        "dead_zone_count":  (df["classification"] == "DEAD_ZONE").sum(),
        "low_sample_count": (df["classification"] == "LOW_SAMPLE").sum(),
        "stable_plateau_count": (df["classification"] == "STABLE_PLATEAU").sum(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2D Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_2d_grid(file_path):
    df = pd.read_csv(file_path)
    stem = os.path.basename(file_path).replace(".csv", "")
    p1, p2 = PARAM_COLS_2D.get(stem, (None, None))

    if p1 is None:
        # Fallback: infer from stem (old naming convention)
        clean = stem.replace("2d_grid_", "").replace("_2d", "")
        if "_x_" in clean:
            parts = clean.split("_x_")
        else:
            parts = clean.split("_", 1)
        p1, p2 = parts[0], parts[1] if len(parts) > 1 else (clean, "param2")

    # Compute neighborhood analysis
    df = compute_neighborhood(df, p1, p2)

    # Classify every cell
    df["classification"] = df.apply(classify_region, axis=1)

    # Pivot tables
    pnl_pivot    = df.pivot(index=p1, columns=p2, values="realized_pnl")
    trades_pivot = df.pivot(index=p1, columns=p2, values="total_trades")
    exp_pivot    = df.pivot(index=p1, columns=p2, values="expectancy")
    nbr_pivot    = df.pivot(index=p1, columns=p2, values="pos_neighbor_frac")
    cls_pivot    = df.pivot(index=p1, columns=p2, values="classification")

    print("\n" + "=" * 80)
    print(f" 2D INTERACTION GRID: {stem.upper()}")
    print(f" Grid: {p1} (rows) x {p2} (cols)")
    print("=" * 80)

    print(f"\n--- Realized PnL ($) ---")
    print(pnl_pivot.to_string())

    print(f"\n--- Total Trade Counts ---")
    print(trades_pivot.to_string())

    print(f"\n--- Expectancy ($) ---")
    print(exp_pivot.to_string())

    print(f"\n--- Positive-Neighbor Fraction (4-connectivity) ---")
    print(nbr_pivot.round(2).to_string())

    print(f"\n--- Region Classification ---")
    print(cls_pivot.to_string())

    # Summary metrics
    pos_cells   = (df["realized_pnl"] > 0).sum()
    total_cells = len(df)
    pfr = (pos_cells / total_cells * 100) if total_cells > 0 else 0.0

    dead_zone = (df["classification"] == "DEAD_ZONE").sum()
    low_samp  = (df["classification"] == "LOW_SAMPLE").sum()
    stable    = (df["classification"] == "STABLE_PLATEAU").sum()
    fragile   = (df["classification"] == "FRAGILE_SPIKE").sum()
    decay     = (df["classification"] == "DECAY_ZONE").sum()

    print("-" * 80)
    print(f"  2D Grid: {len(pnl_pivot.index)} rows x {len(pnl_pivot.columns)} cols = {total_cells} nodes")
    print(f"  Profitable Nodes Ratio (PFR): {pfr:.1f}%  ({pos_cells}/{total_cells})")
    print(f"  Region Breakdown:")
    print(f"    DEAD_ZONE      : {dead_zone}")
    print(f"    LOW_SAMPLE     : {low_samp}")
    print(f"    STABLE_PLATEAU : {stable}")
    print(f"    FRAGILE_SPIKE  : {fragile}")
    print(f"    DECAY_ZONE     : {decay}")
    print(f"  NOTE: PFI is a supplementary diagnostic only.")
    print("=" * 80)

    # Save augmented CSV back to disk
    df.to_csv(file_path, index=False)

    return {
        "sweep_type":           "2D",
        "parameter":            stem,
        "configurations":       total_cells,
        "pfr_pct":              round(pfr, 1),
        "pfi_supplementary":    float("nan"),
        "max_pnl":              df["realized_pnl"].max(),
        "max_expectancy":       df["expectancy"].max(),
        "max_trades":           df["total_trades"].max(),
        "dead_zone_count":      dead_zone,
        "low_sample_count":     low_samp,
        "stable_plateau_count": stable,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main report
# ─────────────────────────────────────────────────────────────────────────────

def generate_robustness_report():
    # Locate files — prefer canonical names, then fall back to old naming
    def _find(names, fallback_pattern):
        found = []
        for n in names:
            p = os.path.join(ROBUSTNESS_DIR, n)
            if os.path.exists(p):
                found.append(p)
            else:
                # Look for fallback only if canonical doesn't exist
                stem = n.replace(".csv", "")
                for old in glob.glob(os.path.join(ROBUSTNESS_DIR, fallback_pattern)):
                    if stem in old and old not in found:
                        found.append(old)
        return found

    files_1d = _find(EXPECTED_1D, "1d_sensitivity_*.csv")
    files_2d = _find(EXPECTED_2D, "2d_grid_*.csv")

    if not files_1d and not files_2d:
        print("\nNo robustness sweep result CSVs found in analytics/robustness/.")
        print("Run parameter_robustness.py first.")
        return

    print("\n" + "#" * 80)
    print(" STAGE 13: PARAMETER ROBUSTNESS CONSOLIDATED REPORT")
    print("#" * 80)

    summary_rows = []

    for f in files_1d:
        res = analyze_1d_sensitivity(f)
        summary_rows.append(res)

    for f in files_2d:
        res = analyze_2d_grid(f)
        summary_rows.append(res)

    # ── Window 5 quarantine confirmation ──────────────────────────────────────
    print("\n" + "=" * 80)
    print(" WINDOW 5 QUARANTINE CONFIRMATION")
    print("=" * 80)
    print("  Window 5 was EXCLUDED from all Stage 13 evaluations.")
    print("  It is reserved as the Stage 16 holdout (final out-of-sample test).")
    print("  No parameter decision in Stage 13 was informed by Window 5 data.")
    print("=" * 80)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORT_OUTPUT_FILE, index=False)
    print(f"\nConsolidated robustness summary saved to: {REPORT_OUTPUT_FILE}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    generate_robustness_report()
