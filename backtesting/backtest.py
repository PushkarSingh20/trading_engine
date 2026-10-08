import pandas as pd
import os
import argparse

from strategy import get_hma_signal

from backtesting.indicators import (
    calculate_atr,
    calculate_vroc_series,
    calculate_adx,
    calculate_hma,
    calculate_hma_slope
)


from backtesting.risk_engine import (
    ATRRiskEngine
)

from backtesting.trailing_stop import (
    TrailingStop
)

from backtesting.performance import (
    PerformanceAnalyzer
)

from backtesting.regime_detector import (
    detect_market_regime
)

from backtesting.cost_model import (
    get_cost_model,
    TransactionCostModel
)

from portfolio import Portfolio

from config import ADX_ENTRY_THRESHOLD


# =====================================
# CONFIG & ARGUMENT PARSING
# =====================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

parser = argparse.ArgumentParser(
    description="VROC Algorithm Historical Backtesting Engine"
)

parser.add_argument(
    "--data",
    default=os.path.join(
        BASE_DIR,
        "historical_data.csv"
    ),
    help="Historical data CSV"
)

parser.add_argument(
    "--summary",
    default=os.path.join(
        BASE_DIR,
        "backtest_summary.csv"
    ),
    help="Summary output CSV"
)

parser.add_argument(
    "--trades",
    default=os.path.join(
        BASE_DIR,
        "backtest_trades.csv"
    ),
    help="Trade log output CSV"
)

parser.add_argument(
    "--warmup",
    type=int,
    default=0,
    help=(
        "Number of warm-up bars at the start of the data file "
        "used ONLY for indicator initialization. No trades are "
        "generated from these bars."
    )
)

# -------------------------------------------------------------
# STRATEGY PARAMETERS (STAGE 13 PARAMETER ROBUSTNESS)
# All defaults strictly preserve Stage 12 frozen baseline values.
# -------------------------------------------------------------
parser.add_argument(
    "--adx-entry",
    type=float,
    default=float(ADX_ENTRY_THRESHOLD),
    help=f"Minimum ADX required for trade entry (default: {ADX_ENTRY_THRESHOLD})"
)

parser.add_argument(
    "--vroc-thresh",
    type=float,
    default=100.0,
    help="VROC threshold for quality score calculation (default: 100.0)"
)

parser.add_argument(
    "--atr-mult",
    type=float,
    default=2.0,
    help="ATR multiplier for stop loss distance (default: 2.0)"
)

parser.add_argument(
    "--target-mult",
    type=float,
    default=2.0,
    help="Target reward multiplier relative to stop distance (default: 2.0)"
)

parser.add_argument(
    "--lookback",
    type=int,
    default=5,
    help="Lookback period for breakout range highs/lows (default: 5)"
)

parser.add_argument(
    "--quality-score",
    type=int,
    default=3,
    help="Minimum quality score (out of 4) required for entry (default: 3)"
)

parser.add_argument(
    "--cost-scenario",
    type=str,
    default="ZERO_COST",
    help="Transaction cost scenario (default: ZERO_COST)"
)

parser.add_argument(
    "--brokerage-pct",
    type=float,
    default=None,
    help="Override brokerage fee percentage"
)

parser.add_argument(
    "--slippage-pct",
    type=float,
    default=None,
    help="Override adverse slippage percentage"
)

args = parser.parse_args()

DATA_FILE = args.data
OUTPUT_TRADES_FILE = args.trades
WARMUP_BARS = args.warmup

LOOKBACK = args.lookback
ADX_ENTRY = args.adx_entry
VROC_THRESH = args.vroc_thresh
ATR_MULT = args.atr_mult
TARGET_MULT = args.target_mult
QUALITY_THRESH = args.quality_score

COST_SCENARIO = args.cost_scenario
cost_model = get_cost_model(COST_SCENARIO)
if args.brokerage_pct is not None:
    cost_model.brokerage_pct = args.brokerage_pct
if args.slippage_pct is not None:
    cost_model.slippage_pct = args.slippage_pct



# =====================================
# LOAD DATA
# =====================================

print("\nLoading historical data...\n")

df = pd.read_csv(DATA_FILE)


# =====================================
# CLEAN COLUMN NAMES
# =====================================

df.columns = [

    str(col).strip().split(".")[0]

    for col in df.columns
]

print("Columns Found:")
print(df.columns)


# =====================================
# FIND DATETIME COLUMN
# =====================================

datetime_col = None

for col in [

    "Datetime",
    "Date",
    "datetime",
    "date",
    "Unnamed: 0"

]:

    if col in df.columns:

        datetime_col = col

        break


if datetime_col is None:

    raise Exception(

        f"No datetime column found.\nColumns: {df.columns}"
    )


# =====================================
# CONVERT DATETIME
# =====================================

df[datetime_col] = pd.to_datetime(

    df[datetime_col],

    errors="coerce"
)

df = df.dropna(subset=[datetime_col])


# =====================================
# CONVERT OHLCV TO NUMERIC
# =====================================

required_columns = [

    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for col in required_columns:

    if col not in df.columns:

        raise Exception(f"Missing column: {col}")

    df[col] = pd.to_numeric(

        df[col],

        errors="coerce"
    )


# =====================================
# REMOVE BAD ROWS
# =====================================

df = df.dropna()

df = df.sort_values(datetime_col)

# =====================================
# INDICATORS
# =====================================

df["ATR"] = calculate_atr(df)

df["VROC"] = calculate_vroc_series(
    df["Volume"]
)

df["ADX"] = calculate_adx(df)

df["HMA_FAST"] = calculate_hma(
    df["Close"],
    20
)

df["HMA_SLOW"] = calculate_hma(
    df["Close"],
    50
)

df["HMA_SLOPE"] = calculate_hma_slope(
    df["HMA_FAST"]
)

df["EMA200"] = (
    df["Close"]
    .ewm(span=200, adjust=False)
    .mean()
)

df = df.dropna()

df = df.reset_index(drop=True)

df = df[df["Volume"] > 0]
print("\n==========================")
print("DATE RANGE")
print("==========================")

print(
    "Start:",
    df[datetime_col].min()
)

print(
    "End:",
    df[datetime_col].max()
)

print(
    "Warm-up Bars:",
    WARMUP_BARS,
    "(no trades generated from first",
    WARMUP_BARS,
    "rows)"
    if WARMUP_BARS > 0
    else ""
)



# =====================================
# DEBUG OUTPUT
# =====================================

print("\n==========================")
print("DATA PREVIEW")
print("==========================")

print(df.head())

is_warmup_present = "is_warmup" in df.columns
tradeable_count = len(df[~df["is_warmup"].astype(bool)]) if is_warmup_present else max(0, len(df) - WARMUP_BARS)

print("\nTotal Rows:", len(df))
print("Tradeable Rows:", tradeable_count)


# =====================================
# PORTFOLIO
# =====================================

portfolio = Portfolio(100000)

risk_engine = ATRRiskEngine(
    atr_multiplier=ATR_MULT,
    target_multiplier=TARGET_MULT
)

trailing_engine = TrailingStop()


# =====================================
# BACKTEST VARIABLES
# =====================================
position = None

entry_price = 0

stop_loss = 0

target_price = 0

trade_log = []


# =====================================
# START BACKTEST
# =====================================

print("\n==========================")
print("STARTING BACKTEST...")
print("==========================\n")


for i in range(LOOKBACK, len(df)):

    # =================================
    # CURRENT ROW
    # =================================

    row = df.iloc[i]

    current_price = float(row["Close"])

    current_time = row[datetime_col]

    # =================================
    # WARM-UP GUARD
    # =================================
    # Warm-up bars are used exclusively for indicator initialization
    # (EMA200, HMA, ATR, ADX, VROC). They must NEVER generate trades
    # or contaminate out-of-sample performance evaluation.

    if is_warmup_present and bool(row["is_warmup"]):
        continue
    elif WARMUP_BARS > 0 and i < WARMUP_BARS:
        continue


    # =================================
    # PREVIOUS LOOKBACK DATA
    # =================================


    prev_data = df.iloc[i - LOOKBACK:i]

    volumes = prev_data["Volume"].tolist()

    highs = prev_data["High"].tolist()

    lows = prev_data["Low"].tolist()


    # =================================
    # CALCULATE VROC
    # =================================

    vroc = row["VROC"]

    atr = row["ATR"]

    adx = row["ADX"]

    hma_slope = row["HMA_SLOPE"]

    hma_fast = row["HMA_FAST"]

    hma_slow = row["HMA_SLOW"]

    avg_atr = (
        df["ATR"]
        .iloc[max(0, i - 50):i]
        .mean()
    )
    # =================================
    # MARKET REGIME
    # =================================

    regime = detect_market_regime(

        adx,

        atr,

        avg_atr,

        hma_slope

    )

    print(
        f"{current_time} | Regime: {regime}"
    )

    # =================================
    # RANGE
    # =================================

    range_high = max(highs)

    range_low = min(lows)


    # =================================
    # ENTRY LOGIC
    # =================================

    if position is None:

        # Only trade in trending markets
        if regime != "TREND":
            continue

        # ADX confirmation (entry-level threshold from CLI / config)
        if adx < ADX_ENTRY:
            continue

        signal_data = get_hma_signal(
            df.iloc[:i + 1]["Close"].tolist()
        )

        # HMA may not be ready yet
        if signal_data is None:
            continue

        signal = signal_data["signal"]

        if signal is None:
            continue
        
        ema200 = row["EMA200"]

        # =========================
        # LONG BIAS
        # =========================

        if signal == "LONG":

            if current_price < ema200:
                continue

            if hma_fast < hma_slow:
                continue


        # =========================
        # SHORT BIAS
        # =========================

        elif signal == "SHORT":

            if current_price > ema200:
                continue

            if hma_fast > hma_slow:
                continue



        # HMA Gap Filter
        hma_gap = abs(
            signal_data["hma_fast"]
            -
            signal_data["hma_slow"]
        )

        if hma_gap < atr:
            continue

        # Strong Trend Filter
        if abs(signal_data["slope"]) < atr * 0.50:
            continue
        

        quality_score = 0

        # ADX
        if adx > 30:
            quality_score += 1

        # HMA Gap
        if hma_gap > atr * 1.5:
            quality_score += 1

        # Slope
        if abs(signal_data["slope"]) > atr:
            quality_score += 1

        # VROC
        if vroc > VROC_THRESH:
            quality_score += 1

        if quality_score < QUALITY_THRESH:
            continue


        position = signal

        entry_price = current_price
        entry_time = current_time

        position_size = risk_engine.calculate_position_size(
            atr
        )

        if position_size <= 0:
            continue

        if signal == "LONG":

            stop_loss = risk_engine.long_stop(
                entry_price,
                atr
            )

            target_price = risk_engine.long_target(
                entry_price,
                atr
            )

        else:

            stop_loss = risk_engine.short_stop(
                entry_price,
                atr
            )

            target_price = risk_engine.short_target(
                entry_price,
                atr
            )

        print(
            f"{current_time} | ENTRY {position} @ {entry_price:.2f}"
        )

        print(
            f"Regime: {regime}"
        )

        print(
            f"Quality Score: {quality_score}/4"
        )

        print(
            f"HMA Fast: {signal_data['hma_fast']} | "
            f"HMA Slow: {signal_data['hma_slow']} | "
            f"Slope: {signal_data['slope']}"
        )

        print(
            f"HMA Gap: {round(hma_gap, 2)}"
        )

        print(
            f"ATR: {round(atr, 2)}"
        )

    # =================================
    # EXIT LOGIC
    # =================================

    else:
        if position == "LONG":

            stop_loss = trailing_engine.long_trail(

                current_price,

                stop_loss,

                atr

            )
            print(
                f"Trailing Stop: {round(stop_loss, 2)}"
            )
        elif position == "SHORT":

            stop_loss = trailing_engine.short_trail(

                current_price,

                stop_loss,

                atr

            )
        should_exit = False


        # =================================
        # LONG POSITION
        # =================================

        if position == "LONG":

            if current_price <= stop_loss:

                should_exit = True

            elif current_price >= target_price:

                should_exit = True


        # =================================
        # SHORT POSITION
        # =================================

        elif position == "SHORT":

            if current_price >= stop_loss:

                should_exit = True

            elif current_price <= target_price:

                should_exit = True


        # =================================
        # EXECUTE EXIT
        # =================================

        if should_exit:

            exit_reason = "TARGET"

            if (
                position == "LONG"
                and
                current_price <= stop_loss
            ):
                exit_reason = "STOP LOSS"

            elif (
                position == "SHORT"
                and
                current_price >= stop_loss
            ):
                exit_reason = "STOP LOSS"

            print(
                f"{current_time} | {exit_reason} HIT"
            )


            # =========================
            # CALCULATE PNL
            # =========================

            if position == "LONG":

                pnl = (

                    current_price
                    -
                    entry_price

                ) * position_size

            else:

                pnl = (

                    entry_price
                    -
                    current_price

                ) * position_size


            # =========================
            # UPDATE PORTFOLIO
            # =========================

            portfolio.update_pnl(
                pnl
            )


            # =========================
            # SAVE TRADE
            # =========================

            trade = {

                "datetime": current_time,

                "position": position,

                "size": position_size,

                "entry_price": round(
                    entry_price,
                    2
                ),

                "exit_price": round(
                    current_price,
                    2
                ),

                "stop_loss": round(
                    stop_loss,
                    2
                ),

                "target_price": round(
                    target_price,
                    2
                ),

                "pnl": round(
                    pnl,
                    2
                ),

                "vroc": round(
                    vroc,
                    2
                ),

                "adx": round(
                    adx,
                    2
                ),

                "atr": round(
                    atr,
                    2
                ),

                "regime": regime,

                "exit_reason": exit_reason

            }

            trade_log.append(
                trade
            )


            # =========================
            # TERMINAL OUTPUT
            # =========================

            print(

                f"{current_time} | "
                f"EXIT {position} @ "
                f"{current_price:.2f} | "
                f"Size: {position_size} | "
                f"PnL: {pnl:.2f}"

            )


            # =========================
            # RESET POSITION
            # =========================

            position = None

            entry_price = 0

            stop_loss = 0

            target_price = 0

            position_size = 0



# =====================================
# SAVE TRADE LOG
# =====================================

trade_df = pd.DataFrame(trade_log)
if "pnl" not in trade_df.columns:
    trade_df = pd.DataFrame(
        trade_log,
        columns=[
            "datetime", "position", "size", "entry_price", "exit_price",
            "stop_loss", "target_price", "pnl", "vroc", "adx", "atr",
            "regime", "exit_reason"
        ]
    )

performance = PerformanceAnalyzer(
    trade_df
)


trade_df.to_csv(

    OUTPUT_TRADES_FILE,

    index=False
)


# =====================================
# FINAL STATS
# =====================================

stats = portfolio.get_stats()


print("\n==========================")
print("BACKTEST COMPLETE")
print("==========================")

print(f"Final Capital : {stats['capital']:.2f}")

print(f"Realized PnL : {stats['realized_pnl']:.2f}")

print(f"Total Trades : {stats['total_trades']}")

print(f"Wins          : {stats['wins']}")

print(f"Losses        : {stats['losses']}")

print(f"Win Rate      : {stats['win_rate']:.2f}%")

print("==========================\n")


print(
    "\n=========================="
)

print(
    "PERFORMANCE REPORT"
)

print(
    "=========================="
)

print(
    f"Profit Factor : {performance.profit_factor()}"
)

print(
    f"Avg Win       : {performance.avg_win()}"
)

print(
    f"Avg Loss      : {performance.avg_loss()}"
)

print(
    f"Risk Reward   : {performance.risk_reward()}"
)

print(
    f"Expectancy    : {performance.expectancy()}"
)

print(
    f"Max Drawdown  : {performance.max_drawdown()}"
)

print(
    f"Sharpe Ratio  : {performance.sharpe_ratio()}"
)

print(
    "=========================="
)


# =====================================
# SAVE SUMMARY
# =====================================

summary = {

    # Portfolio Statistics
    "capital": stats["capital"],
    "realized_pnl": stats["realized_pnl"],
    "total_trades": stats["total_trades"],
    "wins": stats["wins"],
    "losses": stats["losses"],
    "win_rate": stats["win_rate"],

    # Performance Statistics
    "profit_factor": performance.profit_factor(),
    "avg_win": performance.avg_win(),
    "avg_loss": performance.avg_loss(),
    "risk_reward": performance.risk_reward(),
    "expectancy": performance.expectancy(),
    "max_drawdown": performance.max_drawdown(),
    "sharpe_ratio": performance.sharpe_ratio()

}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(

    args.summary,

    index=False

)

print("\nTrade log saved:")
print(args.trades)

print("\nSummary saved:")
print(args.summary)