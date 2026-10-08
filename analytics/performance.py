import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# =====================================
# LOAD TRADE LOG
# =====================================

TRADE_FILE = "backtesting/backtest_trades.csv"

df = pd.read_csv(TRADE_FILE)


# =====================================
# CREATE ANALYTICS FOLDER
# =====================================

os.makedirs("analytics", exist_ok=True)


# =====================================
# BASIC METRICS
# =====================================

total_trades = len(df)

wins = df[df["pnl"] > 0]

losses = df[df["pnl"] <= 0]

gross_profit = wins["pnl"].sum()

gross_loss = abs(losses["pnl"].sum())

net_profit = df["pnl"].sum()

win_rate = (
    len(wins) / total_trades * 100
    if total_trades > 0
    else 0
)

avg_win = (
    wins["pnl"].mean()
    if len(wins) > 0
    else 0
)

avg_loss = (
    losses["pnl"].mean()
    if len(losses) > 0
    else 0
)

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0
    else 0
)


# =====================================
# EXPECTANCY
# =====================================

expectancy = (

    (win_rate / 100) * avg_win

) + (

    (1 - (win_rate / 100)) * avg_loss
)


# =====================================
# EQUITY CURVE
# =====================================

initial_capital = 100000

df["equity"] = (
    initial_capital
    + df["pnl"].cumsum()
)


# =====================================
# DRAWDOWN
# =====================================

df["peak"] = df["equity"].cummax()

df["drawdown"] = (

    df["equity"]
    -
    df["peak"]

)

max_drawdown = df["drawdown"].min()


# =====================================
# RECOVERY FACTOR
# =====================================

if abs(max_drawdown) > 0:

    recovery_factor = (

        net_profit
        /
        abs(max_drawdown)

    )

else:

    recovery_factor = 0


# =====================================
# LARGEST WIN / LOSS
# =====================================

largest_win = df["pnl"].max()

largest_loss = df["pnl"].min()


# =====================================
# RISK REWARD
# =====================================

if avg_loss != 0:

    risk_reward = abs(
        avg_win / avg_loss
    )

else:

    risk_reward = 0


# =====================================
# SHARPE RATIO
# =====================================

returns = df["pnl"]

if returns.std() != 0:

    sharpe_ratio = (

        returns.mean()
        /
        returns.std()

    ) * np.sqrt(252)

else:

    sharpe_ratio = 0


# =====================================
# PRINT RESULTS
# =====================================

print("\n==============================")
print("PERFORMANCE ANALYTICS")
print("==============================")

print(f"Total Trades    : {total_trades}")

print(f"Net Profit      : {net_profit:.2f}")

print(f"Win Rate        : {win_rate:.2f}%")

print(f"Average Win     : {avg_win:.2f}")

print(f"Average Loss    : {avg_loss:.2f}")

print(f"Profit Factor   : {profit_factor:.2f}")

print(f"Risk Reward     : {risk_reward:.2f}")

print(f"Expectancy      : {expectancy:.2f}")

print(f"Largest Win     : {largest_win:.2f}")

print(f"Largest Loss    : {largest_loss:.2f}")

print(f"Max Drawdown    : {max_drawdown:.2f}")

print(f"Recovery Factor : {recovery_factor:.2f}")

print(f"Sharpe Ratio    : {sharpe_ratio:.2f}")

print("==============================\n")


# =====================================
# EQUITY CURVE CHART
# =====================================

plt.figure(figsize=(12, 6))

plt.plot(df["equity"])

plt.title("Equity Curve")

plt.xlabel("Trades")

plt.ylabel("Portfolio Value")

plt.grid(True)

plt.savefig(
    "analytics/equity_curve.png"
)

print(
    "Saved: analytics/equity_curve.png"
)


# =====================================
# DRAWDOWN CHART
# =====================================

plt.figure(figsize=(12, 6))

plt.plot(df["drawdown"])

plt.title("Drawdown Curve")

plt.xlabel("Trades")

plt.ylabel("Drawdown")

plt.grid(True)

plt.savefig(
    "analytics/drawdown.png"
)

print(
    "Saved: analytics/drawdown.png"
)


# =====================================
# MONTHLY RETURNS
# =====================================

df["datetime"] = pd.to_datetime(
    df["datetime"],
    utc=True
)

df["datetime"] = (
    df["datetime"]
    .dt.tz_localize(None)
)

df["month"] = (
    df["datetime"]
    .dt.to_period("M")
)

monthly_returns = (

    df.groupby("month")["pnl"]

    .sum()

    .reset_index()
)

monthly_returns.to_csv(

    "analytics/monthly_returns.csv",

    index=False
)

print(
    "Saved: analytics/monthly_returns.csv"
)


# =====================================
# PERFORMANCE REPORT
# =====================================

report = pd.DataFrame([{

    "total_trades": total_trades,

    "net_profit": net_profit,

    "win_rate": win_rate,

    "avg_win": avg_win,

    "avg_loss": avg_loss,

    "profit_factor": profit_factor,

    "risk_reward": risk_reward,

    "expectancy": expectancy,

    "largest_win": largest_win,

    "largest_loss": largest_loss,

    "max_drawdown": max_drawdown,

    "recovery_factor": recovery_factor,

    "sharpe_ratio": sharpe_ratio

}])

report.to_csv(

    "analytics/performance_report.csv",

    index=False
)

print(
    "Saved: analytics/performance_report.csv"
)