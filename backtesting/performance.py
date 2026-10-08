import pandas as pd
import numpy as np


class PerformanceAnalyzer:

    def __init__(self, trades_df):

        self.df = trades_df.copy()
        if "pnl" not in self.df.columns:
            self.df["pnl"] = pd.Series(dtype=float)

    # ==========================
    # PROFIT FACTOR
    # ==========================

    def profit_factor(self):

        gross_profit = self.df[
            self.df["pnl"] > 0
        ]["pnl"].sum()

        gross_loss = abs(

            self.df[
                self.df["pnl"] < 0
            ]["pnl"].sum()

        )

        if gross_loss == 0:
            if gross_profit > 0:
                # All trades are winners (zero losses) — profit factor is mathematically infinite.
                return float("inf")
            return 0.0

        return round(
            gross_profit / gross_loss,
            2
        )

    # ==========================
    # AVG WIN
    # ==========================

    def avg_win(self):

        wins = self.df[
            self.df["pnl"] > 0
        ]

        if len(wins) == 0:
            return 0

        return round(
            wins["pnl"].mean(),
            2
        )

    # ==========================
    # AVG LOSS
    # ==========================

    def avg_loss(self):

        losses = self.df[
            self.df["pnl"] < 0
        ]

        if len(losses) == 0:
            return 0

        return round(
            losses["pnl"].mean(),
            2
        )

    # ==========================
    # RISK REWARD
    # ==========================

    def risk_reward(self):

        avg_win = self.avg_win()

        avg_loss = abs(
            self.avg_loss()
        )

        if avg_loss == 0:
            return 0

        return round(
            avg_win / avg_loss,
            2
        )

    # ==========================
    # EXPECTANCY
    # ==========================

    def expectancy(self):

        total = len(self.df)

        if total == 0:
            return 0

        wins = len(
            self.df[
                self.df["pnl"] > 0
            ]
        )

        losses = len(
            self.df[
                self.df["pnl"] < 0
            ]
        )

        win_rate = wins / total

        loss_rate = losses / total

        avg_win = self.avg_win()

        avg_loss = abs(
            self.avg_loss()
        )

        expectancy = (

            win_rate * avg_win

            -

            loss_rate * avg_loss

        )

        return round(
            expectancy,
            2
        )

    # ==========================
    # MAX DRAWDOWN
    # ==========================

    def max_drawdown(self):

        if len(self.df) == 0:
            return 0.0

        equity = self.df["pnl"].cumsum()

        running_max = equity.cummax()

        drawdown = equity - running_max

        val = drawdown.min()

        return round(float(val), 2) if not pd.isna(val) else 0.0


    # ==========================
    # SHARPE RATIO
    # ==========================

    def sharpe_ratio(self):
        """
        Per-trade Sharpe ratio (NOT annualized).

        Computed as: mean(per_trade_returns) / std(per_trade_returns)
        where per_trade_return = trade_pnl / INITIAL_CAPITAL (100,000).

        NOT annualized intentionally:
          - Annualizing requires a known trades/year frequency.
          - With very few trades, sqrt(252) produces misleading values
            (e.g. Sharpe=20 from 2 trades is noise, not signal).

        Interpretation:
          > 0.5  : Consistent per-trade edge
          0.2-0.5: Moderate consistency
          < 0.2  : Low edge or high variance
          0       : Insufficient data (< 2 trades)
        """

        INITIAL_CAPITAL = 100000

        if len(self.df) < 2:
            return 0

        per_trade_returns = self.df["pnl"] / INITIAL_CAPITAL

        std = per_trade_returns.std()

        if std == 0:
            return 0

        sharpe = per_trade_returns.mean() / std

        return round(sharpe, 4)