# strategies/__init__.py
# Strategy registry for vroc_algo project.
#
# V1: VROC Breakout/Reversal — implemented in backtesting/backtest.py (frozen)
# V2: EMA5/EMA9 Trend Reversal — implemented in strategies/strategy_v2_ema_reversal.py
#
# IMPORTANT: Do NOT import V1 backtest logic here. V1 is intentionally kept
# in its own frozen module (backtesting/backtest.py) to prevent contamination.
