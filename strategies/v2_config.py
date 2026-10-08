"""
Strategy V2 — EMA5/EMA9 Trend Reversal
Configuration Module

All parameters below are configurable.  Parameters marked [ASSUMPTION] were
NOT precisely specified by the strategy designer; sensible defaults are used
and must be validated/adjusted once 3-minute data is available.

Parameters marked [RULE] come directly from the designer's specification and
must NOT be changed without explicit designer sign-off.

DATA DEPENDENCY NOTE
--------------------
Strategy V2 requires 3-minute candle data for RELIANCE.NS.
The existing backtesting/historical_data.csv contains only 1-hour bars.
This is a BLOCKER for live backtesting.  All unit tests use synthetic
data generated within the test suite itself and are data-independent.
"""

# ============================================================
# TIMEFRAME
# ============================================================

TIMEFRAME = "3min"            # [RULE] primary timeframe

# ============================================================
# EMA PERIODS
# ============================================================

EMA_FAST_PERIOD = 5           # [RULE] EMA5 — primary trend filter
EMA_SLOW_PERIOD = 9           # [RULE] EMA9 — secondary trend context

# ============================================================
# CANDLE PATTERN THRESHOLDS
# ============================================================

# --- Doji ---
# [ASSUMPTION] Body-to-range ratio below which a candle is considered a Doji.
# A common industry default is 0.10 (body <= 10% of high-low range).
DOJI_BODY_RATIO = 0.10        # body_size / candle_range <= threshold -> Doji

# --- Shooting Star / Hammer ---
# [ASSUMPTION] Upper wick must be >= this multiple of the body size.
# Standard definition: upper wick >= 2x body.
SHOOTING_STAR_WICK_RATIO = 2.0   # upper_wick >= ratio * body_size
# [ASSUMPTION] Lower wick must be >= this multiple of the body size for Hammer.
HAMMER_WICK_RATIO = 2.0          # lower_wick >= ratio * body_size
# [ASSUMPTION] Body must be within the lower portion of the candle range.
# For Shooting Star: body sits in the UPPER portion of the range.
# For Hammer: body sits in the LOWER portion of the range.
# Expressed as max fraction of range the body bottom can be above the candle low.
SHOOTING_STAR_BODY_POSITION = 0.35  # body_bottom / range from top <= this
HAMMER_BODY_POSITION = 0.35         # body_bottom / range from bottom <= this

# --- Bearish / Bullish Engulfing ---
# [RULE] Engulfing is body-based only (explicitly confirmed by designer).
# The current candle's body must fully engulf the prior candle's body.
# No additional threshold required beyond the mathematical definition.

# ============================================================
# STOP-LOSS
# ============================================================

# [ASSUMPTION] SL is placed above the high (short) or below the low (long)
# of the reversal candle, plus a configurable buffer expressed as a fraction
# of the candle range.  Buffer of 0.0 places SL exactly at the wick extreme.
SL_BUFFER_FRACTION = 0.0     # fraction of candle range added to wick-to-SL gap
# [ASSUMPTION] Minimum absolute SL distance in price units to avoid near-zero
# stops on very tight doji candles.
SL_MIN_DISTANCE = 0.50       # minimum price-unit distance for SL

# ============================================================
# TARGET — PRIOR SWING LEVEL
# ============================================================

# [RULE] Target = prior market-structure swing level (where the preceding
# move originated), NOT simply the previous candle high/low.
# Implementation: causal swing detection (no lookahead).

# [ASSUMPTION] Lookback window (in bars) used to identify the prior swing.
# Searches backward from the reversal candle to find the most recent
# significant swing high (for short) / swing low (for long).
SWING_LOOKBACK = 30           # bars to look back for swing level

# [ASSUMPTION] A local high/low is considered a swing pivot if it is the
# highest/lowest point in this many consecutive prior bars.
# Only left-side neighbours are checked (right-side would introduce lookahead).
SWING_STRENGTH = 3            # prior bars that must be lower/higher

# [ASSUMPTION] Minimum reward-to-risk ratio.  Trades where the distance to
# the swing target does not meet this threshold vs the SL distance are skipped.
MIN_RR_RATIO = 1.0            # target_distance / sl_distance >= this

# ============================================================
# POSITION SIZING
# ============================================================

# [ASSUMPTION] Risk-per-trade as a fraction of capital.
RISK_PER_TRADE = 0.01         # 1% of capital

# [ASSUMPTION] Starting capital in INR.
INITIAL_CAPITAL = 100_000

# ============================================================
# EMA COMPARISON MODE
# ============================================================

# [RULE] Sell setup: price above EMA5 (and EMA9 when comparison is enabled).
# [RULE] Buy setup: price below EMA5 (and EMA9 when comparison is enabled).
# [ASSUMPTION] Whether to also require price relative to EMA9 in addition to EMA5.
REQUIRE_EMA9_CONFIRMATION = True   # if True, price must be on same side of EMA9

# ============================================================
# WARMUP BARS
# ============================================================

# [ASSUMPTION] Minimum bars needed before emitting signals (allows EMAs to
# stabilise).  Should be >= max(EMA9, SWING_LOOKBACK) + SWING_STRENGTH.
WARMUP_BARS = 50
