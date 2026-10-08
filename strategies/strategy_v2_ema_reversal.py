"""
Strategy V2 — EMA5/EMA9 Trend Reversal
Core Strategy Logic

This module provides:
  - EMA computation (pure, no external indicator library dependency)
  - Candle pattern detectors (Doji, Shooting Star, Hammer,
    Bearish Engulfing, Bullish Engulfing)
  - Causal swing level detector (no lookahead)
  - Signal generator combining all of the above
  - Stop-loss and position-size calculators

DESIGN PRINCIPLES
-----------------
1. No lookahead bias: at bar i, only data up to and including bar i is used.
2. All pattern parameters are explicit keyword arguments (configurable from
   v2_config.py or directly in backtesting/tests).
3. Strategy logic is pure functions — they accept DataFrames / arrays and
   return plain Python types.  No global state.
4. V1 (backtesting/backtest.py) is NOT imported or modified here.

PATTERN RULES (per designer specification)
-------------------------------------------
Doji       : body <= DOJI_BODY_RATIO * candle_range               [ASSUMPTION threshold]
Shooting   : bearish candle; upper_wick >= SHOOTING_STAR_WICK_RATIO * body;
  Star       body located in upper portion of candle range          [ASSUMPTION threshold]
Hammer     : bullish candle; lower_wick >= HAMMER_WICK_RATIO * body;
               body located in lower portion of candle range        [ASSUMPTION threshold]
Bearish    : current bearish body fully engulfs prior candle body  [RULE: body-based]
  Engulfing
Bullish    : current bullish body fully engulfs prior candle body  [RULE: body-based]
  Engulfing

ENTRY RULE
----------
Enter on Candle N+1 (open of next bar after the reversal candle closes).  [RULE]
No look-ahead: the entry price is the open of bar i+1.

TARGET RULE
-----------
Prior market-structure swing level (causal, no-lookahead).                [RULE]
NOT simply the previous candle high/low.
"""

import math
import pandas as pd
import numpy as np


# ===========================================================
# EMA
# ===========================================================

def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Uses pandas ewm with adjust=False (standard recursive EMA).
    Returns a Series aligned with `close`, NaN for the first (period-1) rows.
    """
    ema = close.ewm(span=period, adjust=False).mean()
    # Blank out the warm-up rows to make the NaN boundary explicit.
    ema.iloc[:period - 1] = float("nan")
    return ema


# ===========================================================
# CANDLE HELPERS
# ===========================================================

def _candle_parts(open_: float, high: float, low: float, close: float):
    """
    Decompose a single OHLC candle into its parts.

    Returns (body_size, upper_wick, lower_wick, candle_range).
    body_size and wicks are always non-negative.
    """
    body_top = max(open_, close)
    body_bot = min(open_, close)
    body_size = body_top - body_bot
    candle_range = high - low
    upper_wick = high - body_top
    lower_wick = body_bot - low
    return body_size, upper_wick, lower_wick, candle_range


# ===========================================================
# PATTERN DETECTORS — single bar
# ===========================================================

def is_doji(
    open_: float,
    high: float,
    low: float,
    close: float,
    body_ratio: float = 0.10,
) -> bool:
    """
    True if the candle's body is <= body_ratio * candle_range.

    A zero-range candle (open == high == low == close) is treated as a Doji.
    [ASSUMPTION: body_ratio = 0.10]
    """
    body_size, _, _, candle_range = _candle_parts(open_, high, low, close)
    if candle_range == 0:
        return True
    return (body_size / candle_range) <= body_ratio


def is_shooting_star(
    open_: float,
    high: float,
    low: float,
    close: float,
    wick_ratio: float = 2.0,
    body_position: float = 0.35,
) -> bool:
    """
    True if the candle meets Shooting Star criteria:
      1. Bearish close (close < open_)                [directional check]
      2. upper_wick >= wick_ratio * body_size         [ASSUMPTION: wick_ratio=2.0]
      3. Body sits in the UPPER portion of the range:
         body_bottom (= close for a bearish candle) is within
         body_position fraction of the range FROM THE TOP.    [ASSUMPTION: 0.35]

    A candle with zero body (pure doji) is excluded — use is_doji() instead.
    """
    if close >= open_:
        return False  # not bearish
    body_size, upper_wick, lower_wick, candle_range = _candle_parts(
        open_, high, low, close
    )
    if body_size == 0 or candle_range == 0:
        return False
    if upper_wick < wick_ratio * body_size:
        return False
    # body_bottom = close (bearish candle)
    body_bottom_from_top = high - close
    if (body_bottom_from_top / candle_range) > body_position:
        return False
    return True


def is_hammer(
    open_: float,
    high: float,
    low: float,
    close: float,
    wick_ratio: float = 2.0,
    body_position: float = 0.35,
) -> bool:
    """
    True if the candle meets Hammer criteria:
      1. Bullish close (close > open_)                [directional check]
      2. lower_wick >= wick_ratio * body_size         [ASSUMPTION: wick_ratio=2.0]
      3. Body sits in the LOWER portion of the range:
         body_bottom (= open_ for bullish) is within
         body_position fraction of the range FROM THE BOTTOM.  [ASSUMPTION: 0.35]

    A candle with zero body is excluded.
    """
    if close <= open_:
        return False  # not bullish
    body_size, upper_wick, lower_wick, candle_range = _candle_parts(
        open_, high, low, close
    )
    if body_size == 0 or candle_range == 0:
        return False
    if lower_wick < wick_ratio * body_size:
        return False
    # body_bottom = open_ (bullish candle)
    body_bottom_from_low = open_ - low
    if (body_bottom_from_low / candle_range) > body_position:
        return False
    return True


def is_bearish_engulfing(
    open_prev: float,
    close_prev: float,
    open_curr: float,
    close_curr: float,
) -> bool:
    """
    True if the current candle is a Bearish Engulfing pattern.

    Rules (body-based per designer specification [RULE]):
      1. Previous candle is bullish:  close_prev > open_prev
      2. Current candle is bearish:   close_curr < open_curr
      3. Current body fully engulfs previous body:
           open_curr >= close_prev   AND   close_curr <= open_prev

    Strict inequality on direction; >= / <= on engulfing to allow exact equality.
    """
    prev_bullish = close_prev > open_prev
    curr_bearish = close_curr < open_curr
    if not prev_bullish or not curr_bearish:
        return False
    # current open >= prev close (top of prev body) AND
    # current close <= prev open (bottom of prev body)
    return open_curr >= close_prev and close_curr <= open_prev


def is_bullish_engulfing(
    open_prev: float,
    close_prev: float,
    open_curr: float,
    close_curr: float,
) -> bool:
    """
    True if the current candle is a Bullish Engulfing pattern.

    Rules (body-based per designer specification [RULE]):
      1. Previous candle is bearish:  close_prev < open_prev
      2. Current candle is bullish:   close_curr > open_curr
      3. Current body fully engulfs previous body:
           open_curr <= close_prev   AND   close_curr >= open_prev
    """
    prev_bearish = close_prev < open_prev
    curr_bullish = close_curr > open_curr
    if not prev_bearish or not curr_bullish:
        return False
    return open_curr <= close_prev and close_curr >= open_prev


# ===========================================================
# COMPOSITE BEARISH / BULLISH PATTERN CHECK
# ===========================================================

def is_bearish_reversal_pattern(
    open_prev: float,
    close_prev: float,
    open_curr: float,
    high_curr: float,
    low_curr: float,
    close_curr: float,
    doji_body_ratio: float = 0.10,
    shooting_star_wick_ratio: float = 2.0,
    shooting_star_body_position: float = 0.35,
) -> tuple:
    """
    Returns (is_signal: bool, pattern_name: str).

    Bearish reversal patterns (in priority order):
      1. Bearish Engulfing
      2. Shooting Star
      3. Doji (indecision — treated as potential reversal in bearish context)

    Note: Doji alone is NOT a directional signal; it is only accepted here
    when the EMA context (price > EMA5) is already confirmed by the caller.
    [RULE: Doji represents indecision, NOT unconditional reversal]
    """
    if is_bearish_engulfing(open_prev, close_prev, open_curr, close_curr):
        return True, "BEARISH_ENGULFING"
    if is_shooting_star(
        open_curr, high_curr, low_curr, close_curr,
        wick_ratio=shooting_star_wick_ratio,
        body_position=shooting_star_body_position,
    ):
        return True, "SHOOTING_STAR"
    if is_doji(open_curr, high_curr, low_curr, close_curr, body_ratio=doji_body_ratio):
        return True, "DOJI"
    return False, ""


def is_bullish_reversal_pattern(
    open_prev: float,
    close_prev: float,
    open_curr: float,
    high_curr: float,
    low_curr: float,
    close_curr: float,
    doji_body_ratio: float = 0.10,
    hammer_wick_ratio: float = 2.0,
    hammer_body_position: float = 0.35,
) -> tuple:
    """
    Returns (is_signal: bool, pattern_name: str).

    Bullish reversal patterns (in priority order):
      1. Bullish Engulfing
      2. Hammer
      3. Doji (indecision — treated as potential reversal in bullish context)
    """
    if is_bullish_engulfing(open_prev, close_prev, open_curr, close_curr):
        return True, "BULLISH_ENGULFING"
    if is_hammer(
        open_curr, high_curr, low_curr, close_curr,
        wick_ratio=hammer_wick_ratio,
        body_position=hammer_body_position,
    ):
        return True, "HAMMER"
    if is_doji(open_curr, high_curr, low_curr, close_curr, body_ratio=doji_body_ratio):
        return True, "DOJI"
    return False, ""


# ===========================================================
# CAUSAL SWING DETECTOR — NO LOOKAHEAD
# ===========================================================

def find_prior_swing_high(
    highs: pd.Series,
    current_idx: int,
    lookback: int = 30,
    strength: int = 3,
) -> float | None:
    """
    Find the most recent swing high strictly before current_idx.

    A bar at position j is a swing high if:
        highs[j] > max(highs[j - strength : j])   (left-side only)

    This is purely causal — only uses data up to and including current_idx - 1.
    Returns the price level of the most recent qualifying swing high,
    or None if none found within the lookback window.

    [RULE] No lookahead: right-side bars (j+1 ...) are NOT checked.
    """
    search_start = max(0, current_idx - lookback)
    search_end = current_idx  # exclusive — do not include current bar

    # Iterate from most recent backward
    for j in range(search_end - 1, search_start + strength - 1, -1):
        window_high = highs.iloc[j - strength:j].max()
        if highs.iloc[j] > window_high:
            return float(highs.iloc[j])
    return None


def find_prior_swing_low(
    lows: pd.Series,
    current_idx: int,
    lookback: int = 30,
    strength: int = 3,
) -> float | None:
    """
    Find the most recent swing low strictly before current_idx.

    A bar at position j is a swing low if:
        lows[j] < min(lows[j - strength : j])   (left-side only)

    Purely causal — no lookahead.
    Returns the price level or None.

    [RULE] No lookahead.
    """
    search_start = max(0, current_idx - lookback)
    search_end = current_idx

    for j in range(search_end - 1, search_start + strength - 1, -1):
        window_low = lows.iloc[j - strength:j].min()
        if lows.iloc[j] < window_low:
            return float(lows.iloc[j])
    return None


# ===========================================================
# STOP LOSS CALCULATOR
# ===========================================================

def calculate_short_sl(
    reversal_high: float,
    reversal_range: float,
    buffer_fraction: float = 0.0,
    min_distance: float = 0.50,
) -> float:
    """
    Stop loss for a SHORT entry.
    SL = reversal_candle_high + buffer.
    Buffer = buffer_fraction * reversal_range (at minimum min_distance).
    [RULE] SL above reversal wick.
    """
    buffer = max(buffer_fraction * reversal_range, min_distance)
    return reversal_high + buffer


def calculate_long_sl(
    reversal_low: float,
    reversal_range: float,
    buffer_fraction: float = 0.0,
    min_distance: float = 0.50,
) -> float:
    """
    Stop loss for a LONG entry.
    SL = reversal_candle_low - buffer.
    Buffer = buffer_fraction * reversal_range (at minimum min_distance).
    [RULE] SL below reversal wick.
    """
    buffer = max(buffer_fraction * reversal_range, min_distance)
    return reversal_low - buffer


# ===========================================================
# POSITION SIZE
# ===========================================================

def calculate_position_size(
    capital: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss: float,
) -> int:
    """
    Position size = floor(risk_amount / sl_distance).
    Returns 0 if stop distance is zero or negative (prevents invalid trades).
    """
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance <= 0:
        return 0
    risk_amount = capital * risk_per_trade
    return int(risk_amount / sl_distance)


# ===========================================================
# SIGNAL GENERATOR — MAIN ENTRY POINT
# ===========================================================

def generate_signals(
    df: pd.DataFrame,
    ema_fast_period: int = 5,
    ema_slow_period: int = 9,
    require_ema9: bool = True,
    doji_body_ratio: float = 0.10,
    shooting_star_wick_ratio: float = 2.0,
    shooting_star_body_position: float = 0.35,
    hammer_wick_ratio: float = 2.0,
    hammer_body_position: float = 0.35,
    swing_lookback: int = 30,
    swing_strength: int = 3,
    min_rr_ratio: float = 1.0,
    sl_buffer_fraction: float = 0.0,
    sl_min_distance: float = 0.50,
    warmup_bars: int = 50,
) -> pd.DataFrame:
    """
    Generate V2 trade signals from an OHLCV DataFrame.

    Input columns required: Open, High, Low, Close (Volume not used here).
    Input must be sorted ascending by time with a DatetimeIndex or sequential
    integer index.

    Returns a DataFrame with columns:
        signal_bar     : integer index of the reversal candle (bar N)
        entry_bar      : integer index of the entry bar (bar N+1)
        direction      : "SHORT" or "LONG"
        pattern        : pattern name string
        entry_price    : open of bar N+1   [RULE: next-bar entry, no lookahead]
        stop_loss      : calculated SL
        target         : prior swing level
        ema5           : EMA5 at bar N
        ema9           : EMA9 at bar N
        rr_ratio       : (target_dist / sl_dist)
        valid          : True if RR >= min_rr_ratio AND target found
        skip_reason    : explanation if valid=False, else ""

    NO_LOOKAHEAD CONTRACT:
        At signal_bar=i, only indices [0..i] are accessed.
        Entry price uses df["Open"].iloc[i+1] which is the NEXT bar open —
        this is valid because the signal is emitted at bar i's CLOSE and the
        order is placed for bar i+1's open.
    """
    df = df.copy().reset_index(drop=True)

    opens = df["Open"]
    highs = df["High"]
    lows = df["Low"]
    closes = df["Close"]

    ema5 = calculate_ema(closes, ema_fast_period)
    ema9 = calculate_ema(closes, ema_slow_period)

    records = []

    for i in range(warmup_bars, len(df) - 1):
        # --- EMA values at bar i ---
        e5 = ema5.iloc[i]
        e9 = ema9.iloc[i]
        if pd.isna(e5) or pd.isna(e9):
            continue

        o_curr = float(opens.iloc[i])
        h_curr = float(highs.iloc[i])
        l_curr = float(lows.iloc[i])
        c_curr = float(closes.iloc[i])

        o_prev = float(opens.iloc[i - 1])
        c_prev = float(closes.iloc[i - 1])

        # --- SHORT SETUP ---
        # Price above EMA5 (and EMA9 if require_ema9) at bar i
        short_ema_ok = c_curr > e5
        if require_ema9:
            short_ema_ok = short_ema_ok and (c_curr > e9)

        if short_ema_ok:
            is_pat, pat_name = is_bearish_reversal_pattern(
                o_prev, c_prev, o_curr, h_curr, l_curr, c_curr,
                doji_body_ratio=doji_body_ratio,
                shooting_star_wick_ratio=shooting_star_wick_ratio,
                shooting_star_body_position=shooting_star_body_position,
            )
            if is_pat:
                entry_price = float(opens.iloc[i + 1])
                reversal_range = h_curr - l_curr
                sl = calculate_short_sl(
                    h_curr, reversal_range, sl_buffer_fraction, sl_min_distance
                )
                target = find_prior_swing_high(
                    highs, i, lookback=swing_lookback, strength=swing_strength
                )
                sl_dist = abs(sl - entry_price)
                if target is not None:
                    # For a short, target should be BELOW entry (profit zone)
                    # The swing high found is the ORIGIN of the preceding up-move
                    # i.e. the level where the rise started — this is where
                    # price may fall back to (swing support / prior structure)
                    # Reclarification: find the prior swing LOW as the profit target
                    # for a short (where the preceding up-move originated from).
                    swing_low_target = find_prior_swing_low(
                        lows, i, lookback=swing_lookback, strength=swing_strength
                    )
                    target_price = swing_low_target
                else:
                    target_price = None

                valid, skip_reason = _validate_trade(
                    "SHORT", entry_price, sl, target_price, min_rr_ratio
                )
                records.append({
                    "signal_bar": i,
                    "entry_bar": i + 1,
                    "direction": "SHORT",
                    "pattern": pat_name,
                    "entry_price": entry_price,
                    "stop_loss": sl,
                    "target": target_price,
                    "ema5": float(e5),
                    "ema9": float(e9),
                    "rr_ratio": _rr_ratio("SHORT", entry_price, sl, target_price),
                    "valid": valid,
                    "skip_reason": skip_reason,
                })

        # --- LONG SETUP ---
        # Price below EMA5 (and EMA9 if require_ema9) at bar i
        long_ema_ok = c_curr < e5
        if require_ema9:
            long_ema_ok = long_ema_ok and (c_curr < e9)

        if long_ema_ok:
            is_pat, pat_name = is_bullish_reversal_pattern(
                o_prev, c_prev, o_curr, h_curr, l_curr, c_curr,
                doji_body_ratio=doji_body_ratio,
                hammer_wick_ratio=hammer_wick_ratio,
                hammer_body_position=hammer_body_position,
            )
            if is_pat:
                entry_price = float(opens.iloc[i + 1])
                reversal_range = h_curr - l_curr
                sl = calculate_long_sl(
                    l_curr, reversal_range, sl_buffer_fraction, sl_min_distance
                )
                # For a long, target = prior swing HIGH (where preceding down-move
                # originated — the level from which price fell)
                swing_high_target = find_prior_swing_high(
                    highs, i, lookback=swing_lookback, strength=swing_strength
                )
                target_price = swing_high_target

                valid, skip_reason = _validate_trade(
                    "LONG", entry_price, sl, target_price, min_rr_ratio
                )
                records.append({
                    "signal_bar": i,
                    "entry_bar": i + 1,
                    "direction": "LONG",
                    "pattern": pat_name,
                    "entry_price": entry_price,
                    "stop_loss": sl,
                    "target": target_price,
                    "ema5": float(e5),
                    "ema9": float(e9),
                    "rr_ratio": _rr_ratio("LONG", entry_price, sl, target_price),
                    "valid": valid,
                    "skip_reason": skip_reason,
                })

    return pd.DataFrame(records)


# ===========================================================
# INTERNAL HELPERS
# ===========================================================

def _rr_ratio(direction: str, entry: float, sl: float, target) -> float | None:
    if target is None:
        return None
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return None
    if direction == "SHORT":
        target_dist = entry - target
    else:
        target_dist = target - entry
    if target_dist <= 0:
        return None
    return round(target_dist / sl_dist, 3)


def _validate_trade(
    direction: str,
    entry: float,
    sl: float,
    target,
    min_rr: float,
) -> tuple:
    """
    Returns (valid: bool, skip_reason: str).
    """
    if target is None:
        return False, "NO_SWING_TARGET"
    sl_dist = abs(entry - sl)
    if sl_dist == 0:
        return False, "ZERO_SL_DISTANCE"
    if direction == "SHORT":
        target_dist = entry - target
    else:
        target_dist = target - entry
    if target_dist <= 0:
        return False, "TARGET_WRONG_SIDE"
    rr = target_dist / sl_dist
    if rr < min_rr:
        return False, f"RR_BELOW_MIN ({rr:.2f} < {min_rr})"
    return True, ""
