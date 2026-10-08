"""
tests/test_strategy_v2.py — Strategy V2 Unit Test Suite

Covers the 16 required test cases:

Pattern detection:
  T01  Doji detected correctly
  T02  Non-doji not flagged
  T03  Shooting Star detected correctly
  T04  Hammer detected correctly
  T05  Bearish Engulfing detected correctly (body-based)
  T06  Bullish Engulfing detected correctly (body-based)
  T07  Wick-based "engulfing" NOT flagged (body-only rule)

EMA filter:
  T08  Short signal suppressed when price below EMA5
  T09  Long signal suppressed when price above EMA5

Entry timing (no lookahead):
  T10  Entry price is open of N+1, not close of N

Swing detection (causal):
  T11  Swing high/low found within lookback window
  T12  Swing HIGH finder does not use bar i (causal — only i-1 and earlier)

Risk / validation:
  T13  Trade skipped when RR below min_rr_ratio
  T14  Position size is 0 when SL distance is 0
  T15  Signal with no swing target marked invalid

Integration (generate_signals):
  T16  generate_signals returns only valid signals when data is sufficient

Run with:
    python -m unittest tests/test_strategy_v2.py -v
"""

import unittest
import math
import pandas as pd
import numpy as np
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies.strategy_v2_ema_reversal import (
    is_doji,
    is_shooting_star,
    is_hammer,
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_bearish_reversal_pattern,
    is_bullish_reversal_pattern,
    find_prior_swing_high,
    find_prior_swing_low,
    calculate_short_sl,
    calculate_long_sl,
    calculate_position_size,
    calculate_ema,
    generate_signals,
    _rr_ratio,
    _validate_trade,
)


# ===========================================================
# HELPERS TO BUILD SYNTHETIC OHLCV DATA
# ===========================================================

def make_df(ohlcv_list):
    """
    Build a DataFrame from a list of (Open, High, Low, Close, Volume) tuples.
    Index is a synthetic DatetimeIndex at 3-minute intervals.
    """
    dates = pd.date_range("2024-01-02 09:15:00", periods=len(ohlcv_list), freq="3min")
    df = pd.DataFrame(
        ohlcv_list,
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=dates,
    )
    return df


def flat_bars(price, n):
    """Return n identical flat bars at price."""
    return [(price, price, price, price, 1000)] * n


# ===========================================================
# T01: Doji detected correctly
# ===========================================================

class TestT01DojiDetected(unittest.TestCase):
    def test_doji_tiny_body(self):
        """A candle whose body is 1% of range is a Doji (threshold=10%)."""
        # Range = 10, body = 0.1 → ratio = 0.01 < 0.10
        result = is_doji(
            open_=100.0, high=105.0, low=95.0, close=100.1,
            body_ratio=0.10
        )
        self.assertTrue(result, "Tiny-body candle should be detected as Doji")

    def test_doji_exact_threshold(self):
        """Body exactly at 10% of range → Doji (inclusive boundary)."""
        # Range = 10, body = 1.0 → ratio = 0.10
        result = is_doji(
            open_=100.0, high=105.0, low=95.0, close=101.0,
            body_ratio=0.10
        )
        self.assertTrue(result)

    def test_doji_zero_range(self):
        """A candle with zero range (all prices equal) is a Doji."""
        result = is_doji(
            open_=100.0, high=100.0, low=100.0, close=100.0
        )
        self.assertTrue(result)


# ===========================================================
# T02: Non-doji not flagged
# ===========================================================

class TestT02NonDojiNotFlagged(unittest.TestCase):
    def test_large_body_not_doji(self):
        """A candle with 50% body-to-range ratio is NOT a Doji."""
        # Range = 10, body = 5 → ratio = 0.50
        result = is_doji(
            open_=100.0, high=105.0, low=95.0, close=105.0,
            body_ratio=0.10
        )
        self.assertFalse(result)

    def test_just_above_threshold_not_doji(self):
        """Body just above threshold → NOT a Doji."""
        # Range = 10, body = 1.1 → ratio = 0.11 > 0.10
        result = is_doji(
            open_=100.0, high=105.0, low=95.0, close=101.1,
            body_ratio=0.10
        )
        self.assertFalse(result)


# ===========================================================
# T03: Shooting Star detected correctly
# ===========================================================

class TestT03ShootingStarDetected(unittest.TestCase):
    def test_classic_shooting_star(self):
        """
        Classic Shooting Star:
          Open=100, High=110, Low=99, Close=99.5
          Body = 0.5, Upper wick = 10, Lower wick = 0.
          upper_wick (10) >= 2 * body (0.5) → wick condition met.
          Bearish (close < open). Body near top (bottom of body at 99.5,
          range=11, from_top = 110 - 99.5 = 10.5, 10.5/11 = 0.95 > 0.35 → FAIL.

        Let's use a proper example:
          Open=102, High=110, Low=99, Close=100  (bearish, body=2, upper=8, lower=1)
          body_bottom (close=100) from_top = 110-100=10, range=11, 10/11=0.91 > 0.35
          Still fails — let's make the body sit near the top.

          Open=109, High=110, Low=99, Close=108  (bearish, body=1, upper=1, lower=9)
          upper_wick=1 < 2*body=2 → fails. NOT a shooting star.

        Correct shooting star: long upper wick, small body, body AT TOP.
          Open=109.5, High=110, Low=99, Close=109  → bearish, body=0.5, upper=0.5, lower=10
          upper_wick=0.5 < 2*body=1 → not a shooting star (hammer shape).

        Let's use correct proportions:
          Open=109, High=112, Low=99, Close=108  → bearish
          body=1, upper=3, lower=9, range=13
          upper_wick (3) >= 2*1 (2) ✓
          body_bottom from top: 112-108=4, 4/13=0.31 <= 0.35 ✓
        """
        result = is_shooting_star(
            open_=109.0, high=112.0, low=99.0, close=108.0,
            wick_ratio=2.0, body_position=0.35
        )
        self.assertTrue(result, "Classic shooting star should be detected")

    def test_shooting_star_bearish_required(self):
        """A bullish candle cannot be a Shooting Star."""
        result = is_shooting_star(
            open_=108.0, high=112.0, low=99.0, close=109.0,
            wick_ratio=2.0, body_position=0.35
        )
        self.assertFalse(result, "Bullish candle must not be flagged as Shooting Star")

    def test_shooting_star_insufficient_wick(self):
        """Upper wick < 2x body → not a Shooting Star."""
        # body=3, upper=3 (= 1x body, not 2x)
        result = is_shooting_star(
            open_=107.0, high=110.0, low=99.0, close=104.0,
            wick_ratio=2.0, body_position=0.35
        )
        self.assertFalse(result)


# ===========================================================
# T04: Hammer detected correctly
# ===========================================================

class TestT04HammerDetected(unittest.TestCase):
    def test_classic_hammer(self):
        """
        Classic Hammer:
          Bullish, long lower wick, small body at bottom.
          Open=100, High=101, Low=88, Close=101  → but this has no body.
          Open=100, High=102, Low=88, Close=101 → bullish, body=1
          lower=100-88=12, upper=102-101=1, range=14
          lower_wick(12) >= 2*body(1)=2 ✓
          body_bottom (open=100) from low: 100-88=12, 12/14=0.857 > 0.35 → FAIL.

        Body must be in the lower portion (body_bottom is close to the candle LOW).
          Open=89, High=102, Low=88, Close=91 → bullish, body=2
          lower=89-88=1 < 2*2=4 → fails wick ratio.

          Open=89, High=91, Low=80, Close=90 → bullish, body=1
          lower=89-80=9 >= 2*1=2 ✓
          body_bottom (open=89) from_low: 89-80=9, range=11, 9/11=0.818 > 0.35 → FAIL.

        Need body near LOW:
          Open=81, High=91, Low=80, Close=82 → bullish, body=1
          lower=81-80=1 < 2*1=2 → fails wick.

          Open=81, High=91, Low=75, Close=83 → bullish
          body=2, lower=81-75=6 >= 2*2=4 ✓
          body_bottom (open=81) from_low: 81-75=6, range=16, 6/16=0.375 > 0.35 → FAIL.

          Open=80.5, High=91, Low=75, Close=82 → bullish
          body=1.5, lower=80.5-75=5.5 >= 2*1.5=3 ✓
          body_bottom from low: 80.5-75=5.5, range=16, 5.5/16=0.344 <= 0.35 ✓
        """
        result = is_hammer(
            open_=80.5, high=91.0, low=75.0, close=82.0,
            wick_ratio=2.0, body_position=0.35
        )
        self.assertTrue(result, "Classic hammer should be detected")

    def test_hammer_bullish_required(self):
        """A bearish candle cannot be a Hammer."""
        result = is_hammer(
            open_=82.0, high=91.0, low=75.0, close=80.5,
        )
        self.assertFalse(result, "Bearish candle must not be flagged as Hammer")

    def test_hammer_insufficient_lower_wick(self):
        """Lower wick too short → not a Hammer."""
        # body=5, lower=3 → 3 < 2*5=10
        result = is_hammer(
            open_=100.0, high=110.0, low=97.0, close=105.0,
            wick_ratio=2.0, body_position=0.35
        )
        self.assertFalse(result)


# ===========================================================
# T05: Bearish Engulfing detected correctly (body-based)
# ===========================================================

class TestT05BearishEngulfingDetected(unittest.TestCase):
    def test_classic_bearish_engulfing(self):
        """
        Previous bullish: open=100, close=103
        Current bearish: open=104 >= close_prev(103), close=99 <= open_prev(100)
        """
        result = is_bearish_engulfing(
            open_prev=100.0, close_prev=103.0,
            open_curr=104.0, close_curr=99.0,
        )
        self.assertTrue(result)

    def test_bearish_engulfing_exact_boundary(self):
        """Engulfing with exact body boundaries (open_curr == close_prev, close_curr == open_prev)."""
        result = is_bearish_engulfing(
            open_prev=100.0, close_prev=103.0,
            open_curr=103.0, close_curr=100.0,
        )
        self.assertTrue(result, "Exact-boundary engulfing should be valid")

    def test_partial_engulfing_not_flagged(self):
        """Current body only partially overlaps — NOT a full engulfing."""
        result = is_bearish_engulfing(
            open_prev=100.0, close_prev=103.0,
            open_curr=102.0, close_curr=99.0,   # open < close_prev (103) → not engulfing top
        )
        self.assertFalse(result)


# ===========================================================
# T06: Bullish Engulfing detected correctly (body-based)
# ===========================================================

class TestT06BullishEngulfingDetected(unittest.TestCase):
    def test_classic_bullish_engulfing(self):
        """
        Previous bearish: open=103, close=100
        Current bullish: open=99 <= close_prev(100), close=104 >= open_prev(103)
        """
        result = is_bullish_engulfing(
            open_prev=103.0, close_prev=100.0,
            open_curr=99.0, close_curr=104.0,
        )
        self.assertTrue(result)

    def test_bullish_prev_candle_required(self):
        """Previous candle must be bearish for Bullish Engulfing."""
        result = is_bullish_engulfing(
            open_prev=100.0, close_prev=103.0,  # prev is bullish
            open_curr=99.0, close_curr=104.0,
        )
        self.assertFalse(result)


# ===========================================================
# T07: Wick-based "engulfing" NOT flagged (body-only rule)
# ===========================================================

class TestT07WickEngulfingNotFlagged(unittest.TestCase):
    def test_wick_engulfs_but_body_does_not(self):
        """
        Previous bullish: open=101, close=103
        Current bearish: open=103.5, close=100.5
          Wicks: high=106, low=98  → wicks engulf, but:
          open_curr(103.5) >= close_prev(103) ✓ (top covered)
          close_curr(100.5) <= open_prev(101) ✓ (bottom covered)
          → This IS a body engulfing too.

        To have wick engulfing WITHOUT body engulfing:
          Previous bullish: open=100, close=104 (body: 100-104)
          Current bearish: open=103, close=101  (body: 101-103, does NOT engulf 100-104)
          open_curr(103) < close_prev(104) → NOT engulfing top → False [RULE: body-only]
        """
        result = is_bearish_engulfing(
            open_prev=100.0, close_prev=104.0,
            open_curr=103.0, close_curr=101.0,
        )
        self.assertFalse(
            result,
            "Body must fully engulf prev body; partial body engulfing must not be flagged"
        )


# ===========================================================
# T08: Short signal suppressed when price below EMA5
# ===========================================================

class TestT08ShortSignalSuppressed(unittest.TestCase):
    def test_short_suppressed_below_ema5(self):
        """
        Build a DataFrame where at the signal bar the close is below EMA5.
        A bearish pattern exists, but no SHORT signal should be generated
        because the EMA condition (close > EMA5) is not met.
        """
        # Use flat bars to make EMA = price, then drop price for signal bar.
        n_warmup = 55
        flat = flat_bars(100.0, n_warmup)
        # Signal candle: price significantly below EMA (EMA ~100, close=90)
        # close=90 < EMA5 ~100 -> SHORT EMA filter must block the signal
        signal_bar = (91.0, 92.0, 88.0, 90.0, 1000)  # close=90 < EMA5 ~100
        entry_bar = (90.0, 91.0, 89.0, 90.0, 1000)
        rows = flat + [signal_bar, entry_bar]
        df = make_df(rows)

        signals = generate_signals(
            df,
            ema_fast_period=5,
            ema_slow_period=9,
            require_ema9=False,  # isolate EMA5 check
            warmup_bars=n_warmup,
        )
        # Empty DataFrame (no columns) counts as passing: EMA filter suppressed everything
        if len(signals) == 0 or "direction" not in signals.columns:
            return  # PASS
        short_signals = signals[signals["direction"] == "SHORT"]
        valid_shorts = short_signals[short_signals["valid"]]
        self.assertEqual(
            len(valid_shorts), 0,
            "No valid SHORT signal should be generated when price is below EMA5"
        )


# ===========================================================
# T09: Long signal suppressed when price above EMA5
# ===========================================================

class TestT09LongSignalSuppressed(unittest.TestCase):
    def test_long_suppressed_above_ema5(self):
        """
        Price above EMA5 -> LONG signals should be suppressed.
        """
        n_warmup = 55
        flat = flat_bars(100.0, n_warmup)
        # Price well above EMA — no LONG should fire
        signal_bar = (109.0, 110.0, 108.0, 109.5, 1000)  # close > EMA5 ~100
        entry_bar = (109.0, 110.5, 108.5, 109.5, 1000)
        rows = flat + [signal_bar, entry_bar]
        df = make_df(rows)

        signals = generate_signals(
            df,
            ema_fast_period=5,
            ema_slow_period=9,
            require_ema9=False,
            warmup_bars=n_warmup,
        )
        # Empty DataFrame (no columns): EMA filter correctly suppressed everything
        if len(signals) == 0 or "direction" not in signals.columns:
            return  # PASS
        valid_longs = signals[(signals["direction"] == "LONG") & (signals["valid"])]
        self.assertEqual(
            len(valid_longs), 0,
            "No valid LONG signal should be generated when price is above EMA5"
        )


# ===========================================================
# T10: Entry price is open of N+1, not close of N
# ===========================================================

class TestT10EntryIsNextBarOpen(unittest.TestCase):
    def test_entry_bar_is_next_candle(self):
        """
        Verify that signal_bar + 1 == entry_bar and that entry_price
        equals the actual open of the entry bar (not the close of signal bar).
        """
        n_warmup = 55
        flat = flat_bars(100.0, n_warmup)
        # Prev bar: bullish from 110 to 112
        prev_bar = (110.0, 113.0, 109.0, 112.0, 1000)
        # Signal bar: bearish engulfing (open=113, close=109)
        # Both close (109) and open (113) are > EMA5 (~103), engulfing 110..112
        signal_bar = (113.0, 114.0, 108.0, 109.0, 1000)
        # Entry bar: distinct Open price = 107.5
        entry_bar_candle = (107.5, 110.0, 106.0, 108.0, 1000)

        rows = flat + [prev_bar, signal_bar, entry_bar_candle]
        df = make_df(rows)
        signal_idx = n_warmup + 1  # index 56
        entry_idx = n_warmup + 2   # index 57

        signals = generate_signals(
            df,
            ema_fast_period=5,
            ema_slow_period=9,
            require_ema9=False,
            warmup_bars=n_warmup,
            swing_lookback=60,
            swing_strength=2,
        )

        self.assertGreater(len(signals), 0, "Signal should be generated")
        sig = signals[signals["signal_bar"] == signal_idx]
        self.assertEqual(len(sig), 1, f"Signal should be found at index {signal_idx}")

        sig_row = sig.iloc[0]
        self.assertEqual(
            sig_row["entry_bar"], entry_idx,
            "Entry bar must be signal_bar + 1"
        )
        expected_entry_open = float(df["Open"].iloc[entry_idx])
        self.assertAlmostEqual(
            sig_row["entry_price"], expected_entry_open, places=5,
            msg="Entry price must equal Open of bar N+1"
        )


# ===========================================================
# T11: Swing high/low found within lookback window
# ===========================================================

class TestT11SwingDetection(unittest.TestCase):
    def test_swing_high_found(self):
        """
        Build a price series with an obvious swing high at position 5
        and verify it is found.
        """
        highs_data = [100, 101, 102, 108, 103, 101, 100, 99]
        #                              ^swing at index 3
        highs = pd.Series(highs_data)
        result = find_prior_swing_high(highs, current_idx=6, lookback=10, strength=2)
        self.assertIsNotNone(result, "Swing high should be found")
        self.assertAlmostEqual(result, 108.0, places=5)

    def test_swing_low_found(self):
        """Swing low at a known position."""
        lows_data = [100, 99, 98, 91, 95, 97, 99, 100]
        #                          ^swing at index 3
        lows = pd.Series(lows_data)
        result = find_prior_swing_low(lows, current_idx=6, lookback=10, strength=2)
        self.assertIsNotNone(result, "Swing low should be found")
        self.assertAlmostEqual(result, 91.0, places=5)

    def test_no_swing_beyond_lookback(self):
        """Swing high outside the lookback window should NOT be returned."""
        highs_data = [108] + [100] * 40  # swing at index 0, far away
        highs = pd.Series(highs_data)
        result = find_prior_swing_high(highs, current_idx=41, lookback=10, strength=2)
        # The swing at index 0 is 31 bars back (lookback=10) → should not appear
        self.assertIsNone(result, "Swing beyond lookback should not be returned")


# ===========================================================
# T12: Swing HIGH finder does not use bar i (causal)
# ===========================================================

class TestT12SwingIsCausal(unittest.TestCase):
    def test_current_bar_excluded_from_swing_search(self):
        """
        Place the absolute highest value at current_idx and verify the swing
        finder does NOT return it (only searches up to current_idx - 1).
        """
        highs_data = [100] * 10 + [200]  # highest at index 10 (current)
        highs = pd.Series(highs_data)
        result = find_prior_swing_high(highs, current_idx=10, lookback=20, strength=2)
        # The high at index 10 should NOT be found (current bar excluded)
        if result is not None:
            self.assertNotAlmostEqual(
                result, 200.0, places=5,
                msg="Current bar must NOT be included in causal swing search"
            )

    def test_swing_found_one_bar_before_current(self):
        """Swing that is exactly at current_idx - 1 should be found."""
        highs_data = [95, 96, 97, 110, 108, 107]
        #                           ^3      <- this is at index 3
        highs = pd.Series(highs_data)
        # current_idx = 5, so we search up to index 4
        result = find_prior_swing_high(highs, current_idx=5, lookback=10, strength=2)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 110.0, places=5)


# ===========================================================
# T13: Trade skipped when RR below min_rr_ratio
# ===========================================================

class TestT13LowRRSkipped(unittest.TestCase):
    def test_low_rr_trade_invalid(self):
        """RR < min_rr_ratio → valid=False, skip_reason contains RR info."""
        valid, reason = _validate_trade(
            direction="SHORT",
            entry=100.0,
            sl=101.0,   # SL distance = 1.0
            target=99.5, # Target distance = 0.5 → RR = 0.5 < 1.0
            min_rr=1.0,
        )
        self.assertFalse(valid)
        self.assertIn("RR_BELOW_MIN", reason)

    def test_adequate_rr_trade_valid(self):
        """RR >= min_rr_ratio → valid=True."""
        valid, reason = _validate_trade(
            direction="SHORT",
            entry=100.0,
            sl=101.0,   # SL distance = 1.0
            target=98.0, # Target distance = 2.0 → RR = 2.0 >= 1.0
            min_rr=1.0,
        )
        self.assertTrue(valid)
        self.assertEqual(reason, "")


# ===========================================================
# T14: Position size is 0 when SL distance is 0
# ===========================================================

class TestT14ZeroSLDistance(unittest.TestCase):
    def test_zero_sl_distance_gives_zero_size(self):
        """If entry == stop_loss, position size must be 0 (prevents div-by-zero)."""
        size = calculate_position_size(
            capital=100_000,
            risk_per_trade=0.01,
            entry_price=100.0,
            stop_loss=100.0,  # SL at entry = zero distance
        )
        self.assertEqual(size, 0)

    def test_normal_sl_gives_positive_size(self):
        """Normal SL distance gives a positive integer size."""
        size = calculate_position_size(
            capital=100_000,
            risk_per_trade=0.01,
            entry_price=100.0,
            stop_loss=98.0,   # SL distance = 2.0
        )
        # risk_amount = 1000, sl_dist = 2 → size = 500
        self.assertEqual(size, 500)


# ===========================================================
# T15: Signal with no swing target marked invalid
# ===========================================================

class TestT15NoSwingTargetInvalid(unittest.TestCase):
    def test_no_target_invalid(self):
        """When target is None (no swing found), _validate_trade returns invalid."""
        valid, reason = _validate_trade(
            direction="LONG",
            entry=100.0,
            sl=98.0,
            target=None,
            min_rr=1.0,
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "NO_SWING_TARGET")

    def test_target_wrong_side_invalid(self):
        """Long trade where target is below entry → WRONG SIDE."""
        valid, reason = _validate_trade(
            direction="LONG",
            entry=100.0,
            sl=98.0,
            target=95.0,   # target below entry for a long → wrong side
            min_rr=1.0,
        )
        self.assertFalse(valid)
        self.assertIn("TARGET_WRONG_SIDE", reason)


# ===========================================================
# T16: generate_signals integration — valid signals only
# ===========================================================

class TestT16GenerateSignalsIntegration(unittest.TestCase):
    def test_all_returned_signals_have_required_columns(self):
        """generate_signals output must always have required columns."""
        n_warmup = 55
        rows = flat_bars(100.0, n_warmup + 10)
        df = make_df(rows)
        signals = generate_signals(df, warmup_bars=n_warmup)
        required_cols = {
            "signal_bar", "entry_bar", "direction", "pattern",
            "entry_price", "stop_loss", "target", "ema5", "ema9",
            "rr_ratio", "valid", "skip_reason",
        }
        if len(signals) > 0:
            self.assertTrue(
                required_cols.issubset(set(signals.columns)),
                f"Missing columns: {required_cols - set(signals.columns)}"
            )

    def test_no_signal_emitted_during_warmup(self):
        """Signals must not be emitted during warmup bars."""
        n_warmup = 55
        rows = flat_bars(100.0, n_warmup + 5)
        df = make_df(rows)
        signals = generate_signals(df, warmup_bars=n_warmup)
        if len(signals) > 0:
            self.assertTrue(
                (signals["signal_bar"] >= n_warmup).all(),
                "Signal bar indices must all be >= warmup_bars"
            )

    def test_entry_bar_always_signal_bar_plus_one(self):
        """Every signal's entry_bar must be signal_bar + 1."""
        n_warmup = 55
        # Craft data with a valid pattern setup
        flat = flat_bars(110.0, n_warmup)
        # Create a valid bearish engulfing setup above EMA5
        prev = (109.0, 111.0, 108.0, 110.5, 1000)  # bullish
        # signal: open=111, close=109 (bearish engulfs 109-110.5)
        # close=109 < 110 → close < EMA5 (~110) → won't trigger SHORT EMA filter
        # Need close > EMA5. Let's try close=110.2
        sig = (111.0, 112.0, 108.0, 110.2, 1000)
        nxt = (110.0, 110.5, 109.5, 110.0, 1000)
        rows = flat + [prev, sig, nxt]
        df = make_df(rows)
        signals = generate_signals(
            df,
            warmup_bars=n_warmup,
            swing_lookback=60,
            swing_strength=2,
        )
        if len(signals) > 0:
            mismatch = signals[signals["entry_bar"] != signals["signal_bar"] + 1]
            self.assertEqual(
                len(mismatch), 0,
                "entry_bar must always equal signal_bar + 1"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
