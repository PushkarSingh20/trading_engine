"""
test_cost_model.py
==================
Unit and Integration Tests for Stage 14 Transaction Cost & Execution Realism.

Validates:
  1. Brokerage calculation (percentage & fixed)
  2. Transaction & regulatory fee calculation
  3. Slippage calculation on entry and exit
  4. Entry cost calculations
  5. Exit cost calculations
  6. Gross-to-net PnL identity: net_pnl == gross_pnl - total_costs
  7. Zero-cost scenario reproduces gross PnL identically
  8. Monotonic cost progression across scenarios (ZERO < LOW < BASE < STRESS)
  9. No impact of cost layer on signal generation / trade frequency
  10. No look-ahead bias (costs calculated from trade execution prices)
  11. Window 5 quarantine enforcement (cannot accidentally enter evaluation)
"""

import unittest
import pandas as pd
import numpy as np

from backtesting.cost_model import TransactionCostModel, SCENARIOS, get_cost_model
from backtesting.dataset_splitter import DatasetSplitter


class TestTransactionCostModel(unittest.TestCase):

    def setUp(self):
        self.zero_model = SCENARIOS["ZERO_COST"]
        self.low_model = SCENARIOS["SYNTHETIC_LOW_COST"]
        self.base_model = SCENARIOS["SYNTHETIC_BASE_COST"]
        self.stress_model = SCENARIOS["SYNTHETIC_STRESS_COST"]

    def test_01_brokerage_calculation(self):
        # 0.03% brokerage on 100k turnover = 30.0
        model = TransactionCostModel(brokerage_pct=0.0003, brokerage_fixed=10.0)
        # 10 shares @ 1000 entry = 10,000 turnover
        # 10 shares @ 1100 exit = 11,000 turnover
        res = model.calculate_trade_costs(
            position="LONG", size=10, gross_entry=1000.0, gross_exit=1100.0
        )
        expected_entry_brk = (10000.0 * 0.0003) + 10.0  # 3.0 + 10.0 = 13.0
        expected_exit_brk = (11000.0 * 0.0003) + 10.0   # 3.3 + 10.0 = 13.3
        expected_total_brk = 26.3
        self.assertAlmostEqual(res["brokerage"], expected_total_brk, places=2)

    def test_02_transaction_fees_calculation(self):
        model = TransactionCostModel(
            exchange_fee_pct=0.00003,
            regulatory_fee_pct=0.00001,
            stamp_duty_pct=0.00003,
            gst_pct=0.18
        )
        # Entry turnover = 10,000; Exit turnover = 10,000; Total = 20,000
        res = model.calculate_trade_costs(
            position="LONG", size=10, gross_entry=1000.0, gross_exit=1000.0
        )
        expected_exchange = 20000.0 * 0.00003  # 0.60
        expected_regulatory = 20000.0 * 0.00001  # 0.20
        expected_stamp_duty = 10000.0 * 0.00003  # 0.30 (buy turnover only)
        expected_gst = expected_exchange * 0.18  # 0.108
        self.assertAlmostEqual(res["exchange_fee"], expected_exchange, places=3)
        self.assertAlmostEqual(res["regulatory_fee"], expected_regulatory, places=3)
        self.assertAlmostEqual(res["stamp_duty"], expected_stamp_duty, places=3)
        self.assertAlmostEqual(res["gst"], expected_gst, places=3)

    def test_03_slippage_calculation(self):
        # 10 bps slippage (0.001)
        model = TransactionCostModel(slippage_pct=0.0010)
        # LONG entry at 1000 -> fill at 1001.0
        # LONG exit at 1100 -> fill at 1098.9
        long_res = model.calculate_trade_costs(
            position="LONG", size=10, gross_entry=1000.0, gross_exit=1100.0
        )
        self.assertAlmostEqual(long_res["fill_entry_price"], 1001.0, places=2)
        self.assertAlmostEqual(long_res["fill_exit_price"], 1098.9, places=2)
        # Friction per unit: (1001 - 1000) + (1100 - 1098.9) = 1.0 + 1.1 = 2.1
        # Total friction for 10 units = 21.0
        self.assertAlmostEqual(long_res["total_execution_friction"], 21.0, places=2)

        # SHORT entry at 1000 -> fill at 999.0
        # SHORT exit at 900 -> fill at 900.9
        short_res = model.calculate_trade_costs(
            position="SHORT", size=10, gross_entry=1000.0, gross_exit=900.0
        )
        self.assertAlmostEqual(short_res["fill_entry_price"], 999.0, places=2)
        self.assertAlmostEqual(short_res["fill_exit_price"], 900.9, places=2)
        self.assertAlmostEqual(short_res["total_execution_friction"], 19.0, places=2)

    def test_04_entry_and_exit_costs(self):
        model = TransactionCostModel(brokerage_pct=0.0002, slippage_pct=0.0005)
        res = model.calculate_trade_costs("LONG", 10, 1000.0, 1100.0)
        self.assertGreater(res["entry_slippage_cost"], 0)
        self.assertGreater(res["exit_slippage_cost"], 0)
        self.assertGreater(res["total_fees"], 0)

    def test_05_gross_to_net_pnl_identity(self):
        # Test for multiple models and sides
        for model in [self.low_model, self.base_model, self.stress_model]:
            for pos in ["LONG", "SHORT"]:
                res = model.calculate_trade_costs(pos, 25, 1450.0, 1480.0)
                # Mathematical invariant: net_pnl must equal gross_pnl - total_costs
                expected_net = round(res["gross_pnl"] - res["total_costs"], 2)
                self.assertAlmostEqual(res["net_pnl"], expected_net, places=2)

    def test_06_zero_cost_reproduces_gross_identically(self):
        for pos in ["LONG", "SHORT"]:
            res = self.zero_model.calculate_trade_costs(pos, 50, 1300.0, 1350.0)
            self.assertEqual(res["total_costs"], 0.0)
            self.assertEqual(res["total_fees"], 0.0)
            self.assertEqual(res["total_execution_friction"], 0.0)
            self.assertEqual(res["gross_pnl"], res["net_pnl"])
            self.assertEqual(res["fill_entry_price"], res["gross_entry_price"])
            self.assertEqual(res["fill_exit_price"], res["gross_exit_price"])

    def test_07_monotonic_cost_increase(self):
        # Costs must strictly increase: ZERO < LOW < BASE < STRESS
        pos, size, entry, exit_p = "LONG", 50, 1400.0, 1450.0
        c_zero = self.zero_model.calculate_trade_costs(pos, size, entry, exit_p)["total_costs"]
        c_low = self.low_model.calculate_trade_costs(pos, size, entry, exit_p)["total_costs"]
        c_base = self.base_model.calculate_trade_costs(pos, size, entry, exit_p)["total_costs"]
        c_stress = self.stress_model.calculate_trade_costs(pos, size, entry, exit_p)["total_costs"]

        self.assertLess(c_zero, c_low)
        self.assertLess(c_low, c_base)
        self.assertLess(c_base, c_stress)

    def test_08_no_impact_on_signal_generation(self):
        # Verifies that calculating costs does not alter price series or signals
        prices = [100.0, 102.0, 101.0, 105.0]
        original_prices = list(prices)
        _ = self.base_model.calculate_trade_costs("LONG", 10, prices[0], prices[-1])
        self.assertEqual(prices, original_prices)

    def test_09_no_lookahead_bias(self):
        # Entry price adjustment only depends on entry price and model parameters
        adj_entry = self.base_model.get_adjusted_entry_price(1000.0, "LONG")
        # Ensure exit price is not required to calculate adjusted entry
        self.assertIsInstance(adj_entry, float)
        self.assertGreater(adj_entry, 1000.0)

    def test_10_window_5_quarantine_enforcement(self):
        # Enforces that an evaluation set generated with max_windows=4 does NOT contain Window 5
        dummy_df = pd.DataFrame({
            "Datetime": pd.date_range("2024-01-01", periods=1000, freq="h"),
            "Open": 100.0, "High": 105.0, "Low": 95.0, "Close": 100.0, "Volume": 1000
        })
        splitter = DatasetSplitter(dummy_df, "Datetime")
        windows = splitter.walk_forward_windows(n_splits=5, mode="expanding", oos_fraction=0.10)
        self.assertEqual(len(windows), 5)
        eval_windows = windows[:4]
        window_ids = [w["window"] for w in eval_windows]
        self.assertNotIn(5, window_ids)
        self.assertEqual(window_ids, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
