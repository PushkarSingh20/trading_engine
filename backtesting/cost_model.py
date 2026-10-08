"""
cost_model.py
=============
Stage 14 — Transaction Cost & Execution Realism Model

Provides a fully configurable, transparent, and modular transaction-cost
and execution-friction framework for backtesting and walk-forward validation.

Components modeled:
  1. Brokerage (percentage-based turnover fee and/or fixed per order)
  2. Exchange transaction charges
  3. Statutory / regulatory fees
  4. Stamp duty (applicable to buy/entry turnover)
  5. Goods & Services Tax (GST) applied to applicable fee components
  6. Adverse execution slippage (entry and exit)
  7. Bid/Ask half-spread execution friction

IMPORTANT DISCLAIMER:
  Presets provided herein (SYNTHETIC_LOW_COST, SYNTHETIC_BASE_COST,
  SYNTHETIC_STRESS_COST) are generic synthetic scenarios for robustness
  and degradation testing. They do NOT represent verified fee schedules
  of any specific broker, exchange, or live trading venue.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class TransactionCostModel:
    name: str = "ZERO_COST"
    description: str = "Zero transaction costs and zero execution friction baseline"
    is_synthetic: bool = True

    # Brokerage fee parameters
    brokerage_pct: float = 0.0          # e.g., 0.0003 = 0.03% of turnover
    brokerage_fixed: float = 0.0        # flat fee per order / execution

    # Exchange & regulatory parameters
    exchange_fee_pct: float = 0.0       # Exchange transaction charge (% of turnover)
    regulatory_fee_pct: float = 0.0     # Statutory / SEBI / regulatory charges (% of turnover)
    stamp_duty_pct: float = 0.0         # Stamp duty (% of buy/entry turnover)
    gst_pct: float = 0.0                # Tax applied on (brokerage + exchange fee)

    # Execution friction parameters
    slippage_pct: float = 0.0           # Adverse price slippage (% of price)
    half_spread_pct: float = 0.0        # Half of bid-ask spread (% of price)

    def get_adjusted_entry_price(self, gross_price: float, position: str) -> float:
        """
        Calculates execution fill price for entry after adverse slippage and spread.
          - LONG: buys higher (gross_price * (1 + slippage + spread))
          - SHORT: sells lower (gross_price * (1 - slippage - spread))
        """
        friction = self.slippage_pct + self.half_spread_pct
        if position.upper() == "LONG":
            return gross_price * (1.0 + friction)
        elif position.upper() == "SHORT":
            return gross_price * (1.0 - friction)
        else:
            return gross_price

    def get_adjusted_exit_price(self, gross_price: float, position: str) -> float:
        """
        Calculates execution fill price for exit after adverse slippage and spread.
          - LONG: sells lower (gross_price * (1 - slippage - spread))
          - SHORT: buys higher (gross_price * (1 + slippage + spread))
        """
        friction = self.slippage_pct + self.half_spread_pct
        if position.upper() == "LONG":
            return gross_price * (1.0 - friction)
        elif position.upper() == "SHORT":
            return gross_price * (1.0 + friction)
        else:
            return gross_price

    def calculate_trade_costs(
        self,
        position: str,
        size: int,
        gross_entry: float,
        gross_exit: float,
        fill_entry: Optional[float] = None,
        fill_exit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Computes detailed breakdown of fees, friction, gross PnL, and net PnL.
        Guarantees:
          net_pnl == gross_pnl - total_costs
          total_costs == total_fees + total_execution_friction
        """
        if fill_entry is None:
            fill_entry = self.get_adjusted_entry_price(gross_entry, position)
        if fill_exit is None:
            fill_exit = self.get_adjusted_exit_price(gross_exit, position)

        entry_turnover = fill_entry * size
        exit_turnover = fill_exit * size
        total_turnover = entry_turnover + exit_turnover

        # 1. Brokerage
        entry_brokerage = (entry_turnover * self.brokerage_pct) + self.brokerage_fixed
        exit_brokerage = (exit_turnover * self.brokerage_pct) + self.brokerage_fixed
        total_brokerage = entry_brokerage + exit_brokerage

        # 2. Exchange transaction charge
        entry_exchange = entry_turnover * self.exchange_fee_pct
        exit_exchange = exit_turnover * self.exchange_fee_pct
        total_exchange = entry_exchange + exit_exchange

        # 3. Regulatory / statutory charges
        total_regulatory = total_turnover * self.regulatory_fee_pct

        # 4. Stamp duty (applicable to buy turnover)
        if position.upper() == "LONG":
            buy_turnover = entry_turnover
        else:
            buy_turnover = exit_turnover
        stamp_duty = buy_turnover * self.stamp_duty_pct

        # 5. GST (applied to brokerage + exchange fees)
        gst_base = total_brokerage + total_exchange
        gst = gst_base * self.gst_pct

        total_fees = total_brokerage + total_exchange + total_regulatory + stamp_duty + gst

        # 6. Execution friction (slippage + bid/ask spread)
        if position.upper() == "LONG":
            entry_friction_per_unit = fill_entry - gross_entry
            exit_friction_per_unit = gross_exit - fill_exit
            gross_pnl = (gross_exit - gross_entry) * size
            fill_pnl = (fill_exit - fill_entry) * size
        else:
            entry_friction_per_unit = gross_entry - fill_entry
            exit_friction_per_unit = fill_exit - gross_exit
            gross_pnl = (gross_entry - gross_exit) * size
            fill_pnl = (fill_entry - fill_exit) * size

        entry_slippage_cost = entry_friction_per_unit * size
        exit_slippage_cost = exit_friction_per_unit * size
        total_execution_friction = entry_slippage_cost + exit_slippage_cost

        # Total combined cost
        total_costs = total_fees + total_execution_friction
        net_pnl = gross_pnl - total_costs

        return {
            "position": position.upper(),
            "size": size,
            "gross_entry_price": round(gross_entry, 4),
            "fill_entry_price": round(fill_entry, 4),
            "gross_exit_price": round(gross_exit, 4),
            "fill_exit_price": round(fill_exit, 4),
            "entry_turnover": round(entry_turnover, 2),
            "exit_turnover": round(exit_turnover, 2),
            "total_turnover": round(total_turnover, 2),
            "brokerage": round(total_brokerage, 4),
            "exchange_fee": round(total_exchange, 4),
            "regulatory_fee": round(total_regulatory, 4),
            "stamp_duty": round(stamp_duty, 4),
            "gst": round(gst, 4),
            "total_fees": round(total_fees, 4),
            "entry_slippage_cost": round(entry_slippage_cost, 4),
            "exit_slippage_cost": round(exit_slippage_cost, 4),
            "total_execution_friction": round(total_execution_friction, 4),
            "total_costs": round(total_costs, 4),
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
        }


# ==============================================================================
# CANONICAL SCENARIO PRESETS (Configurable Synthetic Benchmarks)
# ==============================================================================

SCENARIOS: Dict[str, TransactionCostModel] = {
    "ZERO_COST": TransactionCostModel(
        name="ZERO_COST",
        description="Zero transaction costs, fees, slippage, and spread (Baseline)",
        is_synthetic=True,
        brokerage_pct=0.0,
        brokerage_fixed=0.0,
        exchange_fee_pct=0.0,
        regulatory_fee_pct=0.0,
        stamp_duty_pct=0.0,
        gst_pct=0.0,
        slippage_pct=0.0,
        half_spread_pct=0.0,
    ),
    "SYNTHETIC_LOW_COST": TransactionCostModel(
        name="SYNTHETIC_LOW_COST",
        description="Generic low-friction tier: 1 bps brokerage, 0.3 bps exchange, 2 bps slip, 1 bps spread",
        is_synthetic=True,
        brokerage_pct=0.0001,       # 0.01%
        brokerage_fixed=0.0,
        exchange_fee_pct=0.00003,   # 0.003%
        regulatory_fee_pct=0.00001, # 0.001%
        stamp_duty_pct=0.00003,     # 0.003%
        gst_pct=0.18,               # 18% on fees
        slippage_pct=0.0002,        # 0.02% (2 bps)
        half_spread_pct=0.0001,     # 0.01% (1 bps)
    ),
    "SYNTHETIC_BASE_COST": TransactionCostModel(
        name="SYNTHETIC_BASE_COST",
        description="Generic standard tier: 3 bps brokerage, 0.35 bps exchange, 5 bps slip, 2 bps spread",
        is_synthetic=True,
        brokerage_pct=0.0003,       # 0.03%
        brokerage_fixed=0.0,
        exchange_fee_pct=0.000035,  # 0.0035%
        regulatory_fee_pct=0.00002, # 0.002%
        stamp_duty_pct=0.00005,     # 0.005%
        gst_pct=0.18,               # 18% on fees
        slippage_pct=0.0005,        # 0.05% (5 bps)
        half_spread_pct=0.0002,     # 0.02% (2 bps)
    ),
    "SYNTHETIC_STRESS_COST": TransactionCostModel(
        name="SYNTHETIC_STRESS_COST",
        description="Generic stress tier: 5 bps brokerage, 0.5 bps exchange, 15 bps slip, 5 bps spread",
        is_synthetic=True,
        brokerage_pct=0.0005,       # 0.05%
        brokerage_fixed=0.0,
        exchange_fee_pct=0.00005,   # 0.005%
        regulatory_fee_pct=0.00003, # 0.003%
        stamp_duty_pct=0.00010,     # 0.010%
        gst_pct=0.18,               # 18% on fees
        slippage_pct=0.0015,        # 0.15% (15 bps)
        half_spread_pct=0.0005,     # 0.05% (5 bps)
    ),
}


def get_cost_model(scenario_name: str = "ZERO_COST") -> TransactionCostModel:
    """
    Retrieves a TransactionCostModel by scenario name.
    Falls back to ZERO_COST if not recognized.
    """
    key = scenario_name.upper()
    if key in SCENARIOS:
        return SCENARIOS[key]
    # Check partial match
    for k, v in SCENARIOS.items():
        if key in k:
            return v
    raise ValueError(f"Unknown cost scenario: '{scenario_name}'. Available: {list(SCENARIOS.keys())}")
