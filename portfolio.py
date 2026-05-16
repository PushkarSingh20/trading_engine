import time

from config import (
    INITIAL_CAPITAL,
    COOLDOWN_SECONDS
)

from utils import log_trade


class Portfolio:

    def __init__(self):

        self.initial_capital = INITIAL_CAPITAL

        self.capital = INITIAL_CAPITAL

        self.realized_pnl = 0

        self.total_trades = 0

        self.position = None

        self.entry_price = 0

        self.entry_time = None

        self.cooldown = 0

    def has_position(self):

        return self.position is not None

    def in_cooldown(self):

        return time.time() < self.cooldown

    def enter_trade(self, side, price):

        if self.has_position():

            return False

        if self.in_cooldown():

            return False

        self.position = side

        self.entry_price = price

        self.entry_time = time.time()

        print("\n==========================")
        print(f"{side} ENTRY @ {price}")
        print("==========================")

        return True

    def exit_trade(self, exit_price):

        if not self.has_position():

            return

        if self.position == "LONG":

            pnl = exit_price - self.entry_price

        else:

            pnl = self.entry_price - exit_price

        self.realized_pnl += pnl

        self.capital += pnl

        self.total_trades += 1

        print("\n==========================")
        print(f"EXITED {self.position}")
        print(f"ENTRY: {self.entry_price}")
        print(f"EXIT: {exit_price}")
        print(f"PnL: {round(pnl, 2)}")
        print(f"Capital: {round(self.capital, 2)}")
        print("==========================")

        log_trade(
            self.position,
            self.entry_price,
            exit_price,
            pnl
        )

        self.position = None

        self.entry_price = 0

        self.entry_time = None

        self.cooldown = time.time() + COOLDOWN_SECONDS