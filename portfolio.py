import csv
import os


class Portfolio:

    def __init__(self, initial_capital=100000):

        self.initial_capital = initial_capital

        self.capital = initial_capital

        self.position = None

        self.entry_price = 0

        self.realized_pnl = 0

        self.total_trades = 0

        self.wins = 0

        self.losses = 0

        self.trade_log_file = "trades.csv"


    # =========================
    # ENTER TRADE
    # =========================

    def enter_trade(
        self,
        side,
        price
    ):

        if self.position is not None:

            return

        self.position = side

        self.entry_price = price

        print(f"\nENTERED {side} @ {price}")


    # =========================
    # EXIT TRADE
    # =========================

    def exit_trade(
        self,
        exit_price
    ):

        if self.position is None:

            return


        # =====================
        # CALCULATE PNL
        # =====================

        if self.position == "LONG":

            pnl = exit_price - self.entry_price

        else:

            pnl = self.entry_price - exit_price


        # =====================
        # UPDATE CAPITAL
        # =====================

        self.capital += pnl

        self.realized_pnl += pnl

        self.total_trades += 1


        # =====================
        # WIN / LOSS
        # =====================

        if pnl > 0:

            self.wins += 1

        else:

            self.losses += 1


        # =====================
        # LOG TRADE
        # =====================

        self.log_trade(
            self.position,
            self.entry_price,
            exit_price,
            pnl
        )


        # =====================
        # TERMINAL OUTPUT
        # =====================

        print(f"EXITED {self.position}")

        print(f"Entry: {self.entry_price}")

        print(f"Exit: {exit_price}")

        print(f"PnL: {round(pnl, 2)}")

        print(f"Capital: {round(self.capital, 2)}")


        # RESET POSITION

        self.position = None

        self.entry_price = 0


    # =========================
    # POSITION CHECK
    # =========================

    def has_position(self):

        return self.position is not None


    # =========================
    # UPDATE PNL
    # =========================

    def update_pnl(
        self,
        pnl
    ):

        self.capital += pnl

        self.realized_pnl += pnl

        self.total_trades += 1

        if pnl > 0:

            self.wins += 1

        else:

            self.losses += 1


    # =========================
    # WIN RATE
    # =========================

    def get_win_rate(self):

        if self.total_trades == 0:

            return 0

        return round(
            (self.wins / self.total_trades) * 100,
            2
        )


    # =========================
    # SAVE TRADE
    # =========================

    def log_trade(
        self,
        side,
        entry,
        exit_price,
        pnl
    ):

        file_exists = os.path.isfile(
            self.trade_log_file
        )

        with open(
            self.trade_log_file,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            if not file_exists:

                writer.writerow([
                    "Side",
                    "Entry",
                    "Exit",
                    "PnL"
                ])

            writer.writerow([
                side,
                round(entry, 2),
                round(exit_price, 2),
                round(pnl, 2)
            ])


    # =========================
    # FINAL STATS
    # =========================

    def get_stats(self):

        return {

            "capital": round(self.capital, 2),

            "realized_pnl": round(
                self.realized_pnl,
                2
            ),

            "total_trades": self.total_trades,

            "wins": self.wins,

            "losses": self.losses,

            "win_rate": self.get_win_rate()
        }