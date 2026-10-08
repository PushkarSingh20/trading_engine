import json
import time

import pandas as pd

from config import (
    VROC_THRESHOLD,
    LOOP_DELAY,
    HMA_FAST_LEN,
    HMA_SLOW_LEN
)

from portfolio import Portfolio

from utils import initialize_trade_log

from backtesting.indicators import (
    calculate_hma,
    calculate_hma_slope
)


# =========================
# PORTFOLIO
# =========================

portfolio = Portfolio()


# =========================
# LOAD MARKET DATA
# =========================

def load_market_data():

    try:

        with open(
            "market_data.json",
            "r"
        ) as f:

            return json.load(f)

    except:

        return []


# =========================
# VROC
# =========================

def calculate_vroc(volumes):

    if len(volumes) < 2:

        return 0

    highest = max(volumes)

    lowest = min(volumes)

    if lowest == 0:

        return 0

    return round(

        (
            (highest - lowest)
            /
            lowest
        ) * 100,

        2
    )


# =========================
# HMA SIGNAL ENGINE
# =========================

def get_hma_signal(
    prices
):

    if len(prices) < HMA_SLOW_LEN + 5:

        return None

    close_series = pd.Series(prices)

    hma_fast = calculate_hma(

        close_series,

        HMA_FAST_LEN

    )

    hma_slow = calculate_hma(

        close_series,

        HMA_SLOW_LEN

    )

    slope = calculate_hma_slope(
        hma_fast
    )

    fast_value = hma_fast.iloc[-1]

    slow_value = hma_slow.iloc[-1]

    slope_value = slope.iloc[-1]

    if pd.isna(fast_value):
        return None

    if pd.isna(slow_value):
        return None

    if pd.isna(slope_value):
        return None

    signal = None

    if (

        fast_value > slow_value

        and

        slope_value > 0

    ):

        signal = "LONG"

    elif (

        fast_value < slow_value

        and

        slope_value < 0

    ):

        signal = "SHORT"

    return {

        "signal": signal,

        "hma_fast": round(
            fast_value,
            2
        ),

        "hma_slow": round(
            slow_value,
            2
        ),

        "slope": round(
            slope_value,
            4
        )
    }


# =========================
# EXIT ENGINE
# =========================

def calculate_exit(

    position,

    current_signal

):

    if current_signal is None:

        return False

    signal = current_signal["signal"]

    if position == "LONG":

        if signal == "SHORT":

            return True

    elif position == "SHORT":

        if signal == "LONG":

            return True

    return False


# =========================
# MAIN STRATEGY
# =========================

def run_strategy():

    while True:

        market = load_market_data()

        if len(market) < 30:

            print(
                "Waiting for enough data..."
            )

            time.sleep(
                LOOP_DELAY
            )

            continue

        prices = [

            x["price"]

            for x in market

        ]

        volumes = [

            x["volume"]

            for x in market

        ]

        current_price = prices[-1]

        vroc = calculate_vroc(
            volumes
        )

        hma_data = get_hma_signal(
            prices
        )

        if hma_data is None:

            time.sleep(
                LOOP_DELAY
            )

            continue

        signal = hma_data["signal"]

        print(
            "\n========================"
        )

        print(
            f"Price     : {current_price}"
        )

        print(
            f"HMA Fast  : {hma_data['hma_fast']}"
        )

        print(
            f"HMA Slow  : {hma_data['hma_slow']}"
        )

        print(
            f"Slope     : {hma_data['slope']}"
        )

        print(
            f"VROC      : {vroc}%"
        )

        print(
            f"Signal    : {signal}"
        )

        print(
            "========================"
        )

        # =====================
        # ENTRY
        # =====================

        if (

            not portfolio.has_position()

            and

            signal is not None

            and

            vroc >= VROC_THRESHOLD

        ):

            portfolio.enter_trade(

                signal,

                current_price

            )

        # =====================
        # EXIT
        # =====================

        elif portfolio.has_position():

            should_exit = calculate_exit(

                portfolio.position,

                hma_data

            )

            if should_exit:

                portfolio.exit_trade(
                    current_price
                )

        time.sleep(
            LOOP_DELAY
        )


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    initialize_trade_log()

    run_strategy()