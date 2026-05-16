import json
import time

from config import (
    LOOKBACK_TICKS,
    VROC_THRESHOLD,
    STOP_LOSS_PCT,
    TARGET_PCT,
    LOOP_DELAY,
    BREAKOUT_BUFFER
)

from portfolio import Portfolio

from utils import initialize_trade_log


portfolio = Portfolio()


def load_market_data():

    try:

        with open("market_data.json", "r") as f:

            return json.load(f)

    except:

        return []


def calculate_vroc(volumes):

    highest = max(volumes)

    lowest = min(volumes)

    return round(
        ((highest - lowest) / lowest) * 100,
        2
    )


def get_price_range(prices):

    return max(prices), min(prices)


def check_entry(current_price, high, low):

    breakout_above = high * (1 + BREAKOUT_BUFFER)

    breakout_below = low * (1 - BREAKOUT_BUFFER)

    if current_price > breakout_above:

        return "LONG"

    elif current_price < breakout_below:

        return "SHORT"

    return None


def run_strategy():

    while True:

        market = load_market_data()

        if len(market) < LOOKBACK_TICKS + 1:

            print("Waiting for enough data...")

            time.sleep(LOOP_DELAY)

            continue

        recent_ticks = market[-(LOOKBACK_TICKS + 1):]

        range_ticks = recent_ticks[:-1]

        current_tick = recent_ticks[-1]

        prices = [
            x["price"]
            for x in range_ticks
        ]

        volumes = [
            x["volume"]
            for x in range_ticks
        ]

        current_price = current_tick["price"]

        vroc = calculate_vroc(volumes)

        print("\n======================")
        print(f"Price: {current_price}")
        print(f"VROC: {vroc}%")
        print("======================")

        if vroc < VROC_THRESHOLD:

            print("VROC below threshold")

            time.sleep(LOOP_DELAY)

            continue

        high_range, low_range = get_price_range(prices)

        print(f"High Range: {high_range}")
        print(f"Low Range: {low_range}")

        signal = check_entry(
            current_price,
            high_range,
            low_range
        )

        if signal:

            portfolio.enter_trade(
                signal,
                current_price
            )

        if portfolio.has_position():

            entry = portfolio.entry_price

            side = portfolio.position

            if side == "LONG":

                sl = entry * (1 - STOP_LOSS_PCT)

                target = entry * (1 + TARGET_PCT)

                if current_price <= sl:

                    print("LONG STOP LOSS")

                    portfolio.exit_trade(current_price)

                elif current_price >= target:

                    print("LONG TARGET HIT")

                    portfolio.exit_trade(current_price)

            elif side == "SHORT":

                sl = entry * (1 + STOP_LOSS_PCT)

                target = entry * (1 - TARGET_PCT)

                if current_price >= sl:

                    print("SHORT STOP LOSS")

                    portfolio.exit_trade(current_price)

                elif current_price <= target:

                    print("SHORT TARGET HIT")

                    portfolio.exit_trade(current_price)

        time.sleep(LOOP_DELAY)


if __name__ == "__main__":

    initialize_trade_log()

    run_strategy()