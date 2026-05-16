import json
import random
import time

from config import (
    BASE_PRICE,
    BASE_VOLUME,
    CATEGORY_EFFECT,
    LOOP_DELAY
)


MARKET_FILE = "market_data.json"

price = BASE_PRICE

volume = BASE_VOLUME


def load_latest_news():

    try:

        with open("news_history.json", "r") as f:

            news = json.load(f)

            if len(news) == 0:
                return None

            return news[-1]

    except:

        return None


def generate_market_tick():

    global price
    global volume

    latest_news = load_latest_news()

    drift = random.uniform(-2, 2)

    if latest_news:

        category = latest_news["category"]

        effect = CATEGORY_EFFECT[category]

        price_change = effect["price_change"]

        volume_multiplier = effect["volume_multiplier"]

        price += price * price_change

        volume = int(
            (BASE_VOLUME * volume_multiplier)
            + random.randint(-200, 200)
        )

    else:

        volume += random.randint(-100, 100)

    price += drift

    volume = max(volume, 100)

    tick = {

        "price": round(price, 2),

        "volume": volume
    }

    try:

        with open(MARKET_FILE, "r") as f:

            history = json.load(f)

    except:

        history = []

    history.append(tick)

    history = history[-100:]

    with open(MARKET_FILE, "w") as f:

        json.dump(history, f, indent=4)

    print(f"Price: {tick['price']} | Volume: {tick['volume']}")

    return tick


if __name__ == "__main__":

    while True:

        generate_market_tick()

        time.sleep(LOOP_DELAY)