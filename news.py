import json
import random
import time

from datetime import datetime

from config import (
    NEWS_INTERVAL_MIN,
    NEWS_INTERVAL_MAX
)


NEWS_FILE = "news_history.json"


HEADLINES = {

    "best": [
        "Major buyback announced",
        "Record quarterly profits"
    ],

    "good": [
        "Revenue beats estimates",
        "Strong demand observed"
    ],

    "neutral": [
        "No major updates",
        "Mixed market sentiment"
    ],

    "bad": [
        "Weak quarterly guidance",
        "Insider selling increases"
    ],

    "worst": [
        "Fraud allegations reported",
        "Major regulatory issue"
    ]
}


def generate_news():

    category = random.choices(

        ["neutral", "good", "bad", "best", "worst"],

        weights=[60, 15, 15, 5, 5]

    )[0]

    headline = random.choice(
        HEADLINES[category]
    )

    news_data = {

        "timestamp": str(datetime.now()),

        "category": category,

        "headline": headline
    }

    try:

        with open(NEWS_FILE, "r") as f:

            history = json.load(f)

    except:

        history = []

    history.append(news_data)

    with open(NEWS_FILE, "w") as f:

        json.dump(history, f, indent=4)

    print("\n======================")
    print(f"NEWS: {category.upper()}")
    print(headline)
    print("======================")

    return news_data


if __name__ == "__main__":

    while True:

        generate_news()

        delay = random.randint(
            NEWS_INTERVAL_MIN,
            NEWS_INTERVAL_MAX
        )

        time.sleep(delay)