# ==============================
# MARKET SETTINGS
# ==============================

BASE_PRICE = 1000
BASE_VOLUME = 800

# ==============================
# STRATEGY SETTINGS
# ==============================

LOOKBACK_TICKS = 5

VROC_THRESHOLD = 40

BREAKOUT_BUFFER = 0.001

STOP_LOSS_PCT = 0.01

TARGET_PCT = 0.02

POSITION_SIZE = 1

COOLDOWN_SECONDS = 10

# ==============================
# LOOP SETTINGS
# ==============================

LOOP_DELAY = 3

# ==============================
# PORTFOLIO
# ==============================

INITIAL_CAPITAL = 100000

# ==============================
# NEWS SETTINGS
# ==============================

NEWS_INTERVAL_MIN = 10

NEWS_INTERVAL_MAX = 20

# ==============================
# NEWS IMPACT
# ==============================

CATEGORY_EFFECT = {

    "best": {
        "price_change": 0.03,
        "volume_multiplier": 3.0
    },

    "good": {
        "price_change": 0.02,
        "volume_multiplier": 2.0
    },

    "neutral": {
        "price_change": 0.0,
        "volume_multiplier": 1.0
    },

    "bad": {
        "price_change": -0.02,
        "volume_multiplier": 2.0
    },

    "worst": {
        "price_change": -0.03,
        "volume_multiplier": 3.0
    }
}