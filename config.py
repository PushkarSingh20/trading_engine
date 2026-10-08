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
# HMA SETTINGS (STAGE 3)
# ==============================

HMA_FAST_LEN = 9

HMA_SLOW_LEN = 21

# ==============================================================================
# ADX THRESHOLD SETTINGS (STAGE 12 AUDIT NOTE)
# ------------------------------------------------------------------------------
# Inconsistency documented for Stage 13 parameter robustness review:
#   1. config.py:            ADX_THRESHOLD = 20 (base indicator threshold)
#   2. regime_detector.py:   ADX_TREND_THRESHOLD = 25 (regime classifier threshold)
#   3. backtest.py:          ADX_ENTRY_THRESHOLD = 50 (strict entry confirmation)
#
# Values are kept FROZEN for Stage 12 per architecture review instructions.
# They will be systematically evaluated in Stage 13 (Parameter Robustness).
# ==============================================================================
ADX_THRESHOLD = 20
ADX_ENTRY_THRESHOLD = 50

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

# ==============================
# REGIME DETECTOR
# ==============================

ADX_TREND_THRESHOLD = 25

ATR_VOL_MULTIPLIER = 1.20

HMA_SLOPE_THRESHOLD = 0.10