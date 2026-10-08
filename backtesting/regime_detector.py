import pandas as pd


# ==================================
# REGIME SETTINGS
# ==================================

ADX_TREND_THRESHOLD = 25

ATR_VOL_MULTIPLIER = 1.20

HMA_SLOPE_THRESHOLD = 0.10


# ==================================
# DETECT MARKET REGIME
# ==================================

def detect_market_regime(

    adx,

    atr,

    avg_atr,

    hma_slope

):

    # ==========================
    # TREND MARKET
    # ==========================

    if (

        adx > ADX_TREND_THRESHOLD

        and

        abs(hma_slope)
        > HMA_SLOPE_THRESHOLD

    ):

        return "TREND"

    # ==========================
    # VOLATILE MARKET
    # ==========================

    if (

        atr

        >

        avg_atr
        * ATR_VOL_MULTIPLIER

    ):

        return "VOLATILE"

    # ==========================
    # SIDEWAYS MARKET
    # ==========================

    return "SIDEWAYS"