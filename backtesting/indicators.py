import pandas as pd
import numpy as np


# ==================================
# ATR
# ==================================

def calculate_atr(
    df,
    period=14
):

    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"]
        - df["Close"].shift(1)
    ).abs()

    low_close = (
        df["Low"]
        - df["Close"].shift(1)
    ).abs()

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(period).mean()

    return atr


# ==================================
# VROC
# ==================================

def calculate_vroc_series(
    volume,
    period=14
):

    vroc = (
        (
            volume
            - volume.shift(period)
        )
        /
        volume.shift(period)
    ) * 100

    return vroc


# ==================================
# +DM
# ==================================

def calculate_plus_dm(df):

    up_move = (
        df["High"]
        - df["High"].shift(1)
    )

    down_move = (
        df["Low"].shift(1)
        - df["Low"]
    )

    plus_dm = np.where(

        (up_move > down_move)
        &
        (up_move > 0),

        up_move,

        0
    )

    return pd.Series(
        plus_dm,
        index=df.index
    )


# ==================================
# -DM
# ==================================

def calculate_minus_dm(df):

    up_move = (
        df["High"]
        - df["High"].shift(1)
    )

    down_move = (
        df["Low"].shift(1)
        - df["Low"]
    )

    minus_dm = np.where(

        (down_move > up_move)
        &
        (down_move > 0),

        down_move,

        0
    )

    return pd.Series(
        minus_dm,
        index=df.index
    )


# ==================================
# ADX
# ==================================

def calculate_adx(
    df,
    period=14
):

    atr = calculate_atr(
        df,
        period
    )

    plus_dm = calculate_plus_dm(df)

    minus_dm = calculate_minus_dm(df)

    plus_di = (

        plus_dm
        .rolling(period)
        .mean()

        / atr

    ) * 100

    minus_di = (

        minus_dm
        .rolling(period)
        .mean()

        / atr

    ) * 100

    dx = (

        (
            (plus_di - minus_di).abs()
        )
        /
        (
            plus_di + minus_di
        )

    ) * 100

    adx = dx.rolling(period).mean()

    return adx


# ==================================
# WMA
# ==================================

def calculate_wma(
    series,
    period
):

    weights = np.arange(
        1,
        period + 1
    )

    return series.rolling(period).apply(

        lambda prices:

        np.dot(
            prices,
            weights
        ) / weights.sum(),

        raw=True
    )


# ==================================
# HMA
# ==================================

def calculate_hma(
    close,
    period
):

    half_period = int(period / 2)

    sqrt_period = int(np.sqrt(period))

    wma_half = calculate_wma(
        close,
        half_period
    )

    wma_full = calculate_wma(
        close,
        period
    )

    hull_series = (

        2 * wma_half

        - wma_full

    )

    hma = calculate_wma(

        hull_series,

        sqrt_period

    )

    return hma


# ==================================
# HMA SLOPE
# ==================================

def calculate_hma_slope(

    hma_series

):

    return hma_series.diff()


# ==================================
# VWAP (Stage 4)
# ==================================

def calculate_vwap(df):

    typical_price = (

        df["High"]
        +
        df["Low"]
        +
        df["Close"]

    ) / 3

    cumulative_tp_volume = (

        typical_price
        * df["Volume"]

    ).cumsum()

    cumulative_volume = (

        df["Volume"]

    ).cumsum()

    vwap = (

        cumulative_tp_volume
        /
        cumulative_volume

    )

    return vwap