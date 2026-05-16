import json
import pandas as pd
import streamlit as st

from config import INITIAL_CAPITAL


st.set_page_config(
    page_title="VROC Dashboard",
    layout="wide"
)


def load_market_data():

    try:

        with open("market_data.json", "r") as f:

            return json.load(f)

    except:

        return []


def load_trade_data():

    try:

        return pd.read_csv("trades.csv")

    except:

        return pd.DataFrame()


def calculate_stats(df):

    if len(df) == 0:

        return {

            "capital": INITIAL_CAPITAL,

            "pnl": 0,

            "trades": 0,

            "win_rate": 0
        }

    pnl = df["pnl"].sum()

    wins = len(df[df["pnl"] > 0])

    trades = len(df)

    win_rate = (wins / trades) * 100

    return {

        "capital": round(
            INITIAL_CAPITAL + pnl,
            2
        ),

        "pnl": round(pnl, 2),

        "trades": trades,

        "win_rate": round(win_rate, 2)
    }


market_data = load_market_data()

trade_df = load_trade_data()

stats = calculate_stats(trade_df)


st.title("📈 VROC Trading Dashboard")

st.markdown("---")


if len(market_data) > 0:

    latest = market_data[-1]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Current Price",
            latest["price"]
        )

    with col2:

        st.metric(
            "Current Volume",
            latest["volume"]
        )

    market_df = pd.DataFrame(market_data)

    st.line_chart(
        market_df["price"]
    )

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Capital",
        stats["capital"]
    )

with col2:

    st.metric(
        "PnL",
        stats["pnl"]
    )

with col3:

    st.metric(
        "Trades",
        stats["trades"]
    )

with col4:

    st.metric(
        "Win Rate %",
        stats["win_rate"]
    )

st.markdown("---")

st.subheader("Trade History")

if len(trade_df) > 0:

    st.dataframe(
        trade_df,
        use_container_width=True
    )

    trade_df["cumulative_pnl"] = (
        trade_df["pnl"].cumsum()
    )

    st.subheader("PnL Curve")

    st.line_chart(
        trade_df["cumulative_pnl"]
    )

else:

    st.info("No trades yet")