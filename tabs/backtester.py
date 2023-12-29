import ccxt
import pandas as pd
import plotly.express as px
import streamlit as st

def fetch_historical_prices(source, symbol, timeframe='1d', limit=20000):
    if source == 'binance':
        exchange = ccxt.binance()  # Change this if you want to use a different exchange
    else:
        exchange = ccxt.cryptocom()  # Change this if you want to use a different exchange
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=1641018154000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def calculate_returns(df):
    df['returns'] = df['close'].pct_change()
    df['weekday'] = df.index.weekday  # Monday is 0 and Sunday is 6
    df['is_weekend'] = df['weekday'].isin([5, 6])  # 5 and 6 correspond to Saturday and Sunday
    return df

def plot_returns(df, symbol):
    with st.expander(symbol):
        st.dataframe(df)
        fig = px.violin(df, x='is_weekend', y='returns', points="all", title=f'{symbol} Weekend vs Weekday Returns')
        fig.update_layout(xaxis_title='Weekend (1) vs Weekday (0)', yaxis_title='Returns')
        st.plotly_chart(fig)

def backtester_page(ch, env):
    st.header("Cryptocurrency Weekend vs Weekday Returns")
    source = st.selectbox('source:', ['binance', 'cryptocom'])
    symbols = ['SOL/USD']
    for symbol in symbols:
        st.header(f"{symbol} Returns")
        df = fetch_historical_prices(source, symbol)
        df = calculate_returns(df)
        plot_returns(df, symbol)

