
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
import os
import json
import streamlit as st
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import requests

async def show_platyperps(clearing_house: DriftClient):
    binance = requests.get('https://www.binance.com/fapi/v1/premiumIndex?symbol=SOLUSDT').json()
    df = pd.DataFrame(binance, index=['Binance'])
    st.dataframe(df)

    dydx2 = pd.DataFrame(requests.get('https://api.dydx.exchange/v3/markets').json()['markets']).T
    dydx2 = dydx2.loc[['SOL-USD']]
    dydx2.index = ['DYDX']

    dydx = requests.get('https://api.dydx.exchange/v3/candles/SOL-USD?resolution=1DAY').json()
    df = pd.DataFrame(dydx['candles']).loc[0:0]
    df.index=['DYDX']
    df = dydx2.merge(df)
    df.index = ['DYDX']
    # df = pd.concat([df, dydx2],axis=1)
    st.dataframe(df)


    okx = requests.get('https://www.okx.com/priapi/v5/rubik/web/public/funding-rate-arbitrage?ctType=linear&ccyType=USDT&arbitrageType=futures_spot&countryFilter=0').json()['data']
    df = pd.DataFrame(okx)
    df = df[df.ccy=='SOL']
    df.index = ['OKX']
    st.dataframe(df)
    