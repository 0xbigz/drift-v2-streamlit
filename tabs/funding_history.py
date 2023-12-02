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
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.drift_user import get_token_amount

import os
import json
import streamlit as st
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from driftpy.drift_user import get_token_amount
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import datetime

import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import InsuranceFundStakeAccount, SpotMarketAccount
from driftpy.addresses import * 
# using time module
import time
import plotly.express as px
now_ts = datetime.datetime.now().timestamp()



async def funding_history(ch: DriftClient, env):
    state = await get_state_account(ch.program)
    m1, m2, m3 = st.columns(3)
    mi = m1.selectbox('perp market index:', list(range(0, state.number_of_markets)))

    end = m3.number_input('end ts:', value=now_ts)
    start = m2.number_input('start ts:', value=now_ts-60*60*24*7)

    url = 'https://mainnet-beta.api.drift.trade/fundingRates?marketIndex='+str(mi)+'&from='+str(start)+'&'+'to='+str(end)
    ff = requests.get(url).json()['fundingRates']

    df = pd.DataFrame(ff)
    tabs = st.tabs(['rates', 'twaps', 'raw'])

    with tabs[0]:
        r1, r2, r3 = st.columns(3)
        unit = r1.radio('unit:', ['$', '%'], horizontal=True)
        hor = r2.radio('rate extrapolation:', ['hourly', 'daily', 'annual'], horizontal=True)
        iscum = r3.radio('show cumulative:', [True, False], index=1, horizontal=True)

        rate_df = df.pivot_table(index='ts', columns='marketIndex', values='fundingRate')
        rate_df = rate_df.astype(float)/1e9

        if unit == '%':
            rate_df[mi] = rate_df.values / (df['oraclePriceTwap'].astype(float).values/1e6) * 100
        if iscum:
            rate_df['cumulative'] = rate_df[mi].cumsum()

        if hor == 'daily':
            rate_df[mi] *= 24
        elif hor == 'annual':
            rate_df[mi] *= 365.25 * 24
        
        rate_df.index = pd.to_datetime((rate_df.index.astype(float) * 1e9).astype(int), utc=True)
        g1, g2 = st.columns(2)
        g2.dataframe(rate_df.sort_index(ascending=False), use_container_width=True)
        fig = rate_df.plot()
        fig.update_layout(
            title="Funding History",
            xaxis_title='Date',
            yaxis_title=f'Funding Rate ({unit})',
            )
        g1.plotly_chart(fig)

    with tabs[1]:
        r1, r2 = st.columns(2)
        unit = r1.radio('plot:', ['level', 'delta', 'delta/24'], horizontal=True)
        # hor = r2.radio('rate extrapolation:', ['hourly', 'daily', 'annual'], horizontal=True)
        rate_df = df.pivot_table(index='ts', columns='marketIndex', values=['oraclePriceTwap', 'markPriceTwap'])
        rate_df = rate_df.astype(float)/1e6
        rate_df = rate_df.swaplevel(axis=1)[mi]

        # if unit == '%':
        #     rate_df[mi] = rate_df.values / (df['oraclePriceTwap'].astype(float).values/1e6)
        # if hor == 'daily':
        #     rate_df[mi] *= 24
        # elif hor == 'annual':
            # rate_df[mi] *= 365.25
        
        rate_df.index = pd.to_datetime((rate_df.index.astype(float) * 1e9).astype(int), utc=True)
        g1, g2 = st.columns(2)
        g2.dataframe(rate_df, use_container_width=True)

        if unit == 'level':
            g1.plotly_chart(rate_df.plot())
        else:
            diff = (rate_df['markPriceTwap']-rate_df['oraclePriceTwap'])
            # diff.name = 'mark - oracle twap'

            if unit == 'delta/24':
                diff /= 24

            g1.plotly_chart(diff.plot())


    with tabs[2]:
        st.warning(url)
        st.write(df)