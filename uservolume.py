
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
from driftpy.math.oracle import *

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.clearing_house_user import get_token_amount
import os
import json
import streamlit as st
from driftpy.types import MarginRequirementType
from driftpy.constants.banks import devnet_banks, Bank
from driftpy.constants.markets import devnet_markets, Market
from driftpy.addresses import *
from dataclasses import dataclass
from solana.publickey import PublicKey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import time
from enum import Enum
from driftpy.math.margin import MarginCategory, calculate_asset_weight
import datetime

async def show_user_volume(clearing_house: ClearingHouse):
    url = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/'
    url += 'market/%s/trades/%s/%s/%s'
    mol1, molselect, mol0, mol2, _ = st.columns([3, 3, 3, 3, 10])
    market_name = mol1.selectbox('market', ['SOL-PERP', 'BTC-PERP', 'ETH-PERP', '1MBONK-PERP'])
    range_selected = molselect.selectbox('range select:', ['daily', 'weekly'], 0)

    dates = []
    lastest_date = pd.to_datetime(datetime.datetime.now(), utc=True)
    if range_selected == 'daily':
        date = mol0.date_input('select approx. date:', lastest_date, min_value=datetime.datetime(2022,11,4), max_value=lastest_date) #(datetime.datetime.now(tzInfo)))
        dates = [date]
    elif range_selected == 'weekly':
        start_date = mol0.date_input('start date:', lastest_date - datetime.timedelta(days=7), min_value=datetime.datetime(2022,11,4), max_value=lastest_date) #(datetime.datetime.now(tzInfo)))
        end_date = mol2.date_input('end date:', lastest_date, min_value=datetime.datetime(2022,11,4), max_value=lastest_date) #(datetime.datetime.now(tzInfo)))
        dates = pd.date_range(start_date, end_date)
    dfs = []
    data_urls = []
    for date in dates:
        (year, month, day) = (date.strftime("%Y"), str(date.month), str(date.day))
        data_url = url % (market_name, year, month, day)
        data_urls.append(data_url)
        dfs.append(pd.read_csv(data_url))
    dfs = pd.concat(dfs)
    dd, _ = st.columns([6,5])
    with dd.expander('data sources'):
        st.write(data_urls)
    maker_volume = dfs.groupby('maker')['quoteAssetAmountFilled'].sum()
    taker_volume = dfs.groupby('taker')['quoteAssetAmountFilled'].sum()
    df = pd.concat({'maker volume': maker_volume, 'taker volume': taker_volume},axis=1)
    df = df.sort_values('maker volume', ascending=False)
    st.dataframe(df)


    
    