
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
import os
import json
import streamlit as st
from driftpy.constants.banks import devnet_banks, Bank
from driftpy.constants.markets import devnet_markets, Market
from dataclasses import dataclass
from solana.publickey import PublicKey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import time
async def show_user_stats(clearing_house: ClearingHouse):
    ch = clearing_house

    all_user_stats = await ch.program.account['UserStats'].all()
    kp = Keypair()
    ch = ClearingHouse(ch.program, kp)

    df = pd.DataFrame([x.account.__dict__ for x in all_user_stats])
    fees_df = pd.DataFrame([x.account.fees.__dict__ for x in all_user_stats])


    # print(df.columns)
    df = pd.concat([df, fees_df], axis=1)
    # print(df.columns)
    for x in df.columns:
        if x in ['taker_volume30d', 'maker_volume30d', 'filler_volume30d', 'total_fee_paid', 'total_fee_rebate', 'if_staked_quote_asset_amount']:
            df[x] /= 1e6
    
    current_ts = time.time()
    # print(current_ts)
    df['last_trade_seconds_ago'] = int(current_ts) - df[['last_taker_volume30d_ts', 'last_maker_volume30d_ts']].max(axis=1).astype(int)
    
    
    df = df[['authority', 'last_trade_seconds_ago', 'taker_volume30d', 'maker_volume30d', 
    'filler_volume30d', 'total_fee_paid', 'total_fee_rebate', 
    'number_of_sub_accounts', 'is_referrer', 'if_staked_quote_asset_amount'
    ]].sort_values('last_trade_seconds_ago').reset_index(drop=True)

    col1, col2 = st.columns(2)
    col1.metric('30D User Taker Volume', str(np.round(df['taker_volume30d'].sum()/1e6, 2))+'M')
    col2.metric('30D User Maker Volume', str(np.round(df['maker_volume30d'].sum()/1e6, 2))+'M')

    st.dataframe(df)

    dd = df.set_index('last_trade_seconds_ago')[[
        'taker_volume30d',
        # 'filler_volume30d',
     'maker_volume30d']].pipe(np.sign).cumsum().loc[:60*60*24]
    dd.index /= 3600
    active_users_past_24hrs = int(dd.values[-1].max())
    st.plotly_chart(dd.plot(title='# user active over past 24hr = '+str(active_users_past_24hrs)))
        
    