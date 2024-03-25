import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
import copy
import plotly.express as px
pd.options.plotting.backend = "plotly"
from datafetch.transaction_fetch import load_token_balance
# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet, AccountClient
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import MemcmpOpts
from driftpy.drift_client import DriftClient
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.drift_user import DriftUser, get_token_amount
from datafetch.transaction_fetch import transaction_history_for_account, load_token_balance

from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.constants.spot_markets import mainnet_spot_market_configs

import os
import json
import streamlit as st
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market, all_user_stats, DRIFT_WHALE_LIST_SNAP
from anchorpy import EventParser
import asyncio
from driftpy.math.margin import MarginCategory
import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import InsuranceFundStakeAccount, SpotMarketAccount
from driftpy.addresses import * 
import time
from driftpy.user_map.user_map import UserMap, UserMapConfig, PollingConfig
import datetime
import csv


async def users_in_market_page(drift_client: DriftClient, env):
    user_map_result = UserMap(UserMapConfig(drift_client, PollingConfig(0), 
                                 skip_initial_load=False, include_idle=False))
    
    await drift_client.account_subscriber.update_cache()
    
    await user_map_result.subscribe()
    user_keys = list(user_map_result.user_map.keys())
    user_vals = list(user_map_result.values())

    perp_config = mainnet_perp_market_configs if env != 'devnet' else devnet_perp_market_configs
    spot_config = mainnet_spot_market_configs if env != 'devnet' else devnet_spot_market_configs

    num_perps = len(perp_config)
    num_spots = len(spot_config)

    res = {i: [] for i in range(num_perps)}
    res_spot = {i: [] for i in range(num_spots)}

    for k,u in zip(user_keys,user_vals):
        user: DriftUser = u
        for i in range(num_perps):
            pos = user.get_perp_position_with_lp_settle(i)[0]
            if pos.base_asset_amount != 0 or pos.remainder_base_asset_amount != 0 or pos.open_orders != 0 or pos.quote_asset_amount != 0:
                ttt = (pos.base_asset_amount + pos.remainder_base_asset_amount)/1e9
                f_upnl = user.get_unrealized_funding_pnl(i)/1e6
                upnl = user.get_unrealized_pnl(False, i)/1e6

                be_price = np.nan
                if ttt != 0:
                    be_price = (-pos.quote_break_even_amount/1e6)/(ttt)
                res[i].append((k, ttt, be_price, f_upnl, upnl))
        
        for i in range(num_spots):
            pos = user.get_token_amount(i)
            if pos != 0:
                res_spot[i].append((k, pos))

    tab_cat = st.tabs([f'spot ({num_spots})', f'perp ({num_perps})'])

    with tab_cat[0]:
        tabs = st.tabs([spot_config[key].symbol for key in range(num_spots)])
        
        for key,val in res_spot.items():
            sm_i = drift_client.get_spot_market_account(key)
            tabs[key].write(f'{spot_config[key].symbol} (market index = {key})')
            df = pd.DataFrame(val, columns=['userAccount', 'token']).sort_values('token', ascending=True)
            df['token'] /= (10 ** sm_i.decimals)
            tabs[key].write(df)
            
            dust_thres = .001
            dust_dep_df = df[((df['token']<dust_thres) & (df['token']>=0))]
            dust_bor_df = df[((df['token']>-dust_thres) & (df['token']<=0))]

            tabs[key].write(f'dust threshold = {dust_thres} tokens')
            tabs[key].metric('total dust deposits', 
                             f'{dust_dep_df["token"].sum():,.4f}', 
                             f'{len(dust_dep_df)} users')
            tabs[key].metric('total dust borrows', 
                             f'{dust_bor_df["token"].sum():,.4f}', 
                             f'{len(dust_bor_df)} users')

    with tab_cat[1]:
        s1, s2, s3 = st.columns(3)
        tabs = st.tabs([perp_config[key].base_asset_symbol for key in range(num_perps)])
        usdc_market = drift_client.get_spot_market_account(0)

        total_remaining_upnl = 0
        total_funding_upnl = 0
        for key,val in res.items():
            pm_i = drift_client.get_perp_market_account(key)

            df = pd.DataFrame(val, columns=['userAccount', 'base amount', 'be_price', 'funding_upnl', 'remaining_upnl']).sort_values('base amount', ascending=True)
            user_with_base = len(df[df['base amount']!=0])
            user_in_market = len(df)
            tabs[key].write(f'{user_with_base}/{user_in_market} users in {perp_config[key].base_asset_symbol} (market index = {key})')
            tabs[key].write(df)
            funding_upnl_i = df['funding_upnl'].sum()
            total_funding_upnl += funding_upnl_i
            remaining_upnl_i = df['remaining_upnl'].sum()
            total_remaining_upnl += remaining_upnl_i

            market_summary_cols = tabs[key].columns(3)
            pnl_pool_b = pm_i.pnl_pool.scaled_balance * usdc_market.cumulative_deposit_interest/1e10/(1e9)
            fee_pool_b = pm_i.amm.fee_pool.scaled_balance * usdc_market.cumulative_deposit_interest/1e10/(1e9)

            market_summary_cols[0].metric('pnl pool', f'${pnl_pool_b:,.2f}')
            market_summary_cols[0].metric('fee pool', f'${fee_pool_b:,.2f}')

            market_summary_cols[1].metric('funding pnl', f'${funding_upnl_i:,.2f}')
            market_summary_cols[2].metric('remaining pnl', f'${remaining_upnl_i:,.2f}')

        s2.metric('total remaining upnl', f'${total_remaining_upnl:,.2f}')        
        s3.metric('total funding upnl', f'${total_funding_upnl:,.2f}')
