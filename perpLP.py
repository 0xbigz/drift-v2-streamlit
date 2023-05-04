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
from solana.rpc.types import MemcmpOpts
from driftpy.clearing_house import ClearingHouse
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.clearing_house_user import get_token_amount

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

import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import InsuranceFundStake, SpotMarket
from driftpy.addresses import * 

async def perp_lp_page(ch: ClearingHouse):
    state = await get_state_account(ch.program)
    user_lookup = st.radio('lookup users:', [True, False], index=1)
    tabs = st.tabs(['LPs'])

    if user_lookup:
        all_users = await ch.program.account['User'].all(memcmp_opts=[MemcmpOpts(offset=4350, bytes='1')])
        df = pd.DataFrame([x.account.__dict__ for x in all_users])
        df.name = df.name.apply(lambda x: bytes(x).decode('utf-8', errors='ignore'))
        df['public_key'] = [str(x.public_key) for x in all_users]
    else: 
        all_users = []
        df = pd.DataFrame()
    # with st.expander('ref accounts'):
    #     st.write(all_refs_stats)
    with tabs[0]:
        st.write('lp cooldown time:', state.lp_cooldown_time, 'seconds')


        st.write('perp market lp info:')
        a0, a1, a2, a3 = st.columns(4)
        mi = a0.selectbox('market index:', range(0, state.number_of_markets), 0)

        lps = {}
        for usr in all_users:
            for x in usr.account.perp_positions:
                if x.lp_shares != 0:
                    key = str(usr.public_key)
                    print(lps.keys(), key)
                    if key in lps.keys():
                        lps[key].append(x)
                    else:
                        lps[key] = [x]
        if len(lps.keys()):
            dff = pd.concat({key: pd.DataFrame(val) for key,val in lps.items()})
            dff.index.names = ['public_key', 'position_index']
            dff = dff.reset_index()
            dff = df[['authority', 'name', 'last_active_slot', 'public_key', 'last_add_perp_lp_shares_ts']].merge(dff, on='public_key')
            print(dff.columns)
            dff = dff[['authority', 'name', 
            'lp_shares',
            
            'last_active_slot', 'public_key',
        'last_add_perp_lp_shares_ts', 'market_index', 
        #    'position_index',
        'last_cumulative_funding_rate', 'base_asset_amount',
        'quote_asset_amount', 'quote_break_even_amount', 'quote_entry_amount',
        
        'last_base_asset_amount_per_lp', 'last_quote_asset_amount_per_lp',
        'remainder_base_asset_amount',  
        'open_orders',
        ]]
            for col in ['lp_shares', 'last_base_asset_amount_per_lp', 'base_asset_amount', 'remainder_base_asset_amount']:
                dff[col] /= 1e9
            for col in ['quote_asset_amount', 'quote_break_even_amount', 'quote_entry_amount', 'last_quote_asset_amount_per_lp']:
                dff[col] /= 1e6



            # cols = (st.multiselect('columns:', ))
            # dff = dff[cols]
            st.write('all lp positions')
            st.dataframe(dff)


        perp_market = await get_perp_market_account(ch.program, mi)
        # st.write(perp_market.amm)
        bapl = perp_market.amm.base_asset_amount_per_lp/1e9
        qapl = perp_market.amm.quote_asset_amount_per_lp/1e9
        baawul = perp_market.amm.base_asset_amount_with_unsettled_lp/1e9

        a1.metric('base asset amount per lp:', bapl)
        a2.metric('quote asset amount per lp:', qapl)
        a3.metric('unsettled base asset amount with lp:', baawul)