
import sys
sys.path.insert(0, './driftpy/src/')

import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"

from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.accounts import *
from driftpy.constants.numeric_constants import * 
import os
import json
import streamlit as st

async def show_pid_positions(pid='', url='https://api.devnet.solana.com'):
    config = configs['devnet']

    # print(config)
    # random key 
    with open("DRFTL7fm2cA13zHTSHnTKpt58uq5E49yr2vUxuonEtYd.json", 'r') as f: secret = json.load(f) 
    kp = Keypair.from_secret_key(bytes(secret))

    wallet = Wallet(kp)
    connection = AsyncClient(url)
    provider = Provider(connection, wallet)
    ch = ClearingHouse.from_config(config, provider)

    with st.expander('pid='+str(    ch.program_id) + " config"):
        # print(str(config))
        st.text(str(config))

    state = await get_state_account(ch.program)
    market = await get_perp_market_account(ch.program, 0)


    all_users = await ch.program.account['User'].all()

    len(all_users)

    dfs = {}
    for x in all_users:
        key = str(x.public_key)
        dfs[key] = []
        for idx, pos in enumerate(x.account.perp_positions):
            dd = pos.__dict__
            dd['position_index'] = idx
            dfs[key].append(pd.Series(dd))
        dfs[key] = pd.concat(dfs[key],axis=1)                    
    perps = pd.concat(dfs,axis=1).T
    perps.index = perps.index.set_names(['public_key', 'idx2'])
    perps = perps.reset_index()
    for market_index in [0,1,2]:
        df1 = perps[((perps.base_asset_amount!=0) 
                    | (perps.quote_asset_amount!=0)) 
                    & (perps.market_index==market_index)
                ].sort_values('base_asset_amount', ascending=False).reset_index(drop=True)
        df1['base_asset_amount'] /= 1e9
        df1['lp_shares'] /= 1e9
        df1['quote_asset_amount'] /= 1e6
        df1['quote_entry_amount'] /= 1e6
        df1['quote_break_even_amount'] /= 1e6

        df1['entry_price'] = -df1['quote_entry_amount']/df1['base_asset_amount'].apply(lambda x: 1 if x==0 else x)
        df1['breakeven_price'] = -df1['quote_break_even_amount']/df1['base_asset_amount'].apply(lambda x: 1 if x==0 else x)
        df1['cost_basis'] = -df1['quote_asset_amount']/df1['base_asset_amount'].apply(lambda x: 1 if x==0 else x)
        toshow = df1[['public_key', 'open_orders', 'lp_shares', 'base_asset_amount', 'entry_price', 'breakeven_price', 'cost_basis']]
        st.text('User Perp Positions market index = '+str(market_index))
        st.table(toshow)