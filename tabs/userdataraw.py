

import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
from driftpy.accounts.oracle import *
import datetime
import requests
pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.drift_user import get_token_amount
import os
import json
import streamlit as st
from driftpy.types import MarginRequirementType, SpotPosition, PerpPosition, UserAccount, UserStatsAccount
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from driftpy.addresses import *
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import time
from enum import Enum
from driftpy.math.margin import MarginCategory, calculate_asset_weight
import plotly.graph_objs as go
from datafetch.snapshot_fetch import load_user_snapshot

@st.cache_data
def load_github_snap():
    url = "https://api.github.com/repos/0xbigz/drift-v2-flat-data/contents/data/users"
    response = requests.get(url)
    ffs = [x['download_url'] for x in response.json()]

    mega_df = [pd.read_csv(ff, index_col=[0]) for ff in ffs]
    return mega_df


async def userdataraw(clearing_house: DriftClient):
    # connection = clearing_house.program.provider.connection
    class UserAccountEncoder(json.JSONEncoder):
        def default(self, obj):
            # st.write(type(obj))
            # st.write(str(type(obj)))
            if 'Position' in str(type(obj)) or 'Order' in str(type(obj)):
                return obj.__dict__
            elif isinstance(obj, Pubkey):
                return str(obj)
            else:
                return str(obj)
            return super().default(obj)
        
    s1, s2, s3 = st.columns([2,1,2])
    inp = s1.text_input('user account:', )
    mode = s2.radio('mode:', ['live', 'snapshot'])
    commit_hash = 'main'
    if mode == 'snapshot':
        commit_hash = s3.text_input('commit hash:', 'main')
        github_snap = load_github_snap()
        ghs_df = pd.concat(github_snap, axis=1).T.reset_index(drop=True)
        for col in ['total_deposits', 'total_withdraws']:
            ghs_df[col] =  ghs_df[col].astype(float)
        st.write('ghs_df:', len(github_snap))
        tp = ghs_df.groupby('authority')[['total_deposits', 'total_withdraws']].sum()
        st.dataframe(tp.reset_index())
        tp2 = (tp['total_deposits'] - tp['total_withdraws'])/1e6
        tp2 = tp2.sort_values()

        st.dataframe(tp2.describe())
        # st.dataframe(tp)
        st.plotly_chart(tp2.plot())

    if len(inp)>5:
        st.write(inp)
        st.write(Pubkey.from_string(str(inp)))

        if mode == 'live':

            user_pk = Pubkey.from_string(str(inp))
            st.write(user_pk)
            user: UserAccount = (await clearing_house.program.account["User"].fetch(user_pk))
            st.json(json.dumps(user.__dict__, cls=UserAccountEncoder))

            user_authority = user.authority
            st.write('authority:', user_authority)

            user_stats_pk = get_user_stats_account_public_key(clearing_house.program_id, user_authority)

            userstats: UserStatsAccount = (await clearing_house.program.account["UserStats"].fetch(user_stats_pk))

            st.json(json.dumps(userstats.__dict__, cls=UserAccountEncoder))

            st.header('perp positions')
            dff = pd.concat([pd.DataFrame(pos.__dict__, index=[0]) 
                                for i,pos in enumerate(user.perp_positions)],axis=0)
            # print(dff.columns)
            dff = dff[[
                # 'authority', 'name', 
                'lp_shares',
                
            #     'last_active_slot', 'public_key',
            # 'last_add_perp_lp_shares_ts', 
            'market_index', 
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

            # st.write('perp market lp info:')
            # a0, a1, a2, a3 = st.columns(4)
            # mi = a0.selectbox('market index:', range(0, state.number_of_markets), 0)


            # st.write(perp_market.amm)
            # bapl = perp_market.amm.base_asset_amount_per_lp/1e9
            # qapl = perp_market.amm.quote_asset_amount_per_lp/1e6
            # baawul = perp_market.amm.base_asset_amount_with_unsettled_lp/1e9

            # a1.metric('base asset amount per lp:', bapl)
            # a2.metric('quote asset amount per lp:', qapl)
            # a3.metric('unsettled base asset amount with lp:', baawul)


            # cols = (st.multiselect('columns:', ))
            # dff = dff[cols]
            st.write('raw lp positions')
            st.dataframe(dff)

            await clearing_house.account_subscriber.update_cache()
            def get_wouldbe_lp_settle(row):
                def standardize_base_amount(amount, step):
                    remainder = amount % step
                    standard = amount - remainder
                    return standard, remainder

                mi = row['market_index']
                pm = clearing_house.account_subscriber.cache['perp_markets'][int(mi)].data
                delta_baapl = pm.amm.base_asset_amount_per_lp/1e9 - row['last_base_asset_amount_per_lp']
                delta_qaapl = pm.amm.quote_asset_amount_per_lp/1e6 - row['last_quote_asset_amount_per_lp']

                delta_baa = delta_baapl * row['lp_shares']
                delta_qaa = delta_qaapl * row['lp_shares']

                standard_baa, remainder_baa = standardize_base_amount(delta_baa, pm.amm.order_step_size/1e9)

                row['remainder_base_asset_amount'] += remainder_baa
                row['base_asset_amount'] += standard_baa + row['remainder_base_asset_amount']
                row['quote_asset_amount'] += delta_qaa
                row['quote_entry_amount'] += delta_qaa
                row['quote_break_even_amount'] += delta_qaa
                row['entry_price'] =  -row['quote_entry_amount']/row['base_asset_amount']

                return row



            newd = dff.apply(get_wouldbe_lp_settle, axis=1)
            st.write(newd)
                    

        else:
            user, ff = load_user_snapshot(str(inp), commit_hash)
            st.write(ff)
            dd = user.set_index(user.columns[0]).to_json()
            st.json(dd)
        # st.write(user.__dict__['spot_positions'])
        # st.json(user)




