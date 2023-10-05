

import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
from driftpy.math.oracle import *
import datetime
import requests
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
from driftpy.types import MarginRequirementType, SpotPosition, PerpPosition
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
import plotly.graph_objs as go
from datafetch.snapshot_fetch import load_user_snapshot

@st.cache_data
def load_github_snap():
    url = "https://api.github.com/repos/0xbigz/drift-v2-flat-data/contents/data/users"
    response = requests.get(url)
    ffs = [x['download_url'] for x in response.json()]

    mega_df = [pd.read_csv(ff, index_col=[0]) for ff in ffs]
    return mega_df


async def userdataraw(clearing_house: ClearingHouse):
    # connection = clearing_house.program.provider.connection
    class UserAccountEncoder(json.JSONEncoder):
        def default(self, obj):
            # st.write(type(obj))
            # st.write(str(type(obj)))
            if 'Position' in str(type(obj)) or 'Order' in str(type(obj)):
                return obj.__dict__
            elif isinstance(obj, PublicKey):
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
        st.write(PublicKey(str(inp)))

        if mode == 'live':
            user = (await clearing_house.program.account["User"].fetch(PublicKey(str(inp))))
            st.json(json.dumps(user.__dict__, cls=UserAccountEncoder))
        else:
            user, ff = load_user_snapshot(str(inp), commit_hash)
            st.write(ff)
            dd = user.set_index(user.columns[0]).to_json()
            st.json(dd)
        # st.write(user.__dict__['spot_positions'])
        # st.json(user)




