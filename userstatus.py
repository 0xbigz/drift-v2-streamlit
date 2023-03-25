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

async def userstatus_page(ch: ClearingHouse):
    state = await get_state_account(ch.program)

    all_users = await ch.program.account['User'].all(memcmp_opts=[MemcmpOpts(offset=4350, bytes='1')])
    # with st.expander('ref accounts'):
    #     st.write(all_refs_stats)
    st.metric('number of user accounts', state.number_of_sub_accounts, 'active users = '+str(len(all_users)))

    st.write('active users:')
    df = pd.DataFrame([x.account.__dict__ for x in all_users])
    df.name = df.name.apply(lambda x: bytes(x).decode('utf-8', errors='ignore'))
    st.dataframe(df)

    
