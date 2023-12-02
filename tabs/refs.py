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
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio

import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import InsuranceFundStakeAccount, SpotMarketAccount
from driftpy.addresses import * 

async def ref_page(ch: DriftClient):
    # state = await get_state_account(ch.program)
    all_refs_stats = await ch.program.account['ReferrerName'].all()
    with st.expander('ref accounts'):
        st.write(all_refs_stats)
    df = pd.DataFrame([x.account.__dict__ for x in all_refs_stats])
    df.name = df.name.apply(lambda x: bytes(x).decode('utf-8', errors='ignore'))
    st.dataframe(df)

    
