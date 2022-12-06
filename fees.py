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

async def fee_page(ch: ClearingHouse):
    state = await get_state_account(ch.program)
    


    col1, col2 = st.columns(2)
    tbl1, tbl2 = st.columns(2)

    col1.markdown('[Perp Trading](https://github.com/drift-labs/protocol-v2/blob/24704d781748f3266840a3c68a5d4f65e629826c/programs/drift/src/math/fees.rs#L345)')
    
    df= pd.DataFrame(state.perp_fee_structure.fee_tiers)
    df['taker_fee'] = df['fee_numerator']/df['fee_denominator']
    df['maker_rebate'] = df['maker_rebate_numerator']/df['maker_rebate_denominator']
    # df['maker_rebate'] = df['maker_rebate_numerator']/df['maker_rebate_denominator']
    df['referee_fee'] = df['referee_fee_numerator']/df['referee_fee_denominator']
    df['referrer_reward'] = df['referrer_reward_numerator']/df['referrer_reward_denominator']
    df['TIER'] = [x+1 if x < 5 else 'VIP' for x in range(len(df)) ]
    df = df[df.taker_fee!=0]
    tbl1.dataframe(df[['TIER', 'taker_fee', 'maker_rebate', 'referee_fee', 'referrer_reward']])


    col2.markdown('[Spot Trading](https://github.com/drift-labs/protocol-v2/blob/24704d781748f3266840a3c68a5d4f65e629826c/programs/drift/src/math/fees.rs#L381)')
    df= pd.DataFrame(state.spot_fee_structure.fee_tiers)
    df['taker_fee'] = df['fee_numerator']/df['fee_denominator']
    df['maker_rebate'] = df['maker_rebate_numerator']/df['maker_rebate_denominator']
    # df['maker_rebate'] = df['maker_rebate_numerator']/df['maker_rebate_denominator']
    df['referee_fee'] = df['referee_fee_numerator']/df['referee_fee_denominator']
    df['referrer_reward'] = df['referrer_reward_numerator']/df['referrer_reward_denominator']
    df['TIER'] = [x+1 if x < 1 else 'VIP' for x in range(len(df)) ]
    df = df[df.taker_fee!=0]
    tbl2.dataframe(df[['TIER', 'taker_fee', 'maker_rebate', 'referee_fee', 'referrer_reward']])
