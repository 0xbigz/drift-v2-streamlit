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

async def fee_page(ch: DriftClient):
    state = await get_state_account(ch.program)
    
    st.write('note: all values are presented as percentages on notional (e.g. 0.01 = 1%)')

    col1, col2 = st.columns(2)
    tbl1, tbl2 = st.columns(2)

    col1.markdown('[Perp Trading](https://github.com/drift-labs/protocol-v2/blob/24704d781748f3266840a3c68a5d4f65e629826c/programs/drift/src/math/fees.rs#L345)')
    
    df= pd.DataFrame(state.perp_fee_structure.fee_tiers)
    df['taker_fee'] = df['fee_numerator']/df['fee_denominator']
    df['maker_rebate'] = df['maker_rebate_numerator']/df['maker_rebate_denominator']
    # df['maker_rebate'] = df['maker_rebate_numerator']/df['maker_rebate_denominator']
    df['referee_discount'] = df['referee_fee_numerator']/df['referee_fee_denominator']
    df['referrer_reward'] = df['referrer_reward_numerator']/df['referrer_reward_denominator']
    df['TIER'] = [x+1 if x < 5 else 'VIP' for x in range(len(df)) ]
    df = df[df.taker_fee!=0]
    tbl1.dataframe(df[['TIER', 'taker_fee', 'maker_rebate', 'referee_discount', 'referrer_reward']])


    col2.markdown('[Spot Trading](https://github.com/drift-labs/protocol-v2/blob/24704d781748f3266840a3c68a5d4f65e629826c/programs/drift/src/math/fees.rs#L381)')
    df= pd.DataFrame(state.spot_fee_structure.fee_tiers)
    df['taker_fee'] = df['fee_numerator']/df['fee_denominator']
    df['maker_rebate'] = df['maker_rebate_numerator']/df['maker_rebate_denominator']
    # df['maker_rebate'] = df['maker_rebate_numerator']/df['maker_rebate_denominator']
    df['referee_discount'] = df['referee_fee_numerator']/df['referee_fee_denominator']
    df['referrer_reward'] = df['referrer_reward_numerator']/df['referrer_reward_denominator']
    df['TIER'] = [x+1 if x < 1 else 'VIP' for x in range(len(df)) ]
    df = df[df.taker_fee!=0]
    tbl2.dataframe(df[['TIER', 'taker_fee', 'maker_rebate', 'referee_discount', 'referrer_reward']])

    ccol1, ccol2 = st.columns(2)
    ttbl1, ttbl2 = st.columns(2)

    ccol1.write('Other Perp Fees')
    perps = []
    spots = []
    for x in range(state.number_of_markets):
        perps.append(await get_perp_market_account(ch.program, x))
    for x in range(state.number_of_spot_markets):
        spots.append(await get_spot_market_account(ch.program, x))

    liq_fees = [(x.liquidator_fee/1e6, x.if_liquidation_fee/1e6) for x in perps]
    ttbl1.dataframe(pd.DataFrame(liq_fees, 
                              columns=['liquidator_fee', 'insurance_fee'],
                            index=[bytes(x.name).decode('utf-8') for x in perps]))

    ccol2.write('Other Spot Fees')
    liq_fees = [(spot.liquidator_fee/1e6, spot.if_liquidation_fee/1e6, spot.insurance_fund.total_factor/1e6, ) for spot in spots]
    ttbl2.dataframe(pd.DataFrame(liq_fees, 
                              columns=['liquidator_fee', 'insurance_fee', 'borrow_rate_fee'],
                            index=[bytes(x.name).decode('utf-8') for x in spots]))
    
    st.write('definitions:')
    st.write('`liquidator_fee`: additional premium for taking over asset/liability pairs at oracle')   
    st.write('`insurance_fee`: fee reserved for revenue pool -> insurance fund')
    st.write('`borrow_rate_fee`:  fraction of borrow interest reserved for revenue pool -> insurance fund')