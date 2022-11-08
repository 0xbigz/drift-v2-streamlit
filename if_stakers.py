import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"

print(driftpy.__dir__())
# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
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

import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import InsuranceFundStake, SpotMarket
from driftpy.addresses import * 

async def insurance_fund_page(ch: ClearingHouse):
    all_stakers = await ch.program.account['InsuranceFundStake'].all()
    
    authorities = set()
    dfs = []
    for x in all_stakers: 
        key = str(x.public_key)
        staker: InsuranceFundStake = x.account

        if staker.authority not in authorities:
            authorities.add(staker.authority)

        data = staker.__dict__
        data.pop('padding')
        data['key'] = key
        dfs.append(data)

    st.text('Number of Stakers: ' + str(len(all_stakers)))
    st.text('Number of Unique Stakers: ' + str(len(authorities)))

    conn = ch.program.provider.connection
    state = await get_state_account(ch.program)
    for i in range(state.number_of_spot_markets):
        spot = await get_spot_market_account(ch.program, i)
        total_n_shares = spot.insurance_fund.total_shares

        if_vault = get_insurance_fund_vault_public_key(ch.program_id, i)
        v_amount = int((await conn.get_token_account_balance(if_vault))['result']['value']['amount'])

        for staker_df in dfs: 
            if staker_df['market_index'] == i:
                n_shares = staker_df['if_shares']
                balance = v_amount * n_shares / total_n_shares
                staker_df['$ balance'] = f"{balance / QUOTE_PRECISION:,.2f}"

        st.write(f'Spot Market {i} total vault balance: {v_amount/QUOTE_PRECISION:,.2f}')
    
    stakers = pd.DataFrame(data=dfs)
    st.write(stakers)