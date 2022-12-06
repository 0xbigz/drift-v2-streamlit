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

    st.markdown('[USDC Insurance Vault](https://solscan.io/account/2CqkQvYxp9Mq4PqLvAQ1eryYxebUh4Liyn5YMDtXsYci) | [Stake Example (Python)](https://github.com/drift-labs/driftpy/blob/master/examples/if_stake.py)')
    
    col1, col2 = st.columns(2)
    bbs = [col1, col2]
    ccol1, ccol2 = st.columns(2)
    ccs = [ccol1, ccol2]
    
    st.metric("Number of Stakes", str(len(all_stakers)),str(len(authorities))+" Unique Stakers")

    conn = ch.program.provider.connection
    state = await get_state_account(ch.program)
    spot_markets = []
    for i in range(state.number_of_spot_markets):
        spot = await get_spot_market_account(ch.program, i)
        total_n_shares = spot.insurance_fund.total_shares
        user_n_shares = spot.insurance_fund.user_shares
        protocol_n_shares = total_n_shares - user_n_shares
        spot_markets.append(spot)

        if_vault = get_insurance_fund_vault_public_key(ch.program_id, i)
        v_amount = int((await conn.get_token_account_balance(if_vault))['result']['value']['amount'])
        protocol_balance = v_amount * protocol_n_shares / (max(total_n_shares,1))

        for staker_df in dfs: 
            if staker_df['market_index'] == i:
                n_shares = staker_df['if_shares']
                balance = v_amount * n_shares / total_n_shares
                staker_df['$ balance'] = f"{balance / (10 ** spot.decimals):,.2f}"

        name = str(''.join(map(chr, spot.name)))

        symbol = '$' if i==0 else ''
        bbs[i].metric(f'{name} (marketIndex={i}) insurance vault balance:',
         f'{symbol}{v_amount/(10**spot.decimals):,.2f}',
         f'{symbol}{protocol_balance/10**spot.decimals:,.2f} protocol owned'
         ) 

        rev_pool_tokens = get_token_amount(
                        spot.revenue_pool.scaled_balance,
                        spot, 
                        'SpotBalanceType.Deposit()'
                    )

        #capped at 1000% APR
        next_payment = min(rev_pool_tokens, (v_amount*10/365/24))
        ccs[i].metric('revenue pool', f'{symbol}{rev_pool_tokens/10**spot.decimals:,.2f}', f'{symbol}{next_payment/10**spot.decimals:,.2f} next est. hourly payment')

        st.write('')

    
    stakers = pd.DataFrame(data=dfs)

    precisions = [(10 ** spot_markets[r['market_index']].decimals) for i, r in stakers.iterrows()]
    stakers['cost_basis'] /= precisions 
    stakers['if_shares'] /= precisions

    # print(stakers.columns)    

    st.write(stakers[['authority', 'market_index', '$ balance', 'if_shares', 'cost_basis', 'last_withdraw_request_shares', 'if_base',
       'last_withdraw_request_value',
       'last_withdraw_request_ts', 'last_valid_ts',  'key',
       ]])