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
from driftpy.clearing_house_user import get_token_amount
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
# using time module
import time
  
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
    ts = time.time()

    for i in range(state.number_of_spot_markets):
        spot = await get_spot_market_account(ch.program, i)
        total_n_shares = spot.insurance_fund.total_shares
        user_n_shares = spot.insurance_fund.user_shares
        protocol_n_shares = total_n_shares - user_n_shares
        spot_markets.append(spot)

        get_token_amount(spot.revenue_pool.scaled_balance, spot, )
        if_vault = get_insurance_fund_vault_public_key(ch.program_id, i)
        v_amount = int((await conn.get_token_account_balance(if_vault))['result']['value']['amount'])
        protocol_balance = v_amount * protocol_n_shares / (max(total_n_shares,1))
        protocol_balances.append(protocol_balance/10**spot.decimals)

        for staker_df in dfs: 
            if staker_df['market_index'] == i:
                n_shares = staker_df['if_shares']
                balance = v_amount * n_shares / total_n_shares
                staker_df['$ balance'] = f"{balance / (10 ** spot.decimals):,.2f}"

        name = str(''.join(map(chr, spot.name)))

        st.write(f'{name} (marketIndex={i}) insurance vault balance: {v_amount/QUOTE_PRECISION:,.2f} (protocol owned: {protocol_balance/QUOTE_PRECISION:,.2f})')
        st.write(f'time since last settle: {np.round((ts - spot.insurance_fund.last_revenue_settle_ts)/(60*60), 2)} hours')
    stakers = pd.DataFrame(data=dfs)

    total_shares = [spot_markets[r['market_index']].insurance_fund.total_shares for i, r in stakers.iterrows()]
    precisions = [(10 ** spot_markets[r['market_index']].decimals) for i, r in stakers.iterrows()]
    stakers['cost_basis'] /= precisions 
    stakers['if_shares'] /= precisions
    stakers['total_shares'] = total_shares
    stakers['total_shares'] /= precisions
    stakers['own %'] = stakers['if_shares']/stakers['total_shares'] * 100
    stakers['authority'] = stakers['authority'].astype(str)
    # print(stakers.columns)    

    st.write(stakers[['authority', 'market_index', '$ balance', 'if_shares', 'total_shares', 'own %', 'cost_basis', 'last_withdraw_request_shares', 'if_base',
       'last_withdraw_request_value',
       'last_withdraw_request_ts', 'last_valid_ts',  'key',
       ]])

    df = stakers[['$ balance', 'authority', 'own %', 'market_index']]
    
    z1, z2 = st.columns(2)
    pie1, z2 = st.columns(2)

    i = z1.selectbox('market index:', sorted(list(df['market_index'].unique())))
    df = df[df.market_index==i]
    df = pd.concat([df, pd.DataFrame([[protocol_balances[i], 'protocol', 100 - df['own %'].sum(), i]], index=[0], columns=df.columns)])

    fig = px.pie(df, values='own %', names='authority',
                title='MarketIndex='+str(i)+' IF breakdown',
                hover_data=['$ balance'], labels={'$ balance':'balance'})
    pie1.plotly_chart(fig)