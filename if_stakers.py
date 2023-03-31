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
import plotly.express as px

  
async def insurance_fund_page(ch: ClearingHouse):

    try:
        all_stakers = await ch.program.account['InsuranceFundStake'].all()
    except:
        all_stakers = []

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

    st.markdown('[USDC Insurance Vault](https://solscan.io/account/2CqkQvYxp9Mq4PqLvAQ1eryYxebUh4Liyn5YMDtXsYci) | [Stake Example (Python)](https://github.com/0xbigz/driftpy-examples/blob/master/if_stake.py)')
    state = await get_state_account(ch.program)

    bbs = st.columns(state.number_of_spot_markets)
    ccs = st.columns(state.number_of_spot_markets)
    
    dcol1, dcol2 = st.columns(2)
    dcol1.metric("Number of Stakes", str(len(all_stakers)),str(len(authorities))+" Unique Stakers")

    conn = ch.program.provider.connection
    ts = time.time()
    spot_markets = []
    protocol_balances = []

    for i in range(state.number_of_spot_markets):
        spot = await get_spot_market_account(ch.program, i)
        total_n_shares = spot.insurance_fund.total_shares
        user_n_shares = spot.insurance_fund.user_shares


        unstaking_period = spot.insurance_fund.unstaking_period

        if spot.insurance_fund.total_factor:
            factor_for_protocol = spot.insurance_fund.user_factor/spot.insurance_fund.total_factor
        else:
            factor_for_protocol = 0 
        protocol_n_shares = total_n_shares - user_n_shares
        spot_markets.append(spot)

        # get_token_amount(spot.revenue_pool.scaled_balance, spot, )
        if_vault = get_insurance_fund_vault_public_key(ch.program_id, i)
        try:
            v_amount = int((await conn.get_token_account_balance(if_vault))['result']['value']['amount'])
        except:
            v_amount = 0

        protocol_balance = v_amount * protocol_n_shares / (max(total_n_shares,1))
        protocol_balances.append(protocol_balance/10**spot.decimals)

        for staker_df in dfs: 
            if staker_df['market_index'] == i:
                n_shares = staker_df['if_shares']
                balance = v_amount * n_shares / total_n_shares
                staker_df['$ balance'] = balance / (10 ** spot.decimals)

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
        next_payment = min(rev_pool_tokens/5, (v_amount*10/365/24))
        if v_amount > 0:
            staker_apr = (next_payment*24*365.25 * factor_for_protocol)/v_amount
        else:
            staker_apr = 0

        ccs[i].metric('revenue pool', f'{symbol}{rev_pool_tokens/10**spot.decimals:,.2f}', 
        f'{symbol}{next_payment/10**spot.decimals:,.2f} next est. hr payment ('+str(np.round(factor_for_protocol*100, 2))+ '% protocol | '+str(np.round(staker_apr*100, 2))+'% staker APR)'
        )



        st.write('')

        # st.write(f'{name} (marketIndex={i}) insurance vault balance: {v_amount/QUOTE_PRECISION:,.2f} (protocol owned: {protocol_balance/QUOTE_PRECISION:,.2f})')
        st.write(f'{name} (marketIndex={i}) time since last settle: {np.round((ts - spot.insurance_fund.last_revenue_settle_ts)/(60*60), 2)} hours')
    stakers = pd.DataFrame(data=dfs)

    total_shares = [spot_markets[r['market_index']].insurance_fund.total_shares for i, r in stakers.iterrows()]
    precisions = [(10 ** spot_markets[r['market_index']].decimals) for i, r in stakers.iterrows()]
    stakers['cost_basis'] /= precisions 
    stakers['if_shares'] /= precisions
    stakers['last_withdraw_request_shares'] /= precisions
    stakers['last_withdraw_request_shares'] = stakers['last_withdraw_request_shares'].replace(0, np.nan)
    stakers['last_withdraw_request_value'] /= 1e6
    stakers['last_withdraw_request_value'] = stakers['last_withdraw_request_value'].replace(0, np.nan)
    stakers['total_shares'] = total_shares
    stakers['total_shares'] /= precisions
    stakers['available_time_to_withdraw'] = stakers['last_withdraw_request_ts'].apply(lambda x: pd.to_datetime((x + unstaking_period) * 1e9) if x!=0 else x)
    stakers['last_withdraw_request_ts'] = stakers['last_withdraw_request_ts'].apply(lambda x: pd.to_datetime(x * 1e9) if x!=0 else x)
    stakers['own %'] = stakers['if_shares']/stakers['total_shares'] * 100
    stakers['authority'] = stakers['authority'].astype(str)
    # print(stakers.columns)    
    stakers['$ balance'] = stakers['$ balance'].astype(float)

    st.write(stakers[['authority', 'market_index', '$ balance', 'if_shares', 'total_shares', 'own %', 'cost_basis', 'last_withdraw_request_shares', 'if_base',
       'last_withdraw_request_value',
       'last_withdraw_request_ts', 'available_time_to_withdraw', 'last_valid_ts',  'key',
       ]].sort_values('$ balance', ascending=False).reset_index(drop=True))

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



    url_market_prefix = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/market/'
    full_if_url = url_market_prefix+"USDC/insurance-fund-records/2023/3"
    st.write(full_if_url)
    rot = pd.read_csv(full_if_url)
    rot = rot.set_index('ts')['amount']
    rot.index = pd.to_datetime((rot.index * 1e9).astype(int))
    st.plotly_chart(rot.cumsum().plot())


    selected = st.selectbox('authority:', sorted(stakers.authority.unique()))
    url_market_prefix = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/authority/'
    full_if_url = url_market_prefix+selected+"/insurance-fund-stake-records/2023/3"
    st.write(full_if_url)
    rot = pd.read_csv(full_if_url)
    rot = rot.set_index('ts')
    rot.index = pd.to_datetime((rot.index * 1e9).astype(int))
    st.dataframe(rot)