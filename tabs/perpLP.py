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
from solana.rpc.types import MemcmpOpts
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
import datetime
import pytz

from datafetch.s3_fetch import load_user_lp, load_volumes

EPSILON = 1e-9

async def perp_lp_page(ch: DriftClient, env):
    is_devnet = env == 'devnet'
    state = await get_state_account(ch.program)
    a00, a11, a22, a33 = st.columns(4)
    user_lookup = a33.radio('lookup LP users:', ['Active', 'DLP', 'Ever', 'Never'], index=3, 
                           horizontal=True
                           )
    user_accounts_w_lp = []
    if user_lookup != 'Never':
        if user_lookup == 'Ever':
            all_users = await ch.program.account['User'].all(filters=[MemcmpOpts(offset=4350, bytes='2')])
        elif user_lookup == 'DLP':
            all_users = await ch.program.account['User'].all(filters=[MemcmpOpts(offset=72, bytes='7Ev1Wb17tTZuQFYjieDAx2pbrSzym2eaV')])
        else:
            all_users = await ch.program.account['User'].all(filters=[MemcmpOpts(offset=4267, bytes='2i')])
        # with tabs[0]:
        st.write('found', len(all_users), 'current/former LPs')
        df = pd.DataFrame([x.account.__dict__ for x in all_users])
        user_accounts_w_lp = [str(x.public_key) for x in all_users]
        # assert(len(df) == len(all_users))
        if len(df):
            df.name = df.name.apply(lambda x: bytes(x).decode('utf-8', errors='ignore'))
            df['public_key'] = [str(x.public_key) for x in all_users]
    else: 
        all_users = []
        df = pd.DataFrame()
    # with st.expander('ref accounts'):
    #     st.write(all_refs_stats)

    lastest_date = pd.to_datetime(datetime.datetime.now(), utc=True)

    start_date = a11.date_input(
            "start date:",
            lastest_date - datetime.timedelta(days=1),
            min_value=datetime.datetime(2022, 11, 4),
            max_value=lastest_date,
        )  # (datetime.datetime.now(tzInfo)))
    end_date = a22.date_input(
            "end date:",
            lastest_date,
            min_value=datetime.datetime(2022, 11, 4),
            max_value=lastest_date,
        )  # (datetime.datetime.now(tzInfo)))
    dates = pd.date_range(start_date, end_date)
    mi = a00.selectbox('market index:', range(0, state.number_of_markets), 0)
    perp_market = await get_perp_market_account(ch.program, mi)
    market_name = ''.join(map(chr, perp_market.name)).strip(" ")
    tit0, = st.columns([1])
    tit0.write(f'"{market_name}" lp info:')

    def calculate_market_open_bids_asks(base_asset_reserve, min_base_asset_reserve, max_base_asset_reserve):
        max_asks = max(0, max_base_asset_reserve - base_asset_reserve)/1e9
        max_bids = max(0, base_asset_reserve - min_base_asset_reserve)/1e9
        return max_bids, max_asks
    max_bids, max_asks = calculate_market_open_bids_asks(perp_market.amm.base_asset_reserve,
                                                        perp_market.amm.min_base_asset_reserve,
                                                        perp_market.amm.max_base_asset_reserve,
                                                        )

    tabs = st.tabs(['LPs', 'individual LP', 'historical lp fees', 'aggregate fees'])


    with tabs[0]:
        container, = st.columns([1])
        a0, a1, a2, a3 = st.columns(4)

        lps = {}
        for usr in all_users:
            for x in usr.account.perp_positions:
                if (x.lp_shares != 0 and user_lookup == 'Active'):
                    key = str(usr.public_key)
                    print(lps.keys(), key)
                    if key in lps.keys():
                        lps[key].append(x)
                    else:
                        lps[key] = [x]
        if len(lps.keys()):
            dff = pd.concat({key: pd.DataFrame(val) for key,val in lps.items()})
            dff.index.names = ['public_key', 'position_index']
            dff = dff.reset_index()
            dff = df[['authority', 'name', 'last_active_slot', 'public_key', 'last_add_perp_lp_shares_ts']].merge(dff, on='public_key')
            print(dff.columns)
            dff = dff[[
            'authority', 'name', 
            'lp_shares',
            'last_active_slot', 'public_key',
            'last_add_perp_lp_shares_ts', 'market_index', 
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

            # cols = (st.multiselect('columns:', ))
            # dff = dff[cols]
            st.write('all lp positions')
            st.dataframe(dff)

        container.write(f'lp cooldown time: {state.lp_cooldown_time} seconds | jit intensity: {perp_market.amm.amm_jit_intensity}')

        # st.write(perp_market.amm)
        bapl = perp_market.amm.base_asset_amount_per_lp/1e9
        tbapl = perp_market.amm.target_base_asset_amount_per_lp/1e9
        minordsize = perp_market.amm.order_step_size/1e9

        qapl = perp_market.amm.quote_asset_amount_per_lp/1e9
        baawul = perp_market.amm.base_asset_amount_with_unsettled_lp/1e9
        baa_w_amm = perp_market.amm.base_asset_amount_with_amm/1e9

        a1.metric('base asset amount per lp:', bapl, f'target base: {tbapl}')
        a2.metric('quote asset amount per lp:', qapl)
        a3.metric('unsettled base asset amount with lp:', baawul)

        if perp_market.amm.amm_jit_intensity > 100:
            if abs(baa_w_amm) < minordsize:
                st.write('amm doesnt want to make 🚫')
            elif baa_w_amm > 0:
                st.write('amm wants to jit make user LONGS ✅')
            else:
                st.write('amm wants to jit make user SHORTS ✅')

            if bapl > tbapl:
                st.write('lp wants to jit make user LONGS')
            elif bapl < tbapl: 
                st.write('lp wants to jit make user SHORTS')
            else:
                st.write('lp dont want to jit make')

            baa_w_amm = perp_market.amm.base_asset_amount_with_amm/1e9
            protocol_owned_min_side_liq = min(max_asks, max_bids) * (perp_market.amm.sqrt_k-perp_market.amm.user_lp_shares)/perp_market.amm.sqrt_k
            if abs(baa_w_amm) < protocol_owned_min_side_liq/10:
                st.write(f'lp allowed to make  ✅')
            else:
                st.write(f'lp not allowed to make 🚫')

            st.write(f'Total AMM Liquidity (bids: {max_bids} | asks: {max_asks})')
            st.write(f'> protocol_owned_min_side_liq={protocol_owned_min_side_liq}')
            st.write(f'> is baa_w_amm: {baa_w_amm} < {protocol_owned_min_side_liq/10} ?')
        else:
            st.write(f'no lp jit making: amm_jit_intensity={perp_market.amm.amm_jit_intensity} (<=100)')

    with tabs[1]:
        ucol, dcol = st.columns(2)
        if len(user_accounts_w_lp):
            user_pubkey = ucol.selectbox('user account:', user_accounts_w_lp)
        else:
            user_pubkey = ucol.text_input('user account:', '4kojmr5Xbgrfgg5db9bdEuVrDtb1kjBCcNGHsv8S2TdZ')
        # date = dcol.date_input('date:',datetime.now())
        if user_pubkey is not None and user_pubkey.strip() != '':
            # dates = [date]
            df, urls = load_user_lp(dates, user_pubkey, True, is_devnet)
            st.json(urls, expanded=False)
            if len(df):
                df.index = [pd.to_datetime(int(x*1e9)) for x in df.ts]
                df =  df.sort_index()
                df = df.loc[start_date:end_date]
                df['px'] = (-df['deltaQuoteAssetAmount']/df['deltaBaseAssetAmount'].replace(0, np.nan))
                # st.write(df)
                for mi1 in df.marketIndex.unique():
                    df1 = df[df.marketIndex==mi1]
                    s1 = df1['pnl'].cumsum() # pnl
                    s3 = df1['deltaBaseAssetAmount'].cumsum()
                    s4 = df1['deltaQuoteAssetAmount'].cumsum()
                    s2 = df1['px']

                    total_base = df1['deltaBaseAssetAmount'].sum()
                    total_quote = df1['deltaQuoteAssetAmount'].sum()
                    total_price = -total_quote/total_base

                    dir = 'bought'
                    if total_base < 0:
                        dir = 'sold'
                    
                    st.write(dir, total_base, '@ $', total_price)

                    findf = pd.concat({'cumPnl':s1,
                                    'px': s2,
                                        'cumBase': s3,
                                        'cumQuote': s4
                                        },axis=1)
                    findf['cumPrice'] = -findf['cumQuote']/findf['cumBase']
                    fig = findf.plot()
                    fig.update_layout( 
                        title='Perp Market Index='+str(mi1),
                        xaxis_title="date",
                        yaxis_title="value",
                    )
                    st.plotly_chart(fig)
                    st.dataframe(df1)
            st.write(' -------------------- ')
            st.write('grafana export path to file:')

            f1, f2 = st.columns(2)
            #'Info Per Share (From Start)-data-as-joinbyfield-2023-07-18 15_44_09.csv'
            #'Oracle Price-data-2023-07-18 15_22_18.csv'
            ff1 = f1.text_input('info per share:', '')
            ff2 = f2.text_input('oracle price:', )

            if ff1 != '' and ff2 != '':
                q1 = pd.read_csv(ff1).set_index('Time')
                oraclePx = pd.read_csv(ff2).set_index('Time')
                oraclePx = pd.DataFrame(oraclePx).iloc[:,0].apply(lambda x: float(x[1:]))
                bq1 = pd.concat([q1, oraclePx],axis=1)
                bq1.index = pd.to_datetime(bq1.index)
                st.dataframe(bq1)

                ang = df[df.marketIndex==0]
                ang['curShares'] = (((ang['nShares']*10).apply(np.floor)/10)*ang['action'].apply(lambda x: {'addLiquidity': 1,
                                                                                'removeLiquidity':-1,
                                                                                'settleLiquidity':0,
                                                                                }[x])).cumsum().fillna(0)

                st.write(bq1)
                st.write(ang[['curShares', 'action']])
                dres = pd.concat([bq1, ang[['curShares', 'action']]],axis=1).ffill()
                dres_c = dres.loc[ang.index]
                # dres_c['SOL-PERP Oracle Price']*dres_c['baalp1']
                dres_c['myBAA'] = (dres_c['baalp1'].diff()*dres_c['curShares'].shift(1))
                dres_c['myQAA'] = (dres_c['qaalp1'].diff()*dres_c['curShares'].shift(1))

                dres_c['burnCost'] = -(dres_c['myBAA'].abs() * dres_c['SOL-PERP Oracle Price'])
                # dres_c.loc[dres_c['action']!='removeLiquidity', 'burnCost'] = 0

                st.dataframe(dres_c)

                st.plotly_chart(dres[['curShares']].plot())
    with tabs[2]:
        a0, a1, a2, a3 = st.columns(4)
        # mi = a0.selectbox('market index:', range(0, state.number_of_markets), 0, key='mom1')
        # date = a1.date_input('date:',datetime.now(), key='jkdk')

        perp_market = await get_perp_market_account(ch.program, mi)

        # dates = [date]
        st.markdown("""
        - lp jit: 80% * taker fee
        - amm lp jit split: 80% *taker fee * lp fraction of amm
        - with amm: 80% *taker fee * lp fraction of amm
        """)
        name = str(''.join(map(chr, perp_market.name))).strip()
        df, urls = load_volumes(dates, name, True, is_devnet)

        current_lp_faction_of_amm = perp_market.amm.user_lp_shares/perp_market.amm.sqrt_k

        st.json(urls, expanded=False)

        st.write('fraction of amm by user lp:', current_lp_faction_of_amm)
        fees_by_action = df.groupby('actionExplanation')['takerFee'].sum()
        lp_fees = fees_by_action.loc['orderFilledWithAmm']*.8*current_lp_faction_of_amm
        st.metric('lp fees', f'${lp_fees:,.2f}')

        # fees_by_action
        # st.dataframe(df)

    with tabs[3]:
        fees_url = 'https://drift-fee-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/market/'
        fees_url += market_name + '/2023/6'
        
        rx = requests.get(fees_url).json()

        
        # rx = None
        # with open(fees_url,'r') as f:
        #     rx = json.load(f)

        s1, s2, s3 = st.columns(3)

        fees1 = [x['fees'] for x in rx]
        fees2 = [{z['feeType']: float(z['feeAmount'])/1e6 for z in y} for y in fees1]
        df = pd.DataFrame(fees2, index=[x['startTs'] for x in rx])
        df.index = pd.to_datetime(df.index*1000000000)
        df = df.loc[start_date:end_date]
        # st.json(rx)
        # df = pd.read_json(fees_url)

        s1.metric('taker fees:',  f'${ df.sum().sum():,.2f}')


        retained_lb = ((df['liquidation']*.5).sum() 
                       + (df['orderFilledWithAmm'].sum()*.2)
                       + (df['orderFilledWithAmmJit'].sum()*1)
                       + (df['orderFilledWithMatch'].sum()*.5)
                       )
        s2.metric('est l.b. retained fees:',  f'${ retained_lb:,.2f}')


        st.dataframe(df)
