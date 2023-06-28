import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
import copy
import plotly.express as px

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
from helpers import serialize_perp_market_2, serialize_spot_market, all_user_stats
from anchorpy import EventParser
import asyncio

import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import InsuranceFundStake, SpotMarket
from driftpy.addresses import * 

# @st.experimental_memo(ttl=60*5)  # 5 min TTL this time
async def load_users(_ch_user_act, filt):
    if filt == 'All':
        all_users = await _ch_user_act.all()
    elif filt == 'Active':
        all_users = await _ch_user_act.all(memcmp_opts=[MemcmpOpts(offset=4350, bytes='1')])
    elif filt == 'Idle':
        all_users = await _ch_user_act.all(memcmp_opts=[MemcmpOpts(offset=4350, bytes='2')])        
    elif filt == 'Open Order':
        all_users = await _ch_user_act.all(memcmp_opts=[MemcmpOpts(offset=4352, bytes='2')])
    elif filt == 'SuperStakeSOL':
        all_users = await _ch_user_act.all(memcmp_opts=[MemcmpOpts(offset=72, bytes='3LRfP5UkK8aDLdDMsJS3D')])
    else:
        all_users = await _ch_user_act.all(memcmp_opts=[MemcmpOpts(offset=4354, bytes='2')])
    return all_users


async def userstatus_page(ch: ClearingHouse):
    state = await get_state_account(ch.program)
    s1, s2 = st.columns(2)
    filt = s1.radio('filter:', ['All', 'Active', 'Idle', 'Open Order', 'Open Auction', 'SuperStakeSOL'], index=1, horizontal=True)
    oracle_distort = s2.slider('base oracle distortion:', .01, 2.0, 1.0, .1, help="alter non-stable token oracles by this factor multiple (default=1, no alteration)")
    tabs = st.tabs([filt.lower() + ' users', 'LPs', 'oracle scenario analysis'])
    all_users = await load_users(ch.program.account['User'], filt)

    df = pd.DataFrame([x.account.__dict__ for x in all_users])
    df['public_key'] = [str(x.public_key) for x in all_users]

    if oracle_distort == 1:
        oracle_distort = None

    stats_df, chu = await all_user_stats(all_users, ch, oracle_distort)

    # with st.expander('ref accounts'):
    #     st.write(all_refs_stats)
    with tabs[0]:
        col1, col11, col2, col3, col4 = st.columns(5)
        col1.metric('number of user accounts', state.number_of_sub_accounts, filt.lower()+' users = '+str(len(all_users)))

        st.write(filt.lower() + ' users:')
        df.name = df.name.apply(lambda x: bytes(x).decode('utf-8', errors='ignore'))
        st.dataframe(df)

        user_tvl = stats_df.spot_value.sum()
        col2.metric('user tvl', f'${user_tvl:,.2f}', f'${stats_df.upnl.sum():,.2f} upnl')
        state = await get_state_account(ch.program)
        num_markets = state.number_of_spot_markets
        aa = 0
        vamm = 0
        usdc_market = None
        spot_acct_dep = 0
        spot_acct_bor = 0

        for market_index in range(num_markets):
            market = await chu.get_spot_market(market_index)
            if market_index == 0:
                usdc_market = market
            conn = ch.program.provider.connection
            # ivault_pk = market.insurance_fund.vault
            svault_pk = market.vault
            sv_amount = int((await conn.get_token_account_balance(svault_pk))['result']['value']['amount'])
            sv_amount /= (10**market.decimals)
            px1 = ((await chu.get_spot_oracle_data(market)).price/1e6)
            sv_amount *= px1
            spot_acct_dep += (market.deposit_balance * market.cumulative_deposit_interest/1e10)/(1e9)*px1
            spot_acct_bor -= (market.borrow_balance * market.cumulative_borrow_interest/1e10)/(1e9)*px1

            aa += sv_amount
        

        vamm_upnl = 0 
        for market_index in range(state.number_of_markets):
            market = await chu.get_perp_market(market_index)
            px1 = ((await chu.get_perp_oracle_data(market)).price/1e6)
            fee_pool = (market.amm.fee_pool.scaled_balance * usdc_market.cumulative_deposit_interest/1e10)/(1e9)
            pnl_pool = (market.pnl_pool.scaled_balance * usdc_market.cumulative_deposit_interest/1e10)/(1e9)
            vamm += pnl_pool + fee_pool
            vamm_upnl -= market.amm.quote_asset_amount/1e6 + (market.amm.base_asset_amount_with_amm + market.amm.base_asset_amount_with_unsettled_lp)/1e9 * px1

        col11.metric('mkt tvl', f'${spot_acct_dep+spot_acct_bor:,.2f}')
        col3.metric('vamm tvl', f'${vamm:,.2f}', f'{vamm_upnl:,.2f} upnl')

        stats_df['spot_value'] = stats_df['spot_value'].astype(float)
        stats_df['upnl'] = stats_df['upnl'].astype(float)
        ddd = (stats_df['spot_value']+stats_df['upnl'])
        ddd = ddd[ddd<=0]

        sddd = stats_df[stats_df['spot_value']<0]['spot_value']

    
        stats_df.loc[stats_df['spot_value'] < 0, 'spot bankrupt'] = True
        stats_df.loc[((stats_df['upnl'] < 0) & (stats_df['spot_value'] < -stats_df['upnl'])), 'perp bankrupt'] = True

        st.dataframe(stats_df)

        st.write('bankruptcies:', ddd.sum())
        st.write(f'> spot: {sddd.sum():,.2f}, perp: {ddd.sum()-sddd.sum():,.2f}')
        col4.metric('vault tvl', f'${aa:,.2f}', f'{aa-(vamm+user_tvl):,.2f} excess')


        new_accounts = df[df['sub_account_id']==0]
        st.write(
            len(new_accounts), 
        'are subaccount id = 0', 
        '($', 
        stats_df.loc[new_accounts['public_key'].values]['spot_value'].sum() ,
        ')')
        st.write(len(stats_df[stats_df['spot_value']>0]), 'have a balance', '($',stats_df[stats_df['spot_value']>0].sum(),')')

        df111 = pd.concat({
            'leverage': (stats_df['total_liability']/stats_df['spot_value']),
            'position_size': stats_df['spot_value'],
            'position_size2': (stats_df['spot_value']+1).pipe(np.log),
        },axis=1)
        fig111 = px.scatter(df111, x='leverage', y='position_size', size='position_size2', size_max=10, color='leverage', log_y=True)
        st.plotly_chart(fig111)


    with tabs[1]:
        st.write('lp cooldown time:', state.lp_cooldown_time, 'seconds')
        lps = {}
        dff = pd.DataFrame()
        for usr in all_users:
            for x in usr.account.perp_positions:
                if x.lp_shares != 0:
                    key = str(usr.public_key)
                    print(lps.keys(), key)
                    if key in lps.keys():
                        lps[key].append(x)
                    else:
                        lps[key] = [x]
        if len(lps):
            dff = pd.concat({key: pd.DataFrame(val) for key,val in lps.items()})
            dff.index.names = ['public_key', 'position_index']
            dff = dff.reset_index()
            dff = df[['authority', 'name', 'last_active_slot', 'public_key', 'last_add_perp_lp_shares_ts']].merge(dff, on='public_key')
            print(dff.columns)
            dff = dff[['authority', 'name', 
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

        st.write('perp market lp info:')
        a0, a1, a2, a3 = st.columns(4)
        mi = a0.selectbox('market index:', range(0, state.number_of_markets), 0)


        perp_market = await chu.get_perp_market(mi)
        # st.write(perp_market.amm)
        bapl = perp_market.amm.base_asset_amount_per_lp/1e9
        qapl = perp_market.amm.quote_asset_amount_per_lp/1e6
        baawul = perp_market.amm.base_asset_amount_with_unsettled_lp/1e9

        a1.metric('base asset amount per lp:', bapl)
        a2.metric('quote asset amount per lp:', qapl)
        a3.metric('unsettled base asset amount with lp:', baawul)


        # cols = (st.multiselect('columns:', ))
        # dff = dff[cols]
        st.write('raw lp positions')
        st.dataframe(dff)


        def get_wouldbe_lp_settle(row):
            def standardize_base_amount(amount, step):
                remainder = amount % step
                standard = amount - remainder
                return standard, remainder

            mi = row['market_index']
            pm = chu.CACHE['perp_markets'][mi]
            delta_baapl = pm.amm.base_asset_amount_per_lp/1e9 - row['last_base_asset_amount_per_lp']
            delta_qaapl = pm.amm.quote_asset_amount_per_lp/1e6 - row['last_quote_asset_amount_per_lp']

            delta_baa = delta_baapl * row['lp_shares']
            delta_qaa = delta_qaapl * row['lp_shares']

            standard_baa, remainder_baa = standardize_base_amount(delta_baa, pm.amm.order_step_size/1e9)

            row['remainder_base_asset_amount'] += remainder_baa
            row['base_asset_amount'] += standard_baa
            row['quote_asset_amount'] += delta_qaa
            row['quote_entry_amount'] += delta_qaa
            row['quote_break_even_amount'] += delta_qaa

            return row



        newd = dff.apply(get_wouldbe_lp_settle, axis=1)
            

        st.write('would-be settled lp positions')
        st.dataframe(newd)


    with tabs[2]:
        scol, scc = st.columns([4,1])
        omin, omax = scol.slider(
        'Select an oracle multiplier range',
        0.0, 10.0, (0.0, 2.0), step=.05)

        only_perp_index = scc.selectbox('only perp', ([None] + list(range(0, state.number_of_markets))))
        oracle_distorts = [float(x)/10 for x in (list(np.linspace(omin, omax*10, 100)))]
        all_stats = {}

        df1 = pd.DataFrame(chu.CACHE['perp_market_oracles'])
        # with st.expander('market override:'):
        #     edited_df = st.experimental_data_editor(df1)

        for oracle_distort_x in oracle_distorts:
            pure_cache = copy.deepcopy(chu.CACHE)
            stats_df, chunew = await all_user_stats(all_users, 
                                                    ch, 
                                                    oracle_distort_x, 
                                                    pure_cache=pure_cache,
                                                    only_perp_index=only_perp_index
                                                    )

            stats_df['spot_value'] = stats_df['spot_value'].astype(float)
            stats_df['upnl'] = stats_df['upnl'].astype(float)
            ddd = (stats_df['spot_value']+stats_df['upnl'])
            ddd = ddd[ddd<=0]

            sddd = stats_df[stats_df['spot_value']<0]['spot_value']

        
            stats_df.loc[stats_df['spot_value'] < 0, 'spot bankrupt'] = True
            stats_df.loc[((stats_df['upnl'] < 0) & (stats_df['spot_value'] < -stats_df['upnl'])), 'perp bankrupt'] = True
            all_stats[oracle_distort_x] = [ddd.sum(), sddd.sum(), ddd.sum()-sddd.sum()]


        statdf = pd.DataFrame(all_stats, index=['bankruptcy', 'spot bankruptcy', 'perp bankruptcy']).T.abs()
        nom = ""
        if only_perp_index is not None:
            nom = bytes(chu.CACHE['perp_markets'][only_perp_index].name).decode('utf-8').strip(' ')+" "
            perp_market_account_if = chu.CACHE['perp_markets'][only_perp_index].insurance_claim
            max_if_perp_payment = (perp_market_account_if.quote_max_insurance - perp_market_account_if.quote_settled_insurance )/1e6
            statdf['max insurance claim'] = statdf['perp bankruptcy'].clip(0, max_if_perp_payment) + statdf['spot bankruptcy']
        fig = statdf.plot()
        fig.update_layout(
            title=nom+"Bankruptcy Risk",
            xaxis_title='oracle distortion %',
            yaxis_title="bankruptcy $",
        
            )
        st.plotly_chart(fig, use_container_width=True)


    
