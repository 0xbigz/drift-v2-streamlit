import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
import copy
import plotly.express as px
pd.options.plotting.backend = "plotly"
from datafetch.transaction_fetch import load_token_balance
# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet, AccountClient
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import MemcmpOpts
from driftpy.drift_client import DriftClient
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.drift_user import DriftUser, get_token_amount
from datafetch.transaction_fetch import transaction_history_for_account, load_token_balance

from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.constants.spot_markets import mainnet_spot_market_configs

import os
import json
import streamlit as st
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market, all_user_stats, DRIFT_WHALE_LIST_SNAP
from anchorpy import EventParser
import asyncio
from driftpy.math.margin import MarginCategory
import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import InsuranceFundStakeAccount, SpotMarketAccount
from driftpy.addresses import * 
import time
from driftpy.user_map.user_map import UserMap, UserMapConfig, PollingConfig
import datetime
import csv

NUMBER_OF_SPOT = 13
NUMBER_OF_PERP = 28

@st.cache_data(ttl=1800)
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(quoting=csv.QUOTE_NONE).encode('utf-8')

# @st.cache_resource
async def load_user_map(_drift_client, user_map_settings):
    start_time = time.time()
    st.write("cached usermap at:", int(start_time), datetime.datetime.fromtimestamp(start_time))
    include_idle = True if user_map_settings in ['all', 'idle']  else False
    user_map = UserMap(UserMapConfig(_drift_client, PollingConfig(0), skip_initial_load=False, include_idle=include_idle))
    if user_map_settings == 'all':
        await user_map.subscribe()
    elif user_map_settings == 'active':
        await user_map.subscribe()
    elif user_map_settings == 'idle':
        await user_map.subscribe()
    elif user_map_settings == 'whales':
        whales = DRIFT_WHALE_LIST_SNAP
        for x in whales.split('\n'):
            if x.strip(' '):
                await user_map.must_get(x.strip(' '))
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write("loaded usermap in", int(elapsed_time), ' seconds')
    return user_map

def comb_asset_liab(a_l_tup):
    return a_l_tup[0] - a_l_tup[1]

def get_collateral_composition(x: DriftUser, margin_category, n):
    # ua = x.get_user_account()
    net_v = {i: comb_asset_liab(x.get_spot_market_asset_and_liability_value(i, margin_category))/QUOTE_PRECISION for i in range(n)}
    return net_v 

def get_perp_liab_composition(x: DriftUser, margin_category, n):
    # ua = x.get_user_account()
    net_p = {i: x.get_perp_market_liability(i, margin_category, signed=True)/QUOTE_PRECISION for i in range(n)}
    return net_p 

def get_perp_lp_share_composition(x: DriftUser, n):
    # ua = x.get_user_account()
    def get_lp_shares(x, i):
        res = x.get_perp_position(i)
        if res is not None:
            return res.lp_shares/1e9
        else:
            return 0
    net_p = {i: get_lp_shares(x, i) for i in range(n)}
    return net_p 

async def get_usermap_df(_drift_client, user_map_settings, mode, oracle_distor=.1, only_one_index=None, cov_matrix=None):
    perp_n = NUMBER_OF_PERP
    spot_n = NUMBER_OF_SPOT

    def do_dict(x: DriftUser, margin_category: MarginCategory, oracle_cache=None):
        if oracle_cache is not None:
            x.drift_client.account_subscriber.cache = oracle_cache

        user_account = x.get_user_account()
        levs0 = {
        'tokens': [x.get_token_amount(i) for i in range(spot_n)],
        'leverage': x.get_leverage() / MARGIN_PRECISION, 
        'perp_liability': x.get_perp_market_liability(None, margin_category) / QUOTE_PRECISION,
        'spot_asset': x.get_spot_market_asset_value(None, margin_category) / QUOTE_PRECISION,
        'spot_liability': x.get_spot_market_liability_value(None, margin_category) / QUOTE_PRECISION,
        'upnl': x.get_unrealized_pnl(True) / QUOTE_PRECISION,
        'funding_upnl': x.get_unrealized_funding_pnl() / QUOTE_PRECISION,
        'total_collateral': x.get_total_collateral(margin_category or MarginCategory.INITIAL) / QUOTE_PRECISION,
        'margin_req': x.get_margin_requirement(margin_category or MarginCategory.INITIAL) / QUOTE_PRECISION,
        'net_v': get_collateral_composition(x, margin_category, spot_n),
        'net_p': get_perp_liab_composition(x, margin_category, perp_n),
        'net_lp': get_perp_lp_share_composition(x, perp_n),
        'last_active_slot': user_account.last_active_slot,
        'cumulative_perp_funding': user_account.cumulative_perp_funding/QUOTE_PRECISION,
        'settled_perp_pnl': user_account.settled_perp_pnl/QUOTE_PRECISION,
        'name': bytes(user_account.name).decode('utf-8',  errors='ignore').strip(),
        'authority': str(user_account.authority),
        'has_open_order': user_account.has_open_order,
        'sub_account_id': user_account.sub_account_id,
        'next_liquidation_id': user_account.next_liquidation_id,
        'cumulative_spot_fees': user_account.cumulative_spot_fees,
        'total_deposits': user_account.total_deposits,
        'total_withdraws': user_account.total_withdraws,
        'total_social_loss': user_account.total_social_loss,
        'unsettled_pnl_perp_x': x.get_unrealized_pnl(True, market_index=24) / QUOTE_PRECISION,
        }
        levs0['net_usd_value'] = levs0['spot_asset'] + levs0['upnl'] - levs0['spot_liability']
        return levs0
    user_map_result: UserMap = await load_user_map(_drift_client, user_map_settings)
    
    user_keys = list(user_map_result.user_map.keys())
    user_vals = list(user_map_result.values())
    if cov_matrix == 'ignore stables':
        skipped_oracles = [str(x.oracle) for x in mainnet_spot_market_configs if 'USD' in x.symbol]
    elif cov_matrix == 'sol + lst only':
        skipped_oracles = [str(x.oracle) for x in mainnet_spot_market_configs if 'SOL' not in x.symbol]
    elif cov_matrix == 'sol lst only':
        skipped_oracles = [str(x.oracle) for x in mainnet_spot_market_configs if x.symbol not in ['mSOL', 'jitoSOL', 'bSOL']]
    elif cov_matrix == 'sol ecosystem only':
        skipped_oracles = [str(x.oracle) for x in mainnet_spot_market_configs if x.symbol not in ['PYTH', 'JTO']]
    elif cov_matrix == 'wrapped only':
        skipped_oracles = [str(x.oracle) for x in mainnet_spot_market_configs if x.symbol not in ['wBTC', 'wETH']]
    elif cov_matrix == 'stables only':
        skipped_oracles = [str(x.oracle) for x in mainnet_spot_market_configs if 'USD' not in x.symbol]

    if only_one_index is None or len(only_one_index) > 12:
        only_one_index_key = only_one_index
    else:
        only_one_index_key = ([str(x.oracle) for x in mainnet_perp_market_configs if x.base_asset_symbol == only_one_index] \
         +[str(x.oracle) for x in mainnet_spot_market_configs if x.symbol == only_one_index])[0]



    if mode == 'margins':
        levs_none = list(do_dict(x, None) for x in user_vals)
        levs_init = list(do_dict(x, MarginCategory.INITIAL) for x in user_vals)
        levs_maint = list(do_dict(x, MarginCategory.MAINTENANCE) for x in user_vals)
        # print(levs_none[0].keys(), levs_init[0].keys(), levs_maint[0].keys())
        return (levs_none, levs_init, levs_maint), user_keys
    else:
        num_entrs = 3
        new_oracles_dat_up = [{}, {}, {}]

        new_oracles_dat_down = [{}, {}, {}]

        assert(len(new_oracles_dat_down) == num_entrs)
        await _drift_client.account_subscriber.update_cache()
        cache_up = copy.deepcopy(_drift_client.account_subscriber.cache)
        cache_down = copy.deepcopy(_drift_client.account_subscriber.cache)
        for i,(key, val) in enumerate(_drift_client.account_subscriber.cache['oracle_price_data'].items()):
            for i in range(num_entrs):
                new_oracles_dat_up[i][key] = copy.deepcopy(val)
                new_oracles_dat_down[i][key] = copy.deepcopy(val)
            if cov_matrix is not None and key in skipped_oracles:
                continue
            if only_one_index is None or only_one_index_key == key:
                for i in range(num_entrs):
                    oracle_distort_up = max(1 + oracle_distor * (i+1), 1)
                    oracle_distort_down = max(1 - oracle_distor * (i+1), 0)

                    new_oracles_dat_up[i][key].data.price *= oracle_distort_up
                    new_oracles_dat_down[i][key].data.price *= oracle_distort_down

        levs_none = list(do_dict(x, None, None) for x in user_vals)
        levs_up = []
        levs_down = []

        for i in range(num_entrs):
            # print(new_oracles_dat_up[i])
            cache_up['oracle_price_data'] = new_oracles_dat_up[i]
            cache_down['oracle_price_data'] = new_oracles_dat_down[i]
            levs_up_i = list(do_dict(x, None, cache_up) for x in user_vals)
            levs_down_i = list(do_dict(x, None, cache_down) for x in user_vals)
            levs_up.append(levs_up_i)
            levs_down.append(levs_down_i)

        # print(levs_none[0].keys(), levs_init[0].keys(), levs_maint[0].keys())
        return (levs_none, tuple(levs_up), tuple(levs_down)), user_keys

@st.cache_data
def cached_get_usermap_df(_drift_client, user_map_settings, mode, oracle_distort, only_one_index, cov_matrix):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(get_usermap_df(_drift_client, user_map_settings, mode, oracle_distort, only_one_index, cov_matrix))

async def get_protocol_summary(drift_client: DriftClient):
        await drift_client.account_subscriber.update_cache()
        state = drift_client.get_state_account()
        aa = 0
        vamm = 0
        usdc_market = None
        spot_acct_dep = 0
        spot_acct_bor = 0
        states = {}
        vaults = {}

        for market_index in range(state.number_of_spot_markets):
            market = drift_client.get_spot_market_account(market_index)
            if market_index == 0:
                usdc_market = market
            # conn = drift_client.program.provider.connection
            # ivault_pk = market.insurance_fund.vault
            svault_pk = market.vault
            sv_amount = await load_token_balance(drift_client.program.provider.connection, svault_pk)
            sv_amount /= (10**market.decimals)
            px1 = ((drift_client.get_oracle_price_data_for_spot_market(market_index)).price/1e6)
            sdep = (market.deposit_balance * market.cumulative_deposit_interest/1e10)/(1e9)
            sbor = (market.borrow_balance * market.cumulative_borrow_interest/1e10)/(1e9)
            spot_acct_dep += sdep*px1
            spot_acct_bor -= sbor*px1
            states['spot'+str(market_index)] = sdep-sbor
            vaults['spot'+str(market_index)] = sv_amount
            sv_amount *= px1
            aa += sv_amount
        

        vamm_upnl = 0 
        for market_index in range(state.number_of_markets):
            market = drift_client.get_perp_market_account(market_index)
            px1 = ((drift_client.get_oracle_price_data_for_perp_market(market_index)).price/1e6)
            fee_pool = (market.amm.fee_pool.scaled_balance * usdc_market.cumulative_deposit_interest/1e10)/(1e9)
            pnl_pool = (market.pnl_pool.scaled_balance * usdc_market.cumulative_deposit_interest/1e10)/(1e9)
            vamm += pnl_pool + fee_pool
            vamm_upnl -= market.amm.quote_asset_amount/1e6 + (market.amm.base_asset_amount_with_amm + market.amm.base_asset_amount_with_unsettled_lp)/1e9 * px1


        return (vamm, vamm_upnl, aa, vaults, states, spot_acct_dep, spot_acct_bor, state.number_of_sub_accounts)


@st.cache_data
def cached_get_protocol_summary(_drift_client):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(get_protocol_summary(_drift_client))

def update_drift_cache(drift_client):
    async def upcache(drift_client):
        if drift_client.account_subscriber.cache is None:
            await drift_client.account_subscriber.update_cache()

    loop = asyncio.new_event_loop()
    return loop.run_until_complete(upcache(drift_client))

def usermap_page(drift_client: DriftClient, env):
    s1, s11, s2, s3, s4 = st.columns(5)
    # await drift_client.account_subscriber.update_cache();
    perp_market_inspect = s1.selectbox('perp market:', list(range(24)))
    user_map_settings = s11.radio('user map settings:', ['all', 'active', 'idle', 'whales'], index=1)
    mode = s2.radio('mode:', ['oracle_distort', 'margin_cat'])
    mr_var = None
    cov_matrix = None
    only_one_index = None
    oracle_distort = None
    if mode == 'margin':
        kk = s3.radio('margin ratio:', [None, 'initial', 'maint'])
    else:    
        oracle_distort = s4.slider(
            'Select an oracle distort step',
            0.0, 1.0, .1, step=.05, help='3 intervals of the this step will be used for price scenario analysis')

        all_symbols = sorted(list(set([x.base_asset_symbol for x in mainnet_perp_market_configs] 
                                      + [x.symbol.replace('w', '') for x in mainnet_spot_market_configs])))
        only_one_index = s4.selectbox('only single oracle:', 
                                      ([None]+all_symbols 
                                    #    + list(drift_client.account_subscriber.cache['oracle_price_data'].keys())
                                       ),
                                      index=0,
                                      help='select a single oracle to distort, otherwise it falls to distort settings'
                                      )

        cov_matrix = s4.radio('distort settings:', [None, 
                                                    'ignore stables', 
                                                    'sol + lst only', 
                                                    'sol lst only', 
                                                    'sol ecosystem only',
                                                    'wrapped only',
                                                    'stables only'
                                                    ], index=1)
        


    levs, user_keys = cached_get_usermap_df(drift_client, user_map_settings, mode, oracle_distort, only_one_index, cov_matrix)
    if mode == 'margin':
        mr_var = s3.radio('margin ratio:', [None, 'initial', 'maint'])
        if mr_var is None:
            the_lev = levs[0]
        elif mr_var == 'initial':
            the_lev = levs[1]
        else:
            the_lev = levs[2]
    else:
        mr_var = s3.radio('oracle distortion:', [None, 'up', 'down'])
        if mr_var is None:
            the_lev = levs[0]
        elif mr_var == 'up':
            the_lev = levs[1][0]
        else:
            the_lev = levs[2][0]


    df = pd.DataFrame(the_lev)
    df.index = user_keys
    df = df.reset_index()

    csv2 = convert_df(df[['index', 'authority']])
    st.download_button(
        label="Download authority/user map data as CSV",
        data=csv2,
        file_name='user_authority_map_'+datetime.datetime.now().strftime("%Y%m%d")+'.csv',
        mime='text/csv',
    )
    try:
        csv = convert_df(df)
        st.download_button(
            label="Download snapshot data as CSV",
            data=csv,
            file_name='user_snapshot_'+datetime.datetime.now().strftime("%Y%m%d")+'.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.warning('ERROR, cannot create csv: ' + str(e))

    update_drift_cache(drift_client)
    (vamm, vamm_upnl, aa, spot_vaults, spot_states, spot_acct_dep, spot_acct_bor, num_subs) = cached_get_protocol_summary(drift_client)

    tabs = st.tabs(['summary', 'price scenario bankruptcies', 'leverage', 'spot0', 'scatter', 'raw tables',
                    'account profiles'])
    with tabs[0]:
        user_usdc_total = df['tokens'].apply(lambda x: x[0]).sum()/1e6
        usdc_excess = spot_vaults['spot0'] - user_usdc_total
        col1, col11, col2, col3, col4 = st.columns(5)
        col1.metric(f'{len(df)}/{num_subs} of users tvl', 
                    f'${df["net_usd_value"].sum():,.2f}', 
                    f'${user_usdc_total:,.2f} USDC total (excess={usdc_excess})')
        col11.metric('state tvl', f'${spot_acct_dep+spot_acct_bor:,.2f}', 
                     f'{-spot_acct_bor/spot_acct_dep * 100:,.2f}% utilization')
        col2.metric('vault tvl', f'${aa:,.2f}', f'{aa-(spot_acct_dep+spot_acct_bor):,.2f} excess vs state')
        col3.metric('vamm tvl', f'${vamm+vamm_upnl:,.2f}', f'${vamm:,.2f} tvl + {vamm_upnl:,.2f} upnl')
        vamm2 = spot_acct_dep+spot_acct_bor-df["net_usd_value"].sum()
        col4.metric('vamm2 tvl', f'${vamm2:,.2f}', f'{vamm2-(vamm+vamm_upnl):,.2f} excess')


        s1,s2,s3 = st.columns(3)
        s1.metric('total users:', f'{len(df)} / {num_subs}')
        s2.metric('total usd value:', f'${df["net_usd_value"].sum():,.2f}')
        bankrupts = df[df["net_usd_value"] < 0]
        s3.metric('bankruptcies:', f'${bankrupts["net_usd_value"].sum():,.2f}', f'{len(bankrupts)} unique accounts')
        s3.dataframe(bankrupts)
        st.write(df['tokens'])
        recon_tokens = [df['tokens'].apply(lambda x: x[i]).sum()/(10 ** drift_client.get_spot_market_account(i).decimals)
                         for i in range(len(spot_vaults.keys()))]
        net_v2 = [df['net_v'].apply(lambda x: x[i]).sum()
                         for i in range(len(spot_vaults.keys()))]
        aas2 = [spot_vaults['spot'+str(i)]
                         for i in range(len(spot_vaults.keys()))]
        states2 = [spot_states['spot'+str(i)]
                         for i in range(len(spot_vaults.keys()))]
        recon_df = pd.DataFrame([recon_tokens, net_v2, aas2, states2], index=['user_tokens', 'user_value', 'vault_tokens', 'state_tokens']).T

        recon_df['excess_token'] = recon_df['vault_tokens'] - recon_df['user_tokens']
        recon_df['excess_token2'] = recon_df['state_tokens'] - recon_df['user_tokens']
        recon_df.loc[0, 'upnl_user'] = df['upnl'].sum()
        recon_df.loc[0, 'upnl_state'] = vamm_upnl
        s1.write(recon_df)


        value_of_users = df[['net_usd_value']].sort_values(by='net_usd_value').reset_index(drop=True)
        value_of_users['cumulative_usd_value'] = value_of_users['net_usd_value'].cumsum()
        st.plotly_chart(value_of_users.plot())

        st.write('unsettled funding pnl:', df['funding_upnl'].sum())



    def get_rattt(row):
        # st.write(row)
        df1 = pd.Series([val/row['spot_asset'] * (row['perp_liability']+row['spot_liability']) 
                        if val > 0 else 0 for key,val in row['net_v'].items()]
                        )
        df1.index = ['spot_'+str(x)+'_all' for x in df1.index]

        df2 = pd.Series([val/(row['spot_asset']) * (row['perp_liability']) 
                        if val > 0 else 0 for key,val in row['net_v'].items()]
                        )
        df2.index = ['spot_'+str(x)+'_all_perp' for x in df2.index]

        df3 = pd.Series([val/(row['spot_asset']) * (row['spot_liability']) 
                        if val > 0 else 0 for key,val in row['net_v'].items()]
                        )
        df3.index = ['spot_'+str(x)+'_all_spot' for x in df3.index]
        
        df4 = pd.Series([val/(row['spot_asset']) * (row['net_p'][perp_market_inspect]) 
                        if val > 0 and row['net_p'][0] > 0 else 0 for key,val in row['net_v'].items()]
                        )
        df4.index = ['spot_'+str(x)+'_perp_'+str(perp_market_inspect)+'_long' for x in df4.index]

        df5 = pd.Series([val/(row['spot_asset']) * (row['net_p'][perp_market_inspect]) 
                        if val > 0 and row['net_p'][perp_market_inspect] < 0 else 0 for key,val in row['net_v'].items()]
                        )
        df5.index = ['spot_'+str(x)+'_perp_'+str(perp_market_inspect)+'_short' for x in df5.index]
        
        dffin = pd.concat([
            df1,
            df2,
            df3,
            df4,
            df5,
        ])
        return dffin
    df = pd.concat([df, df.apply(get_rattt, axis=1)],axis=1)
    res = pd.DataFrame({('spot'+str(i)): (df["spot_"+str(i)+'_all'].sum(), 
                                        df["spot_"+str(i)+'_all_spot'].sum(),
                                        df["spot_"+str(i)+'_all_perp'].sum() ,
                                        df["spot_"+str(i)+'_perp_'+str(perp_market_inspect)+'_long'].sum(),
                                        df["spot_"+str(i)+'_perp_'+str(perp_market_inspect)+'_short'].sum())
                                        for i in range(NUMBER_OF_SPOT)},
                                        
                    
                    index=['all_liabilities', 'all_spot', 'all_perp', 'perp_0_long', 'perp_0_short'])




    with tabs[1]:
        dfs = [pd.DataFrame(levs[2][i]) for i in range(len(levs[2]))] + [pd.DataFrame(levs[0])] + [pd.DataFrame(levs[1][i]) for i in range(len(levs[1]))]
        xdf = [[-df[df['net_usd_value']<0]['net_usd_value'].sum() for df in dfs],
               [-(df[(df['spot_asset']<df['spot_liability']) & (df['net_usd_value']<0)]['net_usd_value'].sum()) for df in dfs]
            ]
        toplt_fig = pd.DataFrame(xdf, 
                                 index=['bankruptcy', 'spot bankrupt'],
                                 columns=[oracle_distort*(i+1)*-100 for i in range(len(levs[2]))]\
                                 +[0]\
                                 +[oracle_distort*(i+1)*100 for i in range(len(levs[1]))]).T
        toplt_fig['perp bankrupt'] = toplt_fig['bankruptcy'] - toplt_fig['spot bankrupt']
        toplt_fig = toplt_fig.sort_index()
        toplt_fig = toplt_fig.plot()
         # Customize the layout if needed
        toplt_fig.update_layout(title='Bankruptcies in crypto price scenarios',
                        xaxis_title='Oracle Move (%)',
                        yaxis_title='Bankruptcy ($)')
        st.plotly_chart(toplt_fig)

        st.write(df[df['spot_asset']<df['spot_liability']])

    with tabs[2]:
        st.plotly_chart(df['leverage'].plot(kind='hist'))
        st.write(df.replace(0, np.nan).describe())
        lev_one_market = (df['net_p'].apply(lambda x: abs(x[perp_market_inspect]) if x[perp_market_inspect] != 0 else np.nan)/(df['spot_asset']-df['spot_liability']+df['upnl']))
        st.write(lev_one_market.describe())
    with tabs[3]:
        spot_market_inspect = st.selectbox('spot market:', list(range(NUMBER_OF_SPOT)))
        n = 0
        for num,val in enumerate(df.columns):
            if "spot_"+str(spot_market_inspect)+'_all' == val:
                n = num
        col = st.selectbox('column:', df.columns, index=n)
    with tabs[4]:
        # df[].plot
        # import plotly.graph_objects as go
        # fig = go.Figure()
        import plotly.express as px
        fig = px.scatter(df, x='leverage', y='spot_asset', size=col, hover_data=['index', 'leverage', col, 'spot_asset', 'spot_liability'])

        # Customize the layout if needed
        fig.update_layout(title='Bubble Plot of Size and SpotIi',
                        xaxis_title='Size',
                        yaxis_title='SpotIi')
        st.plotly_chart(fig)
    with tabs[5]:
        st.dataframe(res)
        st.dataframe(df)
        df2download = df[['index', 'authority']].to_csv(escapechar='"').encode('utf-8')
        st.download_button(
            label="share browser statistics [bytes="+str(len(df))+"]",
            data=df2download,
            file_name='usermap_snapshot.csv',
            mime='text/csv',
        )
        df2download2 = df.to_csv(escapechar='"').encode('utf-8')
        st.download_button(
            label="share more browser statistics [bytes="+str(len(df))+"]",
            data=df2download2,
            file_name='usermap_full_snapshot.csv',
            mime='text/csv',
        )

        # st.write(df[['index', 'authority', 'cumulative_perp_funding', 'last_active_slot', 'settled_perp_pnl', 'next_liquidation_id', 'total_deposits', 'total_withdraws', 'total_social_loss']])
        # st.download_button(
        #     label="share browser statistics [bytes="+str(len(df))+"]",
        #     data=df2download,
        #     file_name='usermap_snapshot.csv',
        #     mime='text/csv',
        # )
    with tabs[6]:
        df['ever_liquidated'] = (df['next_liquidation_id']>1)
        name_res = df.groupby('name').agg({'leverage':'median', 
                                'perp_liability':'sum', 'net_usd_value':'sum',
                                'settled_perp_pnl':'sum', 'authority':'count',
                                'ever_liquidated':'mean',
                                }).sort_values('authority', ascending=False)
        
        st.dataframe(name_res)
        # dlp_df = df[df.name=='Drift Liquidity Provider']

        dlp_df = df[df.net_lp.apply(lambda x: sum(list(x.values()))!=0)]
        st.write(f'current LPers: {len(dlp_df)}/{ len(df)} users')
        st.write(dlp_df)
        st.write(dlp_df.describe())

        perp_market_inspect_dlp = st.selectbox('perp market:', list(range(NUMBER_OF_PERP)), 
                                               index=10,
                                               
                                               key='tab-5')
        st.write('dlp with positions in perp market=', perp_market_inspect_dlp)

        dlp_one_market = dlp_df[dlp_df.net_lp.apply(lambda x: x[perp_market_inspect_dlp]!=0)]
        st.write(dlp_one_market)

        
        st.write('total lp users:', dlp_df.net_lp.apply(lambda x: x[perp_market_inspect_dlp]!=0).sum())
        st.write('total user lp shares:', dlp_df.net_lp.apply(lambda x: x[perp_market_inspect_dlp]).sum())

        dlp_market_account = drift_client.get_perp_market_account(perp_market_inspect_dlp)
        st.write('total lp shares (market)', dlp_market_account.amm.user_lp_shares/1e9)
        st.write(dlp_market_account)
