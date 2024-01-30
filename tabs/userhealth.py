
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
from driftpy.accounts.oracle import *

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient, AccountSubscriptionConfig
from driftpy.drift_user import DriftUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.drift_user import get_token_amount
import os
import json
import streamlit as st
from driftpy.types import MarginRequirementType
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from driftpy.addresses import *
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import time
from enum import Enum
from driftpy.math.margin import MarginCategory, calculate_asset_weight

@st.cache_data
def get_loaded_auths():

    query_p = st.experimental_get_query_params()
    frens = query_p.get('authority', [])
    if frens != []:
        frens = frens[0].split(',')
    return frens

# def ccc(vals):
#     st.experimental_set_query_params(**{'authority': vals, 'tab':'User-Health'})

async def show_user_health(clearing_house: DriftClient):
    frens = get_loaded_auths()

    # st.write('query string:', frens)

    state = await get_state_account(clearing_house.program)
    ch = clearing_house

    every_user_stats = await ch.program.account['UserStats'].all()
    authorities = sorted([str(x.account.authority) for x in every_user_stats])

    if len(frens)==0:
        user_authorities = st.selectbox(
            'user authorities', 
            authorities, 
            0
            # on_change=ccc
            # frens
        )
        user_authorities = [user_authorities]
    else:
        user_authorities = frens
    # print(user_authorities)
    # user_authority = user_authorities[0]
    if len(user_authorities):

        # await chu.set_cache()
        # cache = chu.CACHE

        # user_stats_pk = get_user_stats_account_public_key(ch.program_id, user_authority_pk)
        # all_user_stats = await ch.program.account['UserStats'].fetch(user_stats_pk)
        user_stats = [x.account for x in every_user_stats if str(x.account.authority) in user_authorities]
        all_summarys = []
        balances = []
        positions = []

        url = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/'
        url += 'user/%s/trades/%s/%s'

        userAccountKeys = []
        for user_authority in user_authorities:
                user_authority_pk = Pubkey.from_string(user_authority)
                # print(user_stats)
                user_stat = [x for x in user_stats if str(x.authority) == user_authority][0]
                # chu = DriftUser(
                #     ch, 
                #     authority=user_authority_pk, 
                #     sub_account_id=0, 
                #     use_cache=False
                # )
                for sub_id in range(user_stat.number_of_sub_accounts_created):    
                    user_account_pk = get_user_account_public_key(clearing_house.program_id,
                    user_authority_pk,
                    sub_id)
                    userAccountKeys.append(user_account_pk)

        st.write('Authority owned Drift User Accounts:',)
        uak_df = pd.DataFrame(userAccountKeys, columns=['userAccountPublicKey'])
        uak_df.index = ['subaccount_'+str(x) for x in uak_df.index]
        st.dataframe(uak_df)

        tabs = st.tabs(['health', 'recent trades'])
        for user_authority in user_authorities:
            user_authority_pk = Pubkey.from_string(user_authority)
            # print(user_stats)
            user_stat = [x for x in user_stats if str(x.authority) == user_authority][0]
            # st.write(user_stat)
            # chu = DriftUser(
            #     ch, 
            #     authority=user_authority_pk, 
            #     sub_account_id=0, 
            #     use_cache=False
            # )
            for sub_id in range(user_stat.number_of_sub_accounts_created):
                print('sub_id:', sub_id)
                user_account_pk = get_user_account_public_key(ch.program.program_id,
                    user_authority_pk,
                    sub_id)
                chu_sub = DriftUser(
                    ch, 
                    user_public_key=user_account_pk, 
                    account_subscription=AccountSubscriptionConfig("cached")
                    # use_cache=True
                )
                chu_sub.account_subscriber.update_cache()
                CACHE = None   
                # try:
                #     await chu_sub.set_cache(CACHE)
                # except Exception as e:
                #     print(e, 'fail')
                #     continue

              


                url2 = url % (str(user_account_pk), '2023', '7')

                with tabs[1]:            
                    st.header('Recent Trades Stats')
                    st.write('data source:', url2)
                if sub_id==0:
                    # try:
                    df = pd.read_csv(url2)
                    df = df.drop_duplicates(['fillerReward', 'baseAssetAmountFilled', 'quoteAssetAmountFilled',
                                            'takerPnl', 'makerPnl', 'takerFee', 'makerRebate', 'refereeDiscount',
                                            'quoteAssetAmountSurplus', 'takerOrderBaseAssetAmount',
                                            'takerOrderCumulativeBaseAssetAmountFilled',
                                            'takerOrderCumulativeQuoteAssetAmountFilled', 'takerOrderFee',
                                            'makerOrderBaseAssetAmount',
                                            'makerOrderCumulativeBaseAssetAmountFilled',
                                            'makerOrderCumulativeQuoteAssetAmountFilled', 'makerOrderFee',
                                            'oraclePrice', 'makerFee', 'txSig', 'slot', 'ts', 'action',
                                            'actionExplanation', 'marketIndex', 'marketType', 'filler',
                                            'fillRecordId', 'taker', 'takerOrderId', 'takerOrderDirection', 'maker',
                                            'makerOrderId', 'makerOrderDirection', 'spotFulfillmentMethodFee',
                                            ]).reset_index(drop=True)
                    df['baseAssetAmountSignedFilled'] = df['baseAssetAmountFilled'] \
                        * df['takerOrderDirection'].apply(lambda x: 2*(x=='long')-1) \
                        * df['taker'].apply(lambda x: 2*(str(x)==user_account_pk)-1)
                    # st.multiselect('columns:', df.columns, None)
                    ot = df.pivot_table(index='slot', columns=['marketType', 'marketIndex'], values='baseAssetAmountSignedFilled', aggfunc='sum').cumsum().ffill()
                    ot.columns = [str(x) for x in ot.columns]

                    tt = df.groupby(['takerOrderId', 'takerOrderDirection']).count().iloc[:,0].reset_index()
                    tt1 = tt.groupby('takerOrderDirection').count()

                    trades = df.groupby(['marketType', 'marketIndex']).sum()[['quoteAssetAmountFilled']]
                    trades.columns = ['volume']

                    with tabs[1]:
                        st.plotly_chart(ot.plot(title='base asset amount over month (WIP)'))
                        st.dataframe(tt1)
                        st.dataframe(trades)

                    # except:
                    #     st.write('cannot load data')

                summary = {}
                summary['authority'] = user_authority+'-'+str(sub_id)
                summary['total_collateral'] = (await chu_sub.get_total_collateral())/1e6
                summary['init total_collateral'] = (await chu_sub.get_total_collateral(MarginCategory.INITIAL))/1e6
                summary['maint total_collateral'] = (await chu_sub.get_total_collateral(MarginCategory.MAINTENANCE))/1e6
                summary['initial_margin_req'] = (await chu_sub.get_margin_requirement(MarginCategory.INITIAL, None, True, False))/1e6
                summary['maintenance_margin_req'] = (await chu_sub.get_margin_requirement(MarginCategory.MAINTENANCE, None, True, False))/1e6
                all_summarys.append(pd.DataFrame(summary, index=[sub_id]))

                for spot_market_index in range(state.number_of_spot_markets):
                    spot_pos = await chu_sub.get_user_spot_position(spot_market_index)
                    if spot_pos is not None:
                        spot = await chu_sub.get_spot_market(spot_market_index)
                        tokens = get_token_amount(
                            spot_pos.scaled_balance,
                            spot, 
                            str(spot_pos.balance_type)
                        )
                        market_name = ''.join(map(chr, spot.name)).strip(" ")

                        dd = {
                        'authority': user_authority+'-'+str(sub_id),
                        'name': market_name,
                        'tokens': tokens/(10**spot.decimals),
                        'net usd value': await chu_sub.get_spot_market_asset_value(None, False, spot_market_index)/1e6,
                        'asset weight': calculate_asset_weight(tokens, spot, None)/1e4,
                        'initial asset weight': calculate_asset_weight(tokens, spot, MarginCategory.INITIAL)/1e4,
                        'maint asset weight': spot.maintenance_asset_weight/1e4,
                        'your maint asset weight': calculate_asset_weight(tokens, spot, MarginCategory.MAINTENANCE)/1e4,
                        'weighted usd value': await chu_sub.get_spot_market_asset_value(MarginCategory.INITIAL, False, spot_market_index)/1e6,
                        'maint weighted usd value': await chu_sub.get_spot_market_asset_value(MarginCategory.MAINTENANCE, False, spot_market_index)/1e6,
                        'liq price': await chu_sub.get_spot_liq_price(spot_market_index)
                        }
                        balances.append(dd)


                for perp_market_index in range(state.number_of_markets):
                    pos = await chu_sub.get_user_position(perp_market_index)
                    if pos is not None:
                        dd = pos.__dict__
                        # if pos.base_asset_amount != 0:
                            # perp_market = await chu_sub.get_perp_market(pos.market_index)
                            # try:
                            #     oracle_price = (await chu_sub.get_perp_oracle_data(perp_market)).price / PRICE_PRECISION
                            # except:
                            #     oracle_price = perp_market.amm.historical_oracle_data.last_oracle_price / PRICE_PRECISION
                            # liq_price = await chu_sub.get_perp_liq_price(pos.market_index)
                            # if liq_price is None: 
                                # liq_delta = None
                            # else:
                            #     perp_liq_prices[pos.market_index] = perp_liq_prices.get(pos.market_index, []) + [liq_price]
                            #     liq_delta = liq_price - oracle_price
                            #     perp_liq_deltas[pos.market_index] = perp_liq_deltas.get(pos.market_index, []) + [(liq_delta, abs(pos.base_asset_amount), leverage/10_000)]

                        # else: 
                        #     liq_delta = None

                        # dd['liq_price_delta'] = liq_delta

                        perp = await chu_sub.get_perp_market(perp_market_index)
                        market_name = ''.join(map(chr, perp.name)).strip(" ")

                        dd2 = {
                        'authority': user_authority+'-'+str(sub_id),
                        'name': market_name,
                        'base': pos.base_asset_amount/1e9,
                        'liq price': await chu_sub.get_perp_liq_price(perp_market_index)
                        }
                        z = {**dd, **dd2}

                        positions.append(z)

        with tabs[0]:
            st.header('Account Health')
            if len(all_summarys)>0:
                sub_dfs = pd.concat(all_summarys)

                st.metric('total account value:', '$'+str(int(sub_dfs['total_collateral'].sum()*100)/100))

                sub_dfs.index.name = 'subaccount_id'
                st.markdown('summary')

                st.dataframe(sub_dfs)
                
            st.markdown('assets/liabilities')
            spotcol, perpcol = st.columns([1,3])
            bal1 = pd.DataFrame(balances).T
            bal1.columns = ['spot'+str(x) for x in bal1.columns]
            df1 = pd.DataFrame(positions)
            if len(df1):
                df1['base_asset_amount'] /= 1e9
                df1['remainder_base_asset_amount'] /= 1e9
                df1['open_bids'] /= 1e9
                df1['open_asks'] /= 1e9
                df1['settled_pnl'] /= 1e6

                df1['lp_shares'] /= 1e9
                df1['quote_asset_amount'] /= 1e6
                df1['quote_entry_amount'] /= 1e6
                df1['quote_break_even_amount'] /= 1e6

                df1['entry_price'] = -df1['quote_entry_amount']/df1['base_asset_amount'].apply(lambda x: 1 if x==0 else x)
                df1['breakeven_price'] = -df1['quote_break_even_amount']/df1['base_asset_amount'].apply(lambda x: 1 if x==0 else x)
                df1['cost_basis'] = -df1['quote_asset_amount']/df1['base_asset_amount'].apply(lambda x: -1 if x==0 else x)

            pos1 = df1.T
            pos1.columns = ['perp'+str(x) for x in pos1.columns]
            spotcol.dataframe(bal1, use_container_width=True)
            perpcol.dataframe(pos1, use_container_width=True)

            with st.expander('user stats'):
                st.dataframe(pd.DataFrame([x for x in user_stats]).T)


        # else:
        #     st.text('not found')

        
        