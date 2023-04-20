
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
from driftpy.math.oracle import *
import datetime

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.clearing_house_user import get_token_amount
import os
import json
import streamlit as st
from driftpy.types import MarginRequirementType
from driftpy.constants.banks import devnet_banks, Bank
from driftpy.constants.markets import devnet_markets, Market
from driftpy.addresses import *
from dataclasses import dataclass
from solana.publickey import PublicKey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import time
from enum import Enum
from driftpy.math.margin import MarginCategory, calculate_asset_weight

@st.experimental_memo
def get_loaded_auths():

    query_p = st.experimental_get_query_params()
    frens = query_p.get('authority', [])
    if frens != []:
        frens = frens[0].split(',')
    return frens

# def ccc(vals):
#     st.experimental_set_query_params(**{'authority': vals, 'tab':'User-Health'})

def load_user_settlepnl(dates, user_key, with_urls=False):
    url = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/'
    url += 'user/%s/settlePnls/%s/%s'
    dfs = []
    data_urls = []
    for date in dates:
        (year, month, _) = (date.strftime("%Y"), str(date.month), str(date.day))
        data_url = url % (user_key, year, month)
        if data_url not in data_urls:
            data_urls.append(data_url)
            dfs.append(pd.read_csv(data_url))
    dfs = pd.concat(dfs)

    if with_urls:
        return dfs, data_urls

    return dfs

def load_user_trades(dates, user_key, with_urls=False):
    url_og = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/'
    url = url_og + 'user/%s/trades/%s/%s'

    liq_url = url_og + 'user/%s/liquidations/%s/%s'

    dfs = []
    data_urls = []
    for date in dates:
        (year, month, _) = (date.strftime("%Y"), str(date.month), str(date.day))
        data_url = url % (user_key, year, month)
        if data_url not in data_urls:
            # try:
                dd = pd.read_csv(data_url)
                data_urls.append(data_url)
                if 'liquidation' in dd['actionExplanation'].unique():
                    data_liq_url = liq_url % (user_key, year, month)
                    data_urls.append(data_liq_url)
                    dd1 = pd.read_csv(data_liq_url)
                    dd = dd.merge(dd1, suffixes=('', '_l'), how='outer', on='txSig')
                dfs.append(dd)
            # except:
            #     pass
    dfs = pd.concat(dfs)

    if with_urls:
        return dfs, data_urls

    return dfs

async def show_user_perf(clearing_house: ClearingHouse):
    frens = get_loaded_auths()

    # st.write('query string:', frens)

    state = await get_state_account(clearing_house.program)
    ch = clearing_house

    every_user_stats = await ch.program.account['UserStats'].all()
    authorities = sorted([str(x.account.authority) for x in every_user_stats])
    authority0, subaccount0, mol0, mol2, _ = st.columns([10, 3, 3, 3, 3])

    if len(frens)==0:
        user_authorities = authority0.selectbox(
            'user authorities', 
            authorities, 
            0
            # on_change=ccc
            # frens
        )
        user_authorities = [user_authorities]
        # user_authorities = ['4FbQvke11D4EdHVsCD3xej2Pncp4LFMTXWJUXv7irxTj']
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
                user_authority_pk = PublicKey(user_authority)
                # print(user_stats)
                user_stat = [x for x in user_stats if str(x.authority) == user_authority][0]
                # chu = ClearingHouseUser(
                #     ch, 
                #     authority=user_authority_pk, 
                #     subaccount_id=0, 
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
        st.dataframe(uak_df.T)
        dates = []
        sub = subaccount0.selectbox('subaccount', [x.split('_')[-1] for x in uak_df.index])
        lastest_date = pd.to_datetime(datetime.datetime.now(), utc=True)
        start_date = mol0.date_input(
            "start date:",
            lastest_date - datetime.timedelta(days=7),
            min_value=datetime.datetime(2022, 11, 4),
            max_value=lastest_date,
        )  # (datetime.datetime.now(tzInfo)))
        end_date = mol2.date_input(
            "end date:",
            lastest_date,
            min_value=datetime.datetime(2022, 11, 4),
            max_value=lastest_date,
        )  # (datetime.datetime.now(tzInfo)))
        dates = pd.date_range(start_date, end_date)
        for user_authority in user_authorities:
            user_authority_pk = PublicKey(user_authority)
            # print(user_stats)
            user_stat = [x for x in user_stats if str(x.authority) == user_authority][0]
            # chu = ClearingHouseUser(
            #     ch, 
            #     authority=user_authority_pk, 
            #     subaccount_id=0, 
            #     use_cache=False
            # )
            for sub_id in range(user_stat.number_of_sub_accounts_created):
                print('sub_id:', sub_id)

                user_account_pk = get_user_account_public_key(
                    clearing_house.program_id,
                    user_authority_pk,
                    sub_id)

                if sub_id==int(sub):
                    # try:
                    micol, srccol = st.columns(2)
                    mi = micol.selectbox('market index:', [0])
                    source = srccol.selectbox('source:', ['trades', 'settledPnl'])

                    if source == 'trades':
                        df, urls = load_user_trades(dates, str(user_account_pk), True)
                        with st.expander('data source(s):'):
                            for x in urls:
                                st.write(x)
                             # st.dataframe(sdf_tot['pnl'].sum())

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
                    
                        # st.dataframe(df[['ts', 'baseAssetAmountFilled', 'taker', 'takerOrderDirection']])
                        df['baseAssetAmountSignedFilled'] = df['baseAssetAmountFilled'] \
                            * df['takerOrderDirection'].apply(lambda x: 2*(x=='long')-1) \
                            * df['taker'].apply(lambda x: 2*(str(x)==str(user_account_pk))-1)
                        # st.dataframe(df)

                        df = df[df.marketIndex==mi]
                        df = df[df.marketType=='perp']
                        user_takers = df[((df.taker.astype(str)==str(user_account_pk)) & ((df.actionExplanation!='liquidation')))]
                        user_makers = df[df.maker.astype(str)==str(user_account_pk)]
                        user_taker_liqs = df[((df.taker.astype(str)==str(user_account_pk)) & ((df.actionExplanation=='liquidation')))]

                        user_takersd = user_takers.groupby('takerOrderId')
                        user_makersd = user_makers.groupby('makerOrderId')

                        user_takersd = pd.concat([user_takersd.agg(
                            {
                            'marketType': 'last',
                            'marketIndex': 'last',
                            'takerOrderDirection': 'last',
                            'actionExplanation': 'last',
                            'takerOrderCumulativeBaseAssetAmountFilled': 'last',
                            'takerOrderCumulativeQuoteAssetAmountFilled': 'last',
                            # 'takerFee': 'sum'
                            }
                        ), 
                        user_takersd.agg(cumulativeTakerFee=('takerFee', 'sum'),
                            first_ts=('ts', 'min'),
                            last_ts=('ts', 'max'),
                            oraclePriceMax=('oraclePrice', 'max'),
                            oraclePriceMin=('oraclePrice', 'min')
                            )
                        ],axis=1)

                        user_takersd['orderFillPrice'] = user_takersd['takerOrderCumulativeQuoteAssetAmountFilled']/user_takersd['takerOrderCumulativeBaseAssetAmountFilled']
                        
                        if len(user_taker_liqs):
                            user_taker_liqsd = user_taker_liqs.groupby('liquidationId')
                            user_taker_liqsd = pd.concat([user_taker_liqsd.agg(
                                {
                                'marketType': 'last',
                                'marketIndex': 'last',
                                'takerOrderDirection': 'last',
                                'actionExplanation': 'last',
                                'takerOrderCumulativeBaseAssetAmountFilled': 'sum',
                                'takerOrderCumulativeQuoteAssetAmountFilled': 'sum',
                                }
                            ), 
                            user_taker_liqsd.agg(
                                cumulativeTakerFee=('takerFee', 'sum'),
                                first_ts=('ts', 'min'),
                                last_ts=('ts', 'max'),
                                oraclePriceMax=('oraclePrice', 'max'),
                                oraclePriceMin=('oraclePrice', 'min')
                                )
                            ],axis=1)
                            user_taker_liqsd['orderFillPrice'] = user_taker_liqsd['takerOrderCumulativeQuoteAssetAmountFilled']/user_taker_liqsd['takerOrderCumulativeBaseAssetAmountFilled']
                            user_taker_liqsd['liquidityStatus'] = 'taker'

                            st.write(user_taker_liqsd)

                        
                        user_makersd = pd.concat([user_makersd.agg(
                            {
                            'marketType': 'last',
                            'marketIndex': 'last',
                            'makerOrderDirection': 'last',
                            'actionExplanation': 'last',
                            'makerOrderCumulativeBaseAssetAmountFilled': 'last',
                            'makerOrderCumulativeQuoteAssetAmountFilled': 'last',
                            #  'makerFee':'sum',
                            }
                        ), 
                        user_makersd.agg(cumulativeMakerFee=('makerFee', 'sum'),
                            first_ts=('ts', 'min'),
                            last_ts=('ts', 'max'),
                            oraclePriceMax=('oraclePrice', 'max'),
                            oraclePriceMin=('oraclePrice', 'min')
                            )
                        ],axis=1)
                        user_makersd['orderFillPrice'] = user_makersd['makerOrderCumulativeQuoteAssetAmountFilled']/user_makersd['makerOrderCumulativeBaseAssetAmountFilled']
                        user_makersd['liquidityStatus'] = 'maker'
                        user_takersd['liquidityStatus'] = 'taker'

                        lls = []
                        if len(user_takers):
                            lls.append(user_takersd)
                        if len(user_taker_liqs):
                            lls.append(user_taker_liqsd)
                            # user_trades_full = pd.concat([user_takersd, user_taker_liqsd, user_makersd])
                        if len(user_makers):
                            lls.append(user_makersd)
                        user_trades_full = pd.concat(lls)
                        user_trades_full = user_trades_full.sort_values(['first_ts'])
                        user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'direction'] = user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'takerOrderDirection']
                        user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'direction'] = user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'makerOrderDirection']
                        user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'size'] = user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'takerOrderCumulativeBaseAssetAmountFilled']
                        user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'size'] = user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'makerOrderCumulativeBaseAssetAmountFilled']
                        user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'notional'] = user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'takerOrderCumulativeQuoteAssetAmountFilled']
                        user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'notional'] = user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'makerOrderCumulativeQuoteAssetAmountFilled']
                        user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'fee'] = user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'cumulativeTakerFee']
                        user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'fee'] = user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'cumulativeMakerFee']

                        st.dataframe(user_trades_full[['direction', 'liquidityStatus', 'size', 'notional', 'orderFillPrice', 'fee', 'actionExplanation']])


                        # st.multiselect('columns:', df.columns, None)
                        ot = df.pivot_table(index='ts', columns=['marketType', 'marketIndex'], values='baseAssetAmountSignedFilled', aggfunc='sum').cumsum().ffill()
                        ot = ot.round(6)
                        ot.columns = [str(x) for x in ot.columns]
                        ot.index = pd.to_datetime((ot.index*1e9).astype(int), utc=True)
                        # ot = ot.resample('1MIN').ffill()
                        st.plotly_chart(ot.plot(title='base asset amount over month (WIP)'))

                        tt = df.groupby(['takerOrderId', 'takerOrderDirection']).count().iloc[:,0].reset_index()
                        tt1 = tt.groupby('takerOrderDirection').count()
                        # st.dataframe(tt1)

                        takerfee_trades = df[df.taker.astype(str)==str(user_account_pk)]['takerFee']
                        # st.plotly_chart(takerfee_trades.plot())

                        trades = df.groupby(['marketType', 'marketIndex']).sum()[['quoteAssetAmountFilled']]
                        trades.columns = ['volume']
                        # st.dataframe(trades)
                        # except:
                        #     st.write('cannot load data')
                    else:
                        sdf = load_user_settlepnl(dates, str(user_account_pk))
                        # st.dataframe(sdf)
                        sdf_tot = sdf.pivot_table(index='ts', columns='marketIndex', 
                                                values=['pnl', 'baseAssetAmount', 'quoteEntryAmount', 'settlePrice'])
                        sdf_tot = sdf_tot.swaplevel(axis=1)
                        
                        sdf_tot = sdf_tot[mi]
                        # st.dataframe(sdf_tot)

                        sdf_tot['costBasisAfter'] = (-(sdf_tot['quoteEntryAmount']) / sdf_tot['baseAssetAmount'] ).fillna(0)
                        sdf_tot['cumulativePnl'] = sdf_tot['pnl'].cumsum()
                        sdf_tot['notional'] = sdf_tot['baseAssetAmount']*sdf_tot['settlePrice']


                        # st.dataframe(sdf_tot)

                        sdf_tot['entryPnl'] = sdf_tot['notional'] + sdf_tot['quoteEntryAmount']
                        # sdf_tot.columns = ['-'.join([str(i) for i in x]) for x in sdf_tot.columns]
                        sdf_tot.index = pd.to_datetime((sdf_tot.index*1e9).astype(int), utc=True)
                        st.plotly_chart(sdf_tot.plot())
                

        st.markdown('user stats')
        st.dataframe(pd.DataFrame([x for x in user_stats]).T)


    # else:
    #     st.text('not found')

    
    