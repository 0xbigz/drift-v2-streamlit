
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
import plotly.graph_objs as go


def make_buy_sell_chart(usr_to_show):
    # Example price series
    usr_to_show.index = pd.to_datetime((usr_to_show.first_ts*1e9).astype(int), utc=True)
    prices = usr_to_show.orderFillPrice

    # Example buy and sell signals
    signals = usr_to_show.direction.apply(lambda x: 1 if x=='long' else -1)  # 1 = buy, -1 = sell
    # Create the chart trace for the price series
    price_trace = go.Scatter(x=prices.index, y=prices, mode='lines', name='Price')

    # Create the chart trace for the buy and sell signals
    signal_trace = go.Scatter(
        x=prices.index[signals != 0],
        y=prices[signals != 0],
        mode='markers',
        marker=dict(
            symbol=['triangle-up'if s == 1 else 'triangle-down' for s in signals],
            color=['green' if s == 1 else 'red' for s in signals],
            size=10,
        ),
        name='Action',
    )

    # Create the chart layout
    layout = go.Layout(
        title='Price Chart (with Buy and Sell Action)',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Price'),
    )

    # Create the chart figure
    fig = go.Figure(data=[price_trace, signal_trace], layout=layout)
    return fig


# @st.experimental_memo
def get_loaded_auths():

    query_p = st.experimental_get_query_params()
    frens = query_p.get('authority', [])
    if frens != []:
        frens = frens[0].split(',')
    return frens

def build_liquidity_status_df(df, user_account_pk):
    def stitch_liq_status():
        user_takersd = make_liq_status_df('taker', False)
        user_taker_liqsd = make_liq_status_df('taker', True)
        user_makersd = make_liq_status_df('maker', None)

        lls = []
        if len(user_takersd):
            lls.append(user_takersd)
        if len(user_taker_liqsd):
            lls.append(user_taker_liqsd)
            # user_trades_full = pd.concat([user_takersd, user_taker_liqsd, user_makersd])
        if len(user_makersd):
            lls.append(user_makersd)
        if len(lls) == 0:
            return pd.DataFrame()

        user_trades_full = pd.concat(lls)
        user_trades_full = user_trades_full.sort_values(['first_ts'])
        if 'takerOrderDirection' in user_trades_full.columns:
            user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'direction'] = user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'takerOrderDirection']
            user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'size'] = user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'takerOrderCumulativeBaseAssetAmountFilled']
            user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'notional'] = user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'takerOrderCumulativeQuoteAssetAmountFilled']
            user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'fee'] = user_trades_full.loc[user_trades_full.liquidityStatus=='taker', 'cumulativeTakerFee']
        if 'makerOrderDirection' in user_trades_full.columns:
            user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'direction'] = user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'makerOrderDirection']
            user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'size'] = user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'makerOrderCumulativeBaseAssetAmountFilled']
            user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'notional'] = user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'makerOrderCumulativeQuoteAssetAmountFilled']
            user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'fee'] = user_trades_full.loc[user_trades_full.liquidityStatus=='maker', 'cumulativeMakerFee']

        return user_trades_full

    def make_liq_status_df(status, is_liq=None):
        assert(status in ['taker', 'maker'])
        user_takers = df[(df[status].astype(str)==str(user_account_pk))]
        if is_liq is not None:
            if is_liq:
                liq_filt = user_takers.actionExplanation=='liquidation'
            else:
                liq_filt = user_takers.actionExplanation!='liquidation'
            user_takers = user_takers[(((liq_filt)))]

        if is_liq == True:
            if 'liquidationId' in user_takers:
                user_takersd = user_takers.groupby('liquidationId')
            else:
                user_takersd = pd.DataFrame()
        else:
            user_takersd = user_takers.groupby(status+'OrderId')

        if len(user_takersd) == 0:
            return user_takersd

        user_takersd = pd.concat([user_takersd.agg(
        {
        'marketType': 'last',
        'marketIndex': 'last',
        status+'OrderDirection': 'last',
        'actionExplanation': 'last',
        status+'OrderCumulativeBaseAssetAmountFilled': 'last',
        status+'OrderCumulativeQuoteAssetAmountFilled': 'last',
        }
        ), 
        user_takersd.agg(
            cumulativeTakerFee=('takerFee', 'sum'),
            cumulativeMakerFee=('makerFee', 'sum'),
            first_ts=('ts', 'min'),
            last_ts=('ts', 'max'),
            oraclePriceMax=('oraclePrice', 'max'),
            oraclePriceMin=('oraclePrice', 'min')
            )
        ],axis=1)

        user_takersd['orderFillPrice'] = user_takersd[status+'OrderCumulativeQuoteAssetAmountFilled']/user_takersd[status+'OrderCumulativeBaseAssetAmountFilled']
        user_takersd['liquidityStatus'] = status

        # if status == 'maker':
        #     st.dataframe(user_takersd)

        return user_takersd

    return stitch_liq_status()

def filter_dups(df):
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
    return df


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
            try:
                dd = pd.read_csv(data_url)
                dfs.append(dd)
            except:
                pass
    if len(dfs):
        dfs = pd.concat(dfs)
    else:
        dfs = pd.DataFrame()

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
            data_urls.append(data_url)
            try:
                dd = pd.read_csv(data_url, nrows=50000)
                if 'liquidation' in dd['actionExplanation'].unique():
                    data_liq_url = liq_url % (user_key, year, month)
                    data_urls.append(data_liq_url)
                    dd1 = pd.read_csv(data_liq_url)
                    dd = dd.merge(dd1, suffixes=('', '_l'), how='outer', on='txSig')
                dfs.append(dd)
            except Exception as e:
                st.warning(data_url+ ' failed to load ('+str(e)+')')
    if len(dfs):
        dfs = pd.concat(dfs)
    else:
        dfs = pd.DataFrame()
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
        # user_authorities = ['4FbQvke11D4EdHVsCD3xej2Pncp4LFMTXWJUXv7irxTj']
    else:
        user_authorities = authority0.selectbox(
            'user authorities', 
            frens, 
            0
            # on_change=ccc
            # frens
        )
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

                    tabs = st.tabs(['per market', 'all time stats'])

                    with tabs[1]:
                        user_acct = await get_user_account(clearing_house.program, user_authority_pk, sub_id)
                        atnd = (user_acct.total_deposits - user_acct.total_withdraws)/1e6
                        atpp = user_acct.settled_perp_pnl/1e6
                        atfp = user_acct.cumulative_perp_funding/1e6
                        atsf = user_acct.cumulative_spot_fees/1e6


                        atsp =  atpp + atfp + atsf
                        s1,s2,s3,s4 = st.columns(4)
                        s1.metric('all time settled pnl:', atsp)

                        if user_acct.spot_positions[0].market_index == 0:
                            s2.metric('all time usdc deposits:', user_acct.spot_positions[0].cumulative_deposits/1e6)

                        if user_acct.spot_positions[1].market_index == 1:
                            s3.metric('all time sol deposits:', user_acct.spot_positions[1].cumulative_deposits/1e6)


                        s1,s2,s3,s4 = st.columns(4)
                        s1.metric('all time net deposits:', atnd)
                        s2.metric('all time perp pnl:', atpp)
                        s3.metric('all time funding pnl:', atfp)
                        s4.metric('all time spot fees:', -atsf)

                    with tabs[0]:
                        mtcol, micol, srccol = st.columns(3)
                        mt = mtcol.selectbox('market type:', ['perp', 'spot'])
                        if mt == 'perp':
                            mi = micol.selectbox('market index:', range(0, state.number_of_markets))
                        else:
                            mi = micol.selectbox('market index:', range(1, state.number_of_spot_markets))
                        source = srccol.selectbox('source:', ['trades', 'settledPnl'])

                        if source == 'trades':
                            df, urls = load_user_trades(dates, str(user_account_pk), True)
                            if len(df) == 0:
                                continue
                            df = df[df.marketIndex==mi]
                            df = df[df.marketType==mt]
                            df = filter_dups(df)
                            df['baseAssetAmountSignedFilled'] = df['baseAssetAmountFilled'] \
                                * df['takerOrderDirection'].apply(lambda x: 2*(x=='long')-1) \
                                * df['taker'].apply(lambda x: 2*(str(x)==str(user_account_pk))-1)

                            with st.expander('data source(s):'):
                                for x in urls:
                                    st.write(x)

                            user_trades_full = build_liquidity_status_df(df, user_account_pk)
                            if len(user_trades_full):
                                usr_to_show = user_trades_full[['direction', 'liquidityStatus', 'size', 'notional', 'orderFillPrice', 'fee', 'actionExplanation', 'first_ts', 'last_ts']]
                                
                                def highlight_survived(s):
                                    return ['background-color: #90EE90']*len(s) if s.direction=='long' else ['background-color: pink']*len(s)

                                m1, m2, m3 = st.columns(3)
                                m3.metric('total fees paid:', f'${usr_to_show["fee"].sum():,.2f}',
                                f'${usr_to_show[usr_to_show.actionExplanation == "liquidation"]["fee"].sum():,.2f} in liquidation fees')

                                m2.metric('total volume:', 
                                f'${usr_to_show["notional"].sum():,.2f}',
                                f'${usr_to_show[usr_to_show.liquidityStatus == "maker"]["notional"].sum():,.2f} was maker volume')
                                

                                st.dataframe(usr_to_show.reset_index(drop=True).style.apply(highlight_survived, axis=1))

                                # # st.multiselect('columns:', df.columns, None)
                                ot = df.pivot_table(index='ts', columns=['marketType', 'marketIndex'], values='baseAssetAmountSignedFilled', aggfunc='sum').cumsum().ffill()
                                ot = ot.round(6)
                                ot.columns = [str(x) for x in ot.columns]
                                ot.index = pd.to_datetime((ot.index*1e9).astype(int), utc=True)
                                # ot = ot.resample('1MIN').ffill()
                                st.plotly_chart(ot.plot(title='position over time'))

                                st.plotly_chart(make_buy_sell_chart(usr_to_show))
                            else:
                                st.write('no history for this market')



                            # tt = df.groupby(['takerOrderId', 'takerOrderDirection']).count().iloc[:,0].reset_index()
                            # tt1 = tt.groupby('takerOrderDirection').count()
                            # # st.dataframe(tt1)

                            # takerfee_trades = df[df.taker.astype(str)==str(user_account_pk)]['takerFee']
                            # # st.plotly_chart(takerfee_trades.plot())

                            # trades = df.groupby(['marketType', 'marketIndex']).sum()[['quoteAssetAmountFilled']]
                            # trades.columns = ['volume']
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
                    

            # st.markdown('user stats')
            # st.dataframe(pd.DataFrame([x for x in user_stats]).T)


        # else:
        #     st.text('not found')

        
        