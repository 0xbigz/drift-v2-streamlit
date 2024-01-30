
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
from driftpy.accounts.oracle import *
import datetime

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
import plotly.graph_objs as go
import requests

from datafetch.s3_fetch import load_user_trades, load_user_settlepnl

def load_deposit_history(userkey, source='api'):
    if source == 'api':
        url = 'https://mainnet-beta.api.drift.trade/deposits/userAccounts/?userPublicKeys='+userkey+'&pageIndex=0&pageSize=1000&sinceId=&sinceTs='
        
        st.warning(url)
        res = requests.get(url).json()['data']['records']
        df = pd.concat([pd.DataFrame(x) for x in res])
        for col in ['amount', 'oraclePrice', 'marketDepositBalance', 'marketWithdrawBalance', 'marketCumulativeDepositInterest',
                    'marketCumulativeBorrowInterest', 'totalDepositsAfter', 'totalWithdrawsAfter',]:
            df[col] = df[col].astype(float)

        return df
    
    return None




def make_buy_sell_chart(usr_to_show):
    # Example price series
    usr_to_show.index = pd.to_datetime((usr_to_show.first_ts*1e9).astype(int), utc=True)

    # Example buy and sell signals
    # Create the chart trace for the price series
    traces = []
    for mt in usr_to_show.marketType.unique():
        for mi in usr_to_show.marketIndex.unique():
            uu = usr_to_show[(usr_to_show.marketIndex==mi) & (usr_to_show.marketType==mt)]
            signals = uu.direction.apply(lambda x: 1 if x=='long' else -1)  # 1 = buy, -1 = sell
            prices = uu.orderFillPrice
            price_trace = go.Scatter(x=prices.index, y=prices, mode='lines', name='Price-'+str(mt)+'-'+str(mi))

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
                name='Action-'+str(mt)+'-'+str(mi)
            )
            traces.append(price_trace)
            traces.append(signal_trace)


    # Create the chart layout
    layout = go.Layout(
        title='Price Chart (with Buy and Sell Action)',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Price'),
    )

    # Create the chart figure
    fig = go.Figure(data=traces, layout=layout)
    return fig


# @st.cache_data
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

async def show_user_perf(clearing_house: DriftClient):
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
        user_authorities = [user_authorities]
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

        # url = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/'
        # url += 'user/%s/trades/%s/%s'

        userAccountKeys = []
        user_stat = None
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
            user_authority_pk = Pubkey.from_string(user_authority)
            user_stat = [x for x in user_stats if str(x.authority) == user_authority][0]

            for sub_id in range(user_stat.number_of_sub_accounts_created):
                # print('sub_id:', sub_id)

                user_account_pk = get_user_account_public_key(
                    clearing_house.program_id,
                    user_authority_pk,
                    sub_id)

                if sub_id==int(sub):
                    chu = DriftUser(
                        ch, 
                        user_public_key=user_account_pk, 
                        # use_cache=True
                        account_subscription=AccountSubscriptionConfig("cached"))
                    
                    await chu.drift_client.account_subscriber.update_cache()
                    await chu.account_subscriber.update_cache()

                    
                    user_acct = chu.get_user_account()
                    nom = bytes(user_acct.name).decode('utf-8')

                    st.write('"'+nom+'"')
                    tabs1 = st.tabs(['trades', 'deposits', 'all time stats', 'snapshot'])


                    atnd_token_current_value = 0

                    with tabs1[3]:
                        from datafetch.snapshot_fetch import load_user_snapshot
                        dd = st.columns(3)
                        commit_hash = dd[0].text_input(label='commit', value='main')


                        # df, ff = load_user_snapshot(str(user_account_pk), commit_hash)
                        # dd[-1].write(ff)
                        # st.dataframe(df)

                    with tabs1[1]:
                        df = load_deposit_history(str(user_account_pk))
                        df['amount2'] = df['amount'] * df['direction'].apply(lambda x: -1 if x!='deposit' else 1)
                        df2 = df.pivot_table(index='ts', columns='marketIndex', values='amount2')

                        for col in df2.columns:
                            if col in [1,2]: #msol/sol
                                df2[col]/=1e9
                            else:
                                df2[col]/=1e6

                        df2.index = pd.to_datetime((df2.index.astype(int)*1e9).astype(int),  utc=True)
                        net_dep_amounts = df2.fillna(0).sum()
                        ccs = st.columns(len(net_dep_amounts))
                        for i,c in enumerate(ccs):  
                            # atnd_token_ts_value += net_dep_amounts.iloc[i] * 
                            mi = net_dep_amounts.index[i]
                            spot_market = chu.get_spot_market_account(mi)
                            current_price = (chu.get_oracle_data_for_spot_market(mi)).price / PRICE_PRECISION
                            atnd_token_current_value += net_dep_amounts.iloc[i] * current_price
                            c.metric('net deposit tokens (market_index='+str(mi)+')', net_dep_amounts.iloc[i])



                        st.plotly_chart(df2.fillna(0).cumsum().plot())
                        # st.write(df.sum())
                        # st.plotly_chart(df.plot())

                    with tabs1[2]:
                        atnd = (user_acct.total_deposits - user_acct.total_withdraws)/1e6
                        atpp = user_acct.settled_perp_pnl/1e6
                        atfp = user_acct.cumulative_perp_funding/1e6
                        atsf = user_acct.cumulative_spot_fees/1e6

                        atsp =  atpp + atsf
                        s1,s2,s3,s4 = st.columns(4)
                        tc =(chu.get_total_collateral())/1e6
                        s1.metric(
                            'Account Value:', 
                        f'${tc:,.3f}', 
                        f'{tc - atnd_token_current_value:,.3f} all time pnl')
                        
                        s1.metric('Net Deposits ($ @ ts)', 
                                  f'${atnd:,.3f}', 
                                  f'{user_acct.total_deposits/1e6:,.3f} lifetime deposits')

                        s1.metric('Net Deposits ($ now)', 
                                  f'${atnd_token_current_value:,.3f}',
                                  )
                        gains_since_inception = {}
                        for idx in range(len(user_acct.spot_positions)):
                            spot_pos = user_acct.spot_positions[idx]
                            if spot_pos.cumulative_deposits != 0:
                                    spot_market_act = await get_spot_market_account(ch.program, spot_pos.market_index)
                                    scale = 10**spot_market_act.decimals
                                    spot_market_name = bytes(spot_market_act.name).decode('utf-8')
                                    if str(spot_pos.balance_type) == 'SpotBalanceType.Deposit()':
                                        tokens = spot_pos.scaled_balance * spot_market_act.cumulative_deposit_interest/1e10/(1e9)
                                    else:
                                        tokens = -spot_pos.scaled_balance * spot_market_act.cumulative_borrow_interest/1e10/(1e9)
                                    

                                    gains_since_inception[spot_pos.market_index] = tokens - (spot_pos.cumulative_deposits/scale)
                                    s3.metric(spot_market_name+' deposits:', 
                                              f'{tokens:,.3f}',
                                              f'{(gains_since_inception[spot_pos.market_index]):,.3f} from inception',
                                    )

                        s1,s2,s3,s4 = st.columns(4)
                        s1.metric('all time pnl + fees:', f'${atsp:,.3f}', f'{atpp:,.3f} perp pnl')
                        # s2.metric('all time perp pnl:', atpp)
                        s3.metric('all time funding pnl:', atfp)
                        s4.metric('all time spot fees:', -atsf)

                        # if abs(atsp - (atpp)) > 2e-6:
                        #     st.write('excess settled pnl?:', atsp - (atpp))

                        excess_usdc_gains_since_inception = gains_since_inception.get(0, 0) - atsp

                        st.write('excess usdc gains:', excess_usdc_gains_since_inception)

                    with tabs1[0]:
                        mtcol, micol, srccol = st.columns(3)
                        mt = mtcol.selectbox('market type:', ['perp', 'spot', 'all'])
                        mi = None
                        sources = ['trades', 'settledPnl', 'all']

                        if mt == 'perp':
                            mi = micol.selectbox('market index:', range(0, state.number_of_markets))
                        elif mt == 'spot':
                            mi = micol.selectbox('market index:', range(1, state.number_of_spot_markets))
                            sources = ['trades']

                        source = srccol.selectbox('source:', sources)

                        if source == 'trades' or source == 'all':
                            df, urls = load_user_trades(dates, str(user_account_pk), True)
                            if len(df) == 0:
                                continue

                            if mi is not None:
                                df = df[df.marketIndex==mi]
                            if mt in ['perp', 'spot']:
                                df = df[df.marketType==mt]

                            df = filter_dups(df)
                            df['baseAssetAmountSignedFilled'] = df['baseAssetAmountFilled'] \
                                * df['takerOrderDirection'].apply(lambda x: 2*(x=='long')-1) \
                                * df['taker'].apply(lambda x: 2*(str(x)==str(user_account_pk))-1)
                            df['quoteAssetAmountSignedFilled'] = df['quoteAssetAmountFilled'] \
                                * df['takerOrderDirection'].apply(lambda x: 2*(x=='long')-1) \
                                * df['taker'].apply(lambda x: 2*(str(x)==str(user_account_pk))-1)
                            
                            with st.expander('data source(s):'):
                                for x in urls:
                                    st.write(x)

                            user_trades_full = build_liquidity_status_df(df, user_account_pk)
                            if len(user_trades_full):
                                usr_to_show = user_trades_full[['marketType', 'marketIndex', 
                                                                'direction', 'liquidityStatus', 'size', 'notional', 
                                                                'orderFillPrice', 'fee', 'actionExplanation',
                                                                'first_ts', 'last_ts']]
                                
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
                                # st.write(df.columns)
                                cost_ser = df.pivot_table(index='ts', columns=['marketType', 'marketIndex'], values='quoteAssetAmountSignedFilled',aggfunc='sum').cumsum().ffill()
                                cost_ser.columns = [str(x) for x in cost_ser.columns]
                                cost_ser.index = pd.to_datetime((cost_ser.index*1e9).astype(int), utc=True)


                                px_ser = df.pivot_table(index='ts', columns=['marketType', 'marketIndex'], values='oraclePrice', aggfunc='last').ffill()
                                px_ser.columns = [str(x) for x in px_ser.columns]
                                px_ser.index = pd.to_datetime((px_ser.index*1e9).astype(int), utc=True)
                                pnl_ser = px_ser * ot - cost_ser
                                pnl_ser['total'] = pnl_ser.sum(axis=1)
                                st.plotly_chart(pnl_ser.plot(title='pnl over time'))

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
                        elif source == 'settledPnl' or source == 'all':
                            sdf = load_user_settlepnl(dates, str(user_account_pk))
                            # st.dataframe(sdf)
                            sdf_tot = sdf.pivot_table(index='ts', columns='marketIndex', 
                                                    values=['pnl', 'baseAssetAmount', 'quoteEntryAmount', 'settlePrice'])
                            
                            # if isinstance(mi, int):
                            sdf_tot0 = sdf_tot.swaplevel(axis=1) # put market index on top
                            mis = None
                            if mi is not None:
                                # sdf_tot = sdf_tot[mi]
                                # sdf_tot = sdf_tot.swaplevel(axis=1)
                                mis = [mi]
                            else:
                                mis = list(sdf_tot0.columns.levels[0].unique())
                            for mi in mis:
                                sdf_tot = sdf_tot0[mi]
                                sdf_tot.index = pd.to_datetime((sdf_tot.index*1e9).astype(int), utc=True)
                                sdf_tot = sdf_tot.loc[dates[0]:dates[1]]

                                # st.dataframe(sdf_tot)
                                sdf_tot['costBasisAfter'] = (-(sdf_tot['quoteEntryAmount']) / sdf_tot['baseAssetAmount'] ).fillna(0)
                                sdf_tot['cumulativePnl'] = sdf_tot['pnl'].cumsum()
                                sdf_tot['notional'] = (sdf_tot['baseAssetAmount']*sdf_tot['settlePrice'])


                                sdf_tot['entryPnl'] = sdf_tot['notional'] + sdf_tot['quoteEntryAmount']
                                # sdf_tot.columns = ['-'.join([str(i) for i in x]) for x in sdf_tot.columns]
                                m1, m2 = st.columns(2)

                                m1.metric('pnl:', sdf_tot['pnl'].sum())
                                st.dataframe(sdf_tot)

                                st.plotly_chart(sdf_tot.plot())
                        

            # st.markdown('user stats')
            # st.dataframe(pd.DataFrame([x for x in user_stats]).T)


        # else:
        #     st.text('not found')

        
        