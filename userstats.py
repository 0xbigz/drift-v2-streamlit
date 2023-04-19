
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
import plotly.express as px

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
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
import time
async def show_user_stats(clearing_house: ClearingHouse):
    ch = clearing_house

    all_user_stats = await ch.program.account['UserStats'].all()
    kp = Keypair()
    ch = ClearingHouse(ch.program, kp)

    df = pd.DataFrame([x.account.__dict__ for x in all_user_stats])
    fees_df = pd.DataFrame([x.account.fees.__dict__ for x in all_user_stats])
    print(df.columns)

    # print(df.columns)
    df = pd.concat([df, fees_df], axis=1)
    # print(df.columns)
    for x in df.columns:
        if x in ['taker_volume30d', 'maker_volume30d', 'filler_volume30d', 'total_fee_paid', 'total_fee_rebate', 'total_referrer_reward', 'if_staked_quote_asset_amount']:
            df[x] /= 1e6
    
    current_ts = time.time()
    # print(current_ts)
    df['last_trade_seconds_ago'] = int(current_ts) - df[['last_taker_volume30d_ts', 'last_maker_volume30d_ts']].max(axis=1).astype(int)
    df['last_fill_seconds_ago'] = int(current_ts) - df['last_filler_volume30d_ts'].astype(int)

    volume_scale = (1 - df['last_trade_seconds_ago']/(60*60*24*30)).apply(lambda x: max(0, x))
    fill_scale =  (1 - df['last_fill_seconds_ago']/(60*60*24*30)).apply(lambda x: max(0, x))
    # print(volume_scale)

    
    df['filler_volume30d_calc'] = df[['filler_volume30d']].sum(axis=1)\
        .mul(fill_scale, axis=0)
    df['taker_volume30d_calc'] = df[['taker_volume30d']].sum(axis=1)\
        .mul(volume_scale, axis=0)
    df['maker_volume30d_calc'] = df[['maker_volume30d']].sum(axis=1)\
        .mul(volume_scale, axis=0)
    df['total_30d_volume_calc'] = df[['taker_volume30d', 'maker_volume30d']].sum(axis=1)\
        .mul(volume_scale, axis=0)
    df['authority'] = df['authority'].astype(str)
    df['referrer'] = df['referrer'].astype(str)
    df = df[['authority', 'total_30d_volume_calc', 'taker_volume30d_calc', 'maker_volume30d_calc', 'last_trade_seconds_ago', 'taker_volume30d', 'maker_volume30d', 
    'filler_volume30d_calc', 'filler_volume30d', 'total_fee_paid', 'total_fee_rebate', 
    'number_of_sub_accounts', 'is_referrer', 'if_staked_quote_asset_amount', 'referrer', 'total_referrer_reward',
    'last_fill_seconds_ago',
    ]].sort_values('last_trade_seconds_ago').reset_index(drop=True)



    tabs = st.tabs(['Volume', 'Refferals', 'Fillers'])
    with tabs[0]:
        pie1, pie2 = st.columns(2)

        net_vamm_maker_volume = df['taker_volume30d_calc'].sum() - df['maker_volume30d_calc'].sum()

        other = pd.DataFrame(df.sort_values('taker_volume30d_calc', ascending=False).iloc[10:].sum(axis=0)).T
        other['authority'] = 'Other'
        dfmin = pd.concat([df.sort_values('taker_volume30d_calc', ascending=False).head(10), other],axis=0)
        dfmin['authority'] = dfmin['authority'].apply(lambda x: str(x)[:4]+'...'+str(x)[-4:] if x !="Other" else x)

        fig = px.pie(dfmin, values='taker_volume30d_calc', names='authority',
                    title='30D Taker Volume Breakdown ('+  str(int(df['taker_volume30d_calc'].pipe(np.sign).sum())) +' unique)',
                    hover_data=['taker_volume30d_calc'], 
                    # labels={'$ balance':'balance'}
                    )
        pie1.plotly_chart(fig)

        other = pd.DataFrame(df.sort_values('maker_volume30d_calc', ascending=False).iloc[10:].sum(axis=0)).T
        if net_vamm_maker_volume > 0:
            vamm = other.copy()
            vamm.maker_volume30d_calc = net_vamm_maker_volume
            other = pd.concat([other, vamm])
        other['authority'] = ['Other', 'vAMM']
        dfmin = pd.concat([df.sort_values('maker_volume30d_calc', ascending=False).head(10), other],axis=0)
        dfmin['authority'] = dfmin['authority'].apply(lambda x: str(x)[:4]+'...'+str(x)[-4:] if x not in ["Other", 'vAMM'] else x)

        fig = px.pie(dfmin, values='maker_volume30d_calc', names='authority',
                    title='30D Maker Volume Breakdown ('+  str(int(df['maker_volume30d_calc'].pipe(np.sign).sum())) +' unique)',
                    hover_data=['maker_volume30d_calc'], 
                    # labels={'$ balance':'balance'}
                    )
        pie2.plotly_chart(fig)


        col1, col2, col3 = st.columns(3)
        col1.metric('30D User Taker Volume', str(np.round(df['taker_volume30d_calc'].sum()/1e6, 2))+'M',
            str(int(df['taker_volume30d_calc'].pipe(np.sign).sum())) + ' unique'
        )
        col2.metric('30D User Maker Volume', str(np.round(df['maker_volume30d_calc'].sum()/1e6, 2))+'M',
        
        
                str(int(df['maker_volume30d_calc'].pipe(np.sign).sum())) + ' unique'

        )
        col3.metric('30D vAMM Volume (Net Maker)', 
        str(np.round(net_vamm_maker_volume/1e6, 2))+'M')

        st.dataframe(df)
        

        dd = df.set_index('last_trade_seconds_ago')[[
            'taker_volume30d',
            # 'filler_volume30d',
        'maker_volume30d']].pipe(np.sign).cumsum().loc[:60*60*24]
        dd.index /= 3600
        active_users_past_24hrs = int(dd.values[-1].max())
        st.plotly_chart(dd.plot(title='# user active over past 24hr = '+str(active_users_past_24hrs)))
    
    with tabs[1]:
        st.write('ref leaderboard')
        oo = df.groupby('referrer')
        tt = oo.count().iloc[:,0:1]
        tt2 = oo.authority.agg(list)
        tt2.columns = ['referees']
        tt.columns = ['number reffered']
        val = df.loc[df.authority.isin(tt.index), ['authority', 'total_referrer_reward']]
        val = val.set_index('authority')
        tt = pd.concat([tt, val, tt2], axis=1)
        tt = tt.loc[[x for x in tt.index if x != '11111111111111111111111111111111']]
        tt = tt.sort_values('number reffered', ascending=False)
        st.dataframe(tt)
        
    with tabs[2]:
        st.write('filler leaderboard')
        df2 = df[df.filler_volume30d>0]
        df2 = df2.set_index('authority')[['filler_volume30d_calc', 'last_fill_seconds_ago']].sort_values(by='filler_volume30d_calc', ascending=False)
        st.dataframe(df2)