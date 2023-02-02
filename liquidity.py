
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
from glob import glob


def get_mm_score_for_snap_slot(df):
    d = df[(df.orderType=='limit') 
    # & (df.postOnly)
    ]

    #todo: too slow
    assert(len(d.snap_slot.unique())==1)
    top6bids = d[d.direction=='long'].groupby('price').sum().sort_values('price', ascending=False)[['baseAssetAmount']]
    top6asks = d[d.direction=='short'].groupby('price').sum()[['baseAssetAmount']]

    tts = pd.concat([top6bids['baseAssetAmount'].reset_index(drop=True), top6asks['baseAssetAmount'].reset_index(drop=True)],axis=1)
    tts.columns = ['bs','as']
    min_q = (2000/float(tts.mean().mean()))
    q = ((tts['bs']+tts['as'])/2).apply(lambda x: max(x, min_q))
    score_scale = tts.min(axis=1)/q * 100
    score_scale = score_scale * pd.Series([2, .75, .5, .4, .3, .2])

    for i,x in enumerate(top6bids.index[:6]):
        d.loc[(d.price==x)  & (d.direction=='long'), 'score'] = score_scale.values[i]
    for i,x in enumerate(top6asks.index[:6]):
        d.loc[(d.price==x) & (d.direction=='short'), 'score'] = score_scale.values[i]
    
    return d


def get_mm_stats(df, user, oracle, bbo2):
    df = df[df.user == user]
    # print(df.columns)

    bbo = df.groupby(['direction', 'snap_slot']).agg({'price':['min', 'max']}).unstack(0)['price'].swaplevel(axis=1)
    ll = {}
    if 'long' in bbo.columns:
        lmax = bbo['long']['max']
        ll['best dlob bid'] = (lmax)
    ll['oracle'] = (oracle)
    if 'short' in bbo.columns:
        smin = bbo['short']['min']
        ll['best dlob offer'] = (smin)

    bbo_user = pd.concat(ll,axis=1).reindex(ll['oracle'].index)
    bbo_user = bbo_user.loc[bbo2.index[0]:bbo2.index[-1]]
    # bbo_user['score'] = df.groupby(['direction', 'snap_slot'])['score'].sum()
    bbo_user_score = df.groupby('snap_slot')['score'].sum().loc[bbo2.index[0]:bbo2.index[-1]]
    bbo_user = pd.concat([bbo_user, bbo_user_score],axis=1)
    bbo_user_avg_score = bbo_user['score'].fillna(0).mean()
    # if(float(bbo_user_avg_score) > 90):
    #     print(user)
    #     print(bbo_user['score'].describe())
    #     print(bbo_user['score'].fillna(0))

    try:
        uptime_pct = len(bbo_user.dropna())/len(bbo_user)
        bid_best_pct = (((bbo_user['best dlob bid']-bbo2['best dlob bid'])/bbo2['best dlob bid']) == 0).mean()
        bid_within_best_pct = (((bbo_user['best dlob bid']-bbo2['best dlob bid'])/bbo2['best dlob bid']) > -.0003).mean()
        offer_best_pct = (((bbo_user['best dlob offer']-bbo2['best dlob offer'])/bbo2['best dlob offer']) == 0).mean()
        offer_within_best_pct = (((bbo_user['best dlob offer']-bbo2['best dlob offer'])/bbo2['best dlob offer']) < .0003).mean()
    except:
        bid_best_pct = 0
        bid_within_best_pct = 0
        offer_best_pct = 0
        offer_within_best_pct = 0
    bbo_stats = pd.DataFrame(
        [[uptime_pct, bid_best_pct, bid_within_best_pct, offer_best_pct, offer_within_best_pct, bbo_user_avg_score/100]],
        index=[user],
        columns=['uptime%', 'best_bid%', 'near_best_bid%', 'best_offer%', 'near_best_offer%', 'avg score']
        ).T * 100

    return bbo_user, bbo_stats


def mm_page(clearing_house: ClearingHouse):    
    st.title('best bid/offer')
    market_index = st.selectbox('market index', [0, 1, 2])
    dfs = []

    tt = 'perp'+str(market_index)
    ggs = glob('../drift-v2-orderbook-snap/'+tt+'/*.csv')

    df = None
    if len(ggs):
        print('building new data file with', len(ggs), 'records!')
        for x in sorted(ggs)[-3500:]:
            df = pd.read_csv(x) 
            df['snap_slot'] = int(x.split('_')[-1].split('.')[0])
            df = get_mm_score_for_snap_slot(df)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        df.to_csv('data/'+tt+'.csv', index=False)
    else:
        df = pd.read_csv('data/'+tt+'.csv')

    oracle = df.groupby('snap_slot')['oraclePrice'].max()
    bbo = df.groupby(['direction', 'snap_slot']).agg({'price':['min', 'max']}).unstack(0)['price'].swaplevel(axis=1)
    lmax = bbo['long']['max']
    smin = bbo['short']['min']

    bbo2 = pd.concat([lmax, oracle, smin],axis=1)
    bbo2.columns = ['best dlob bid', 'oracle', 'best dlob offer']
    st.text('last updated at slot: ' + str(bbo2.index[-1]))
    values = st.slider(
     'Select a range of slot values',
     int(bbo2.index[0]), int(bbo2.index[-1]), (int(bbo2.index[0]), int(bbo2.index[-1])))

    bbo2snippet = bbo2.loc[values[0]:values[1]]
    st.markdown('[data source](https://github.com/0xbigz/drift-v2-orderbook-snap)'+\
        ' ([slot='+str(bbo2snippet.index[0])+'](https://github.com/0xbigz/drift-v2-orderbook-snap/blob/main/'+tt+'/orderbook_slot_'+str(bbo2snippet.index[0])+'.csv))')

    # st.write('slot range:', values)
    st.plotly_chart(bbo2snippet.plot(title='perp market index='+str(market_index)))

    all_stats = []
    for user in df.user.unique():
        bbo_user, bbo_user_stats = get_mm_stats(df, user, oracle, bbo2snippet)
        all_stats.append(bbo_user_stats)

    st.title('mm leaderboard')
    st.dataframe(pd.concat(all_stats, axis=1).T.sort_values('best_bid%', ascending=False))

    st.title('individual mm lookup')

    user = st.selectbox('individual maker', df.user.unique())
        
    bbo_user, bbo_user_stats = get_mm_stats(df, user, oracle, bbo2)

    # st.text('user bestbid time='+str(np.round(offer_best_pct*100, 2))+'%')
    # st.text('user bid within 3bps of best time='+str(np.round(offer_within_best_pct*100, 2))+'%')
    # st.text('user bestoffer time='+str(np.round(offer_best_pct*100, 2))+'%')
    # st.text('user offer within 3bps of best time='+str(np.round(offer_within_best_pct*100, 2))+'%')
    # st.text('user uptime='+str(np.round(uptime_pct*100, 2))+'%')
    st.text('stats last updated at slot: ' + str(bbo_user.index[-1]))
    st.text('current slot:')
    st.plotly_chart(bbo_user.loc[values[0]:values[1]].plot(title='perp market index='+str(market_index)))

    st.title('individual snapshot lookup')
    slot = st.select_slider('individual snapshot', df.snap_slot.unique().tolist())
    st.dataframe(df[df.snap_slot.astype(int)==int(slot)])


        
    