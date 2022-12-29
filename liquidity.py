
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


def get_mm_stats(df, user, oracle, bbo2):
    df = df[df.user == user]

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
        [[uptime_pct, bid_best_pct, bid_within_best_pct, offer_best_pct, offer_within_best_pct]],
        index=[user],
        columns=['uptime%', 'best_bid%', 'near_best_bid%', 'best_offer%', 'near_best_offer%']
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
        for x in sorted(ggs):
            df = pd.read_csv(x) 
            df['snap_slot'] = int(x.split('_')[-1].split('.')[0])
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
        bbo_user, bbo_user_stats = get_mm_stats(df, user, oracle, bbo2)
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
    st.plotly_chart(bbo_user.plot(title='perp market index='+str(market_index)))


        
    