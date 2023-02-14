
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"
import datetime
import pytz

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


def slot_to_timestamp_est(slot):
    # ts = X - (Y - slot)/2
    # X - ts = (Y - slot)/2
    # slot = Y - 2 * (X - ts)
    return 1676400329 - (177774625-slot)*.5

def get_slots_for_date(date: datetime.date):
    utimestamp = (date - datetime.date(1970,1,1)).total_seconds()
    start = 177774625-2*(1676400329-utimestamp)
    end = start + 60*60*24*2
    return [start, end]

def get_mm_score_for_snap_slot(df):
    d = df[(df.orderType=='limit') 
    # & (df.postOnly)
    ]
    d['baseAssetAmountLeft'] = d['baseAssetAmount'] - d['baseAssetAmountFilled']
    assert(len(d.snap_slot.unique())==1)

    market_index = d.marketIndex.max()
    oracle = d.oraclePrice.max()
    best_bid = d[d.direction=='long']['price'].max()
    best_ask = d[d.direction=='short']['price'].min()
    if(best_bid > best_ask):
        if best_bid > oracle:
            best_bid = best_ask
        else:
            best_ask = best_bid

    mark_price = (best_bid+best_ask)/2
    within_bps_of_price = (mark_price * .0005)

    if market_index == 1 or market_index == 2:
        within_bps_of_price = (mark_price * .00025)

    def rounded_threshold(x):
        return np.round(float(x)/within_bps_of_price) * within_bps_of_price

    d['priceRounded'] = d['price'].apply(rounded_threshold)
    # print(d)

    top6bids = d[d.direction=='long'].groupby('priceRounded').sum().sort_values('priceRounded', ascending=False)[['baseAssetAmountLeft']]
    top6asks = d[d.direction=='short'].groupby('priceRounded').sum()[['baseAssetAmountLeft']]

    tts = pd.concat([top6bids['baseAssetAmountLeft'].reset_index(drop=True), top6asks['baseAssetAmountLeft'].reset_index(drop=True)],axis=1)
    tts.columns = ['bs','as']
    # print(tts)
    min_q = (5000/mark_price)
    q = ((tts['bs']+tts['as'])/2).apply(lambda x: max(x, min_q)).max()
    # print('q=', q)
    score_scale = tts.min(axis=1)/q * 100
    score_scale = score_scale * pd.Series([2, .75, .5, .4, .3, .2]) #, .09, .08, .07])

    for i,x in enumerate(top6bids.index[:6]):
        ba = d.loc[(d.priceRounded==x)  & (d.direction=='long'), 'baseAssetAmountLeft']
        ba /= ba.sum()
        d.loc[(d.priceRounded==x)  & (d.direction=='long'), 'score'] = score_scale.values[i] * ba
    for i,x in enumerate(top6asks.index[:6]):
        ba = d.loc[(d.priceRounded==x)  & (d.direction=='short'), 'baseAssetAmountLeft']
        ba /= ba.sum()
        d.loc[(d.priceRounded==x) & (d.direction=='short'), 'score'] = score_scale.values[i] * ba
    
    return d


def get_mm_stats(df, user, oracle, bbo2):
    all_snap_slots = sorted(list(df.snap_slot.unique()))
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
    bbo_user = bbo_user.reindex(all_snap_slots).loc[bbo2.index[0]:bbo2.index[-1]]
    # bbo_user['score'] = df.groupby(['direction', 'snap_slot'])['score'].sum()
    bbo_user_score = df.groupby('snap_slot')['score'].sum().reindex(all_snap_slots).loc[bbo2.index[0]:bbo2.index[-1]]
    bbo_user = pd.concat([bbo_user, bbo_user_score],axis=1)
    bbo_user_avg_score = bbo_user['score'].fillna(0).mean()
    # if(float(bbo_user_avg_score) > 90):
    #     print(user)
    #     print(bbo_user['score'].describe())
    #     print(bbo_user['score'].fillna(0))

    near_threshold = .001

    try:
        uptime_pct = len(bbo_user.dropna())/len(bbo_user)
        bid_best_pct = (((bbo_user['best dlob bid']-bbo2['best dlob bid'])/bbo2['best dlob bid']) == 0).mean()
        bid_within_best_pct = (((bbo_user['best dlob bid']-bbo2['best dlob bid'])/bbo2['best dlob bid']) >= -near_threshold).mean()
        offer_best_pct = (((bbo_user['best dlob offer']-bbo2['best dlob offer'])/bbo2['best dlob offer']) == 0).mean()
        offer_within_best_pct = (((bbo_user['best dlob offer']-bbo2['best dlob offer'])/bbo2['best dlob offer']) <= near_threshold).mean()
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


    mol1, molselect, mol0, mol2 = st.columns([3, 3, 3, 10])
    market_index = mol1.selectbox('market index', [0, 1, 2, 3])
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
        df.to_csv('data/'+tt+'.csv.gz', index=False, compression='gzip')
    else:
        df = pd.read_csv('data/'+tt+'.csv.gz')

    df = df.reset_index(drop=True)
    oracle = df.groupby('snap_slot')['oraclePrice'].max()
    tabs = st.tabs(['bbo', 'leaderboard', 'individual mm', 'individual snapshot'])

    # st.write('slot range:', values)
    with tabs[0]:
        st.title('best bid/offer')
        do1, do2 = st.columns([3, 1])
        quote_trade_size = do1.number_input('trade size ($)', 0, None, 50000)
        do2 = do2.write('base size=' + str((quote_trade_size/oracle).max().max().round(4)))

    base_trade_size = (quote_trade_size/oracle).max().max().round(4)
    # print(base_trade_size)

    def wm(x):
        weights = df.loc[x.index, "baseAssetAmount"]

        direction = df.loc[x.index, "direction"]
        assert(len(direction.unique())==1)
        direction = direction.max()

        if direction == 'long':
            weights = weights.iloc[::-1]

        full_fills = 1*(weights.cumsum()<=base_trade_size)
        remainder = base_trade_size- (weights*full_fills).sum()
        partial_fill_index = full_fills[(full_fills==0)].index

        if len(partial_fill_index):
            partial_fill_index = partial_fill_index[0]
            full_fills.loc[partial_fill_index] = remainder/weights.loc[partial_fill_index]
        elif remainder != 0:
            full_fills *= np.nan
        weights = (weights * full_fills)#.fillna(0)
        if direction == 'long':
            weights = weights.iloc[::-1]

        return np.average(x, weights=weights)

    # print(df.groupby(['direction', 'snap_slot']).mean()[['price', 'baseAssetAmount']])
    bbo = df.groupby(['direction', 'snap_slot']).agg(
        baseAssetAmount=("baseAssetAmount", "sum"),
        max_price=("price", 'max'), 
        min_price=("price", 'min'), 
        price_weighted_mean=("price", wm),
    ).unstack(0)
    # print(bbo)
    
    bbo = bbo.swaplevel(axis=1)
    lmax = bbo['long']['max_price']
    smin = bbo['short']['min_price']
    lpwm = bbo['long']['price_weighted_mean']
    spwm = bbo['short']['price_weighted_mean']

    bbo2 = pd.concat([lpwm, lmax, oracle, smin, spwm],axis=1)
    bbo2.columns = ['short fill', 'best dlob bid', 'oracle', 'best dlob offer', 'long fill']
    last_slot_update = bbo2.index[-1]

    st.text('stats last updated at slot: ' + str(bbo2.index[-1]) +' (approx. '+ str(pd.to_datetime(slot_to_timestamp_est(last_slot_update)*1e9))+')')
   

    tzInfo = pytz.timezone('UTC')


    range_selected = molselect.selectbox('range select:', ['single day', 'slot range'], 0)
    if range_selected == 'single day':
        date = mol0.date_input('select approx. date:', min_value=datetime.datetime(2022,11,4), max_value=(datetime.datetime.now(tzInfo)))
        values = get_slots_for_date(date)
    else:
        values = mol2.slider(
        'Select a range of slot values',
        int(bbo2.index[0]), int(bbo2.index[-1]), (int(bbo2.index[0]), int(bbo2.index[-1])))
        mol2.write('approx date range: '+ str(list(pd.to_datetime([slot_to_timestamp_est(x)*1e9 for x in values]))))

    bbo2snippet = bbo2.loc[values[0]:values[1]]
    st.markdown('[data source](https://github.com/0xbigz/drift-v2-orderbook-snap)'+\
        ' ([slot='+str(bbo2snippet.index[0])+'](https://github.com/0xbigz/drift-v2-orderbook-snap/blob/main/'+tt+'/orderbook_slot_'+str(bbo2snippet.index[0])+'.csv))')


    # st.write('slot range:', values)
    with tabs[0]:
        plot1, plot0, plot2 = st.columns([4, 1, 4])
        plot1.plotly_chart(bbo2snippet.plot(title='perp market index='+str(market_index)))

        df1 = pd.concat({
            'buy impact': (bbo2snippet['long fill'] - bbo2snippet['best dlob offer'])/bbo2snippet['best dlob offer'],
            'sell impact': (bbo2snippet['short fill'] - bbo2snippet['best dlob bid'])/bbo2snippet['best dlob bid'],
        },axis=1)*100
        fig = df1.plot(title='perp market index='+str(market_index))
        fig.update_layout(
                    yaxis_title="Price Impact (%)",
                    legend_title="Trade Impact",
                )
        plot2.plotly_chart(fig)


    with tabs[1]:
        all_stats = []
        score_emas = {}
        for user in df.user.unique():
            bbo_user, bbo_user_stats = get_mm_stats(df, user, oracle, bbo2snippet)
            score_emas[str(user)] = (bbo_user['score'].fillna(0).ewm(100).mean())
            all_stats.append(bbo_user_stats)

        st.title('mm leaderboard')
        all_stats_df = pd.concat(all_stats, axis=1).T.sort_values('best_bid%', ascending=False)
        st.dataframe(all_stats_df)
        topmm = all_stats_df.index.to_list()
        # print(topmm)
        st.plotly_chart(pd.concat(score_emas, axis=1)[topmm[:10]].fillna(0).plot())

    with tabs[2]:

        st.title('individual mm lookup')

        user = st.selectbox('individual maker', df.user.unique())
            
        bbo_user, bbo_user_stats = get_mm_stats(df, user, oracle, bbo2)

        # st.text('user bestbid time='+str(np.round(offer_best_pct*100, 2))+'%')
        # st.text('user bid within 3bps of best time='+str(np.round(offer_within_best_pct*100, 2))+'%')
        # st.text('user bestoffer time='+str(np.round(offer_best_pct*100, 2))+'%')
        # st.text('user offer within 3bps of best time='+str(np.round(offer_within_best_pct*100, 2))+'%')
        # st.text('user uptime='+str(np.round(uptime_pct*100, 2))+'%')
        bbo_user['score'] = bbo_user['score'].fillna(0)
        bbo_user['ema_score'] = bbo_user['score'].fillna(0).ewm(100).mean()
        st.plotly_chart(bbo_user.loc[values[0]:values[1]].plot(title='perp market index='+str(market_index)))

    with tabs[3]:
        st.title('individual snapshot lookup')
        # print(df.columns)

        sol1, sol0, sol2 = st.columns([2,1,2])
        slot = sol1.select_slider('individual snapshot', df.snap_slot.unique().tolist(), df.snap_slot.max())

        slippage = sol2.select_slider('slippage (%)', list(range(1, 100)), 5)

        toshow = df[['score', 'price', 'priceRounded', 'baseAssetAmountLeft', 'direction', 'user', 'status', 'orderType',
        'marketType', 'baseAssetAmount', 'marketIndex',  'oraclePrice', 'slot', 'snap_slot', 'orderId', 'userOrderId', 
        'baseAssetAmountFilled', 'quoteAssetAmountFilled', 'reduceOnly',
        'triggerPrice', 'triggerCondition', 'existingPositionDirection',
        'postOnly', 'immediateOrCancel', 'oraclePriceOffset', 'auctionDuration',
        'auctionStartPrice', 'auctionEndPrice', 'maxTs', 
        ]]

        toshow_snap = toshow[df.snap_slot.astype(int)==int(slot)]
        bids = toshow_snap[toshow_snap.direction=='long'].groupby('price').sum().sort_index(ascending=False)['baseAssetAmountLeft'].cumsum()
        asks = toshow_snap[toshow_snap.direction=='short'].groupby('price').sum().sort_index(ascending=True)['baseAssetAmountLeft'].cumsum()

        markprice = (bids.index.max()+asks.index.min()) /2
        ddd = pd.concat({'bids':bids, 'asks':asks},axis=1).sort_index().loc[markprice*(1-slippage/100): markprice*(1+slippage/100)].replace(0, np.nan)
        fig = ddd.plot(kind='line', title='book depth')
        st.plotly_chart(fig)
        st.dataframe(toshow_snap)


            
        