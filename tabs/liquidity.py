
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"
import datetime
import pytz
import time

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.addresses import get_user_account_public_key
import os
import json
import streamlit as st
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
from glob import glob
import requests
from tabs.uservolume import load_volumes

def load_realtime_book(market_index):
    x = requests.get('https://dlob.drift.trade/orders/json')
    
    try:
        x = x.json()
    except:
        print(x)
        
    market_to_oracle_map = pd.DataFrame(x['oracles']).set_index('marketIndex').to_dict()['price']
    market_to_oracle_map

    df = pd.DataFrame([order['order'] for order in x['orders']])
    user = pd.DataFrame([order['user'] for order in x['orders']], columns=['user'])
    df = pd.concat([df, user],axis=1)
    df['oraclePrice'] = None
    df.loc[df.marketType=='perp', 'oraclePrice'] = df.loc[df.marketType=='perp', 'marketIndex'].apply(lambda x: market_to_oracle_map.get(x, 0))

    #todo, invariant may change w/ future spot markets
    df.loc[df.marketType=='spot', 'oraclePrice'] = df.loc[df.marketType=='spot', 'marketIndex'].apply(lambda x: market_to_oracle_map.get(int(x)-1, 0))

    df1 = df[(df.orderType=='limit')]
    df1.loc[((df1['price'].astype(int)==0) & (df1['oraclePrice'].astype(int)!=0)), 'price'] = df1['oraclePrice'].astype(int) + df1['oraclePriceOffset'].astype(int)

    for col in ['price', 'oraclePrice', 'oraclePriceOffset']:
        df1[col] = df1[col].astype(int)
        df1[col] /= 1e6
        
    for col in ['quoteAssetAmountFilled']:
        df1[col] = df1[col].astype(int)
        df1[col] /= 1e6 

    for col in ['baseAssetAmount', 'baseAssetAmountFilled']:
        df1[col] = df1[col].astype(int)
        df1[col] /= 1e9
        

    # market_types = sorted(df.marketType.unique())
    # market_indexes = sorted(df.marketIndex.unique())
    
    mdf = df1[((df1.marketType=='perp') & (df1.marketIndex==market_index))]
    if len(mdf)==0:
        return mdf

    mdf = mdf[['price', 'baseAssetAmount', 'direction', 'user', 'status', 'orderType', 'marketType', 'slot', 'orderId', 'userOrderId',
'marketIndex',  'baseAssetAmountFilled',
'quoteAssetAmountFilled',  'reduceOnly', 'triggerPrice',
'triggerCondition', 'existingPositionDirection', 'postOnly',
'immediateOrCancel', 'oraclePriceOffset', 'auctionDuration',
'auctionStartPrice', 'auctionEndPrice', 'maxTs', 'oraclePrice']]
    mdf = mdf.sort_values('price').reset_index(drop=True)

    return mdf



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

    mbps = .0005 # 5 bps
    mbpsA = mbps/5
    mbpsB = mbps*3/5

    # within_bps_of_price = (mark_price * mbps)

    # if market_index == 1 or market_index == 2:
    #     within_bps_of_price = (mark_price * .00025)

    def rounded_threshold(x, direction):
        within_bps_of_price = (mark_price * mbps)
        if direction == 'long':
            if x >= best_bid*(1-mbpsA):
                within_bps_of_price = (mark_price * mbpsA)
            elif x >= best_bid*(1-mbpsB):
                within_bps_of_price = (mark_price * mbpsB)
            res = np.floor(float(x)/within_bps_of_price) * within_bps_of_price
        else:
            if x <= best_ask*(1+mbpsA):
                within_bps_of_price = (mark_price * mbpsA)
            elif x <= best_ask*(1+mbpsB):
                within_bps_of_price = (mark_price * mbpsB)

            res = np.ceil(float(x)/within_bps_of_price) * within_bps_of_price

        return res

    d['priceRounded'] = d.apply(lambda x: rounded_threshold(x['price'], x['direction']), axis=1)
    d['level'] = np.nan
    # print(d)

    top6bids = d[d.direction=='long'].groupby('priceRounded').sum().sort_index(ascending=False)[['baseAssetAmountLeft']]
    top6asks = d[d.direction=='short'].groupby('priceRounded').sum()[['baseAssetAmountLeft']]

    tts = pd.concat([top6bids['baseAssetAmountLeft'].reset_index(drop=True), top6asks['baseAssetAmountLeft'].reset_index(drop=True)],axis=1)
    tts.columns = ['bs','as']
    # print(tts)
    min_q = (5000/mark_price)
    q = ((tts['bs']+tts['as'])/2).apply(lambda x: max(x, min_q)).max()
    # print('q=', q)
    score_scale = tts.min(axis=1)/q * 100
    # target bps of for scoring [1,3,5,10,15,20]

    score_scale = score_scale * pd.Series([2, .75, .5, .4, .3, .2]) #, .09, .08, .07])
    chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for i,x in enumerate(top6bids.index[:6]):
        char = chars[i]
        ba = d.loc[(d.priceRounded==x)  & (d.direction=='long'), 'baseAssetAmountLeft']
        ba /= ba.sum()
        d.loc[(d.priceRounded==x)  & (d.direction=='long'), 'score'] = score_scale.values[i] * ba
        d.loc[(d.priceRounded==x)  & (d.direction=='long'), 'level'] = char+'-bid'
    for i,x in enumerate(top6asks.index[:6]):
        char = chars[i]
        ba = d.loc[(d.priceRounded==x)  & (d.direction=='short'), 'baseAssetAmountLeft']
        ba /= ba.sum()
        d.loc[(d.priceRounded==x) & (d.direction=='short'), 'score'] = score_scale.values[i] * ba
        d.loc[(d.priceRounded==x) & (d.direction=='short'), 'level'] = char+'-ask'
    
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
    bbo_user_score = df.groupby('snap_slot')[['score', 'baseAssetAmountLeft']].sum()\
        .reindex(all_snap_slots).loc[bbo2.index[0]:bbo2.index[-1]]
    bbo_user = pd.concat([bbo_user, bbo_user_score],axis=1)
    bbo_user_avg_score = bbo_user['score'].fillna(0).mean()
    bbo_user_sizes = bbo_user[bbo_user['score']>0]['baseAssetAmountLeft'].fillna(0)
    bbo_user_median_size = bbo_user_sizes.median()
    bbo_user_min_size = bbo_user_sizes.min()
    # if(float(bbo_user_avg_score) > 90):
    #     print(user)
    #     print(bbo_user['score'].describe())
    #     print(bbo_user['score'].fillna(0))

    near_threshold = .002 # 20 bps

    try:
        bid_up = (((bbo_user['best dlob bid']-bbo2['best dlob bid'])/bbo2['best dlob bid']))
        ask_up = ((bbo_user['best dlob offer']-bbo2['best dlob offer'])/bbo2['best dlob offer']) 
        availability_pct = len(bbo_user.dropna())/len(bbo_user)
        bid_best_pct = (((bbo_user['best dlob bid']-bbo2['best dlob bid'])/bbo2['best dlob bid']) == 0).mean()
        bid_within_best_pct = (bid_up >= -near_threshold).mean()
        offer_best_pct = (((bbo_user['best dlob offer']-bbo2['best dlob offer'])/bbo2['best dlob offer']) == 0).mean()
        offer_within_best_pct = (ask_up <= near_threshold).mean()
        uptime_pct = ((bid_up >= -near_threshold) & (ask_up <= near_threshold)).mean()
        bbo_user['bid_vs_best(%)'] = bid_up.astype(float)
        bbo_user['ask_vs_best(%)'] = ask_up.astype(float)
    except:
        bid_up = 0
        ask_up = 0
        uptime_pct = 0
        availability_pct = 0
        bid_best_pct = 0
        bid_within_best_pct = 0
        offer_best_pct = 0
        offer_within_best_pct = 0
    bbo_stats = pd.DataFrame(
        [[availability_pct, uptime_pct, bid_best_pct, bid_within_best_pct, offer_best_pct, offer_within_best_pct, 
        bbo_user_avg_score/100, bbo_user_median_size/100, bbo_user_min_size/100]],
        index=[user],
        columns=['availability%', 'uptime%', 'best_bid%', 'uptime_bid%', 'best_offer%', 'uptime_offer%', 'avg score', 'median size', 'min size']
        ).T * 100

    return bbo_user, bbo_stats




@st.cache_data(ttl=3600*2)  # 2 hr TTL this time
def get_data_by_market_index(market_type, market_index, source):
    dfs = []
    tt = market_type+str(market_index)
    # ggs = glob('../drift-v2-orderbook-snap/'+tt+'/*.csv')

    df = None
    if False:
        print('building new data file with', len(ggs), 'records!')
        for x in sorted(ggs)[-3500:]:
            df = pd.read_csv(x) 
            df['snap_slot'] = int(x.split('_')[-1].split('.')[0])
            df = get_mm_score_for_snap_slot(df)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        df.to_csv('data/'+tt+'.csv.gz', index=False, compression='gzip')
    else:
        print('reading csv')
        if source == '':
            source = 'data/'+tt+'.csv.gz'
        else:
            source = source+tt+'.csv.gz'
        # df = pd.read_csv('data/'+tt+'.csv.gz')
        df = pd.read_csv(source)
    df = df.reset_index(drop=True)
    return df

async def mm_page(clearing_house: DriftClient):    
    st.warning('historical data only. this data source for snapshot is in the process of being migrated')
    ss1, ss2, ss3 = st.columns([3,1,1])
    best_default_source = 'https://github.com/0xbigz/drift-v2-orderbook-scored/raw/main/data/'
    # best_default_source = 'https://github.com/ansatz-research/drift-v2-orderbook-scored/raw/main/data/data/'
    source = ss1.text_input('source:', best_default_source)
    
    ss = json.loads((await clearing_house.program.provider.connection.get_slot()).to_json())['result']
    tt = json.loads((await clearing_house.program.provider.connection.get_block_time(ss)).to_json())['result']
    SLOT1 = ss2.number_input('slot ref:', min_value=0, value=ss)
    ss2.write('https://explorer.solana.com/block/'+str(SLOT1))

    TS1 = ss3.number_input('unix_timestamp ref:', min_value=0, value=tt)
    def slot_to_timestamp_est(slot, ts_for_latest_slot=None, latest_slot=None):
        # ts = X - (Y - slot)/2
        # X - ts = (Y - slot)/2
        # slot = Y - 2 * (X - ts)
        if ts_for_latest_slot is not None:
            return ts_for_latest_slot - (latest_slot-slot) * .484

        return TS1 - (SLOT1-slot) * .484
        return 1681152470 - (187674632-slot) * .484
        # return 1679940031 - (185018514-slot) * .484
        # return 1678888672 - (182733090-slot) * .484
        # return 1676400329 - (177774625-slot)*.5

    def get_slots_for_date(date: datetime.date):
        utimestamp = (date - datetime.date(1970,1,1)).total_seconds()
        start = SLOT1-2*(TS1-utimestamp)
        end = start + 60*60*24*2
        return [start, end]


    mol00, mol1, molselect, mol0, mol2 = st.columns([2, 3, 3, 3, 10])

    market_type = mol00.selectbox('market type', ['perp', 'spot'])
    market_indexes = list(range(1,7))
    if market_type == 'perp':
        market_indexes = list(range(0,19))
    market_index = mol1.selectbox(market_type+' market index', market_indexes)
    now = datetime.datetime.now()
    current_ts = time.mktime(now.timetuple())
    tt = market_type+str(market_index)
    df_full = get_data_by_market_index(market_type, market_index, source if 'data' in source else '')

    # all_slots = df_full.snap_slot.unique()
    oldest_slot = df_full.snap_slot.min()
    newest_slot = df_full.snap_slot.max()

    # load_volumes([now], 'SOL-PERP')
    latest_slot = None
    ts_for_latest_slot = None
    # latest_slot = s2.number_input('latest slot:', value=newest_slot)
    # ts_for_latest_slot = s3.number_input('latest slot ts:')
    # s3.write('current ts='+ str(int(current_ts)))
    # if ts_for_latest_slot == 0:
        # ts_for_latest_slot = None

    week_ago_slot = df_full[df_full.snap_slot>newest_slot-(2*60*60*24*7)].snap_slot.min()

    oracle = df_full.groupby('snap_slot')['oraclePrice'].max()
    tabs = st.tabs(['bbo', 'leaderboard', 'individual mm', 'individual snapshot', 'real time', 'csv reader', 'program tx'])

    # st.write('slot range:', values)
    with tabs[0]:
        st.title('best bid/offer')
        do1, do2, do3 = st.columns([3, 1, 1])
        quote_trade_size = do1.number_input('trade size ($)', 0, None, 50000)
        do2 = do2.write('base size=' + str((quote_trade_size/oracle).max().max().round(4)))
        threshold = do3.slider('threshold (bps)', 0, 100, 20, step=5)


    base_trade_size = (quote_trade_size/oracle).max().max().round(4)
    # print(base_trade_size)

    def wm(x):
        weights = df.loc[x.index, "baseAssetAmount"]

        direction = df.loc[x.index, "direction"]
        # print(direction)
        if(len(direction)==0):
            return np.nan

        # assert(len(direction.unique())==1)
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

    tzInfo = pytz.timezone('UTC')
    latest_slot_full = df_full.snap_slot.max()

    range_selected = molselect.selectbox('range select:', ['daily', 'range', 'weekly', 'last month'], 0)
    if range_selected == 'daily':
        lastest_date = pd.to_datetime(slot_to_timestamp_est(latest_slot_full, ts_for_latest_slot, latest_slot)*1e9, utc=True)
        date = mol0.date_input('select approx. date:', lastest_date, min_value=datetime.datetime(2022,11,4), max_value=lastest_date) #(datetime.datetime.now(tzInfo)))
        values = get_slots_for_date(date)
    elif range_selected == 'range':
        start_slot = mol2.number_input('start slot:', oldest_slot, newest_slot, oldest_slot)
        end_slot = mol2.number_input('end slot:', oldest_slot, newest_slot, newest_slot)
        values = [start_slot, end_slot]
        mol2.write('approx date range: '+ str(list(pd.to_datetime([slot_to_timestamp_est(x, ts_for_latest_slot, latest_slot)*1e9 for x in values]))))
    elif range_selected == 'weekly':
        values = mol2.slider(
        'Select a range of slot values',
        int(week_ago_slot), int(newest_slot), (int(week_ago_slot), int(newest_slot)))
        mol2.write('approx date range: '+ str(list(pd.to_datetime([slot_to_timestamp_est(x, ts_for_latest_slot, latest_slot)*1e9 for x in values]))))
    elif range_selected == 'last month':
        values = mol2.slider(
        'Select a range of slot values',
        int(oldest_slot), int(newest_slot), (int(oldest_slot), int(newest_slot)))
        mol2.write('approx date range: '+ str(list(pd.to_datetime([slot_to_timestamp_est(x, ts_for_latest_slot, latest_slot)*1e9 for x in values]))))

    df = df_full[(df_full.snap_slot>=values[0]) & (df_full.snap_slot<=values[1])]
    # print(df_full.snap_slot.max(), 'vs', values[0], values[1])
    st.write(df_full.snap_slot.max())
    assert(df_full.snap_slot.max() >= values[0])
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

    bbo2 = pd.concat([lpwm, lmax, df.groupby('snap_slot')['oraclePrice'].max(), smin, spwm],axis=1)
    bbo2.columns = ['short fill', 'best dlob bid', 'oracle', 'best dlob offer', 'long fill']
    last_slot_update = bbo2.index[-1]

    st.text('stats last updated at slot: ' + str(bbo2.index[-1]) +' (approx. '+ str(pd.to_datetime(slot_to_timestamp_est(last_slot_update, ts_for_latest_slot, latest_slot)*1e9))+')')
   


    bbo2snippet = bbo2#.loc[values[0]:values[1]]
    st.markdown('[data source](https://github.com/0xbigz/drift-v2-orderbook-snap)'+\
        ' ([start slot='+str(bbo2snippet.index[0])+'](https://github.com/0xbigz/drift-v2-orderbook-snap/blob/main/'+tt+'/orderbook_slot_'+str(bbo2snippet.index[0])+'.csv))'+\
        ' ([end slot='+str(bbo2snippet.index[-1])+'](https://github.com/0xbigz/drift-v2-orderbook-snap/blob/main/'+tt+'/orderbook_slot_'+str(bbo2snippet.index[-1])+'.csv))')

    # st.write('slot range:', values)
    with tabs[0]:
        (summarytxt,summarytxt2) = st.columns(2)
        plot1, plot0, plot2 = st.columns([4, 1, 4])
        plot1.plotly_chart(bbo2snippet.plot(title=market_type+' market index='+str(market_index)))
        df1 = pd.concat({
            'buy offset + impact': (bbo2snippet['long fill'] - bbo2snippet['oracle'])/bbo2snippet['oracle'],
            'buy impact': (bbo2snippet['long fill'] - bbo2snippet['best dlob offer'])/bbo2snippet['best dlob offer'],
            'sell impact': (bbo2snippet['short fill'] - bbo2snippet['best dlob bid'])/bbo2snippet['best dlob bid'],
            'sell offset + impact': (bbo2snippet['short fill'] - bbo2snippet['oracle'])/bbo2snippet['oracle'],

        },axis=1)*100

        buy_pct_within = ((df1['buy impact']<threshold/100)*1).sum()/len(df1['buy impact'])
        sell_pct_within = ((df1['sell impact']>-threshold/100)*1).sum()/len(df1['sell impact'])
        both_pct_within = (((df1['buy impact']<threshold/100) & (df1['sell impact']>-threshold/100))*1).sum()/len(df1['buy impact'])
        
        
        buy_pct_within2 = ((df1['buy offset + impact']<threshold/100)*1).sum()/len(df1['buy offset + impact'])
        sell_pct_within2 = ((df1['sell offset + impact']>-threshold/100)*1).sum()/len(df1['sell offset + impact'])
        both_pct_within2 = (((df1['buy offset + impact']<threshold/100) & (df1['sell offset + impact']>-threshold/100))*1).sum()/len(df1['sell offset + impact'])
        
        fig = df1[['buy impact', 'sell impact']].plot(title='perp market index='+str(market_index))
        fig.update_layout(
                    yaxis_title="Price Impact (%)",
                    legend_title="Trade Impact",
                )
        fig.add_hline(y=threshold/100, line_width=3, line_dash="dash", line_color="green")
        fig.add_hline(y=-threshold/100, line_width=3, line_dash="dash", line_color="green")

        plot2.plotly_chart(fig)

        summarytxt.metric("fill vs oracle within threshold", 
        ' '+str(np.round(both_pct_within2*100, 1))+' % of time',
        'buys='+str(np.round(buy_pct_within2*100, 1))+'%, sells='+str(np.round(sell_pct_within2*100, 1))+'%')

        summarytxt2.metric("price impact within threshold", 
        ' '+str(np.round(both_pct_within*100, 1))+' % of time',
        'buys='+str(np.round(buy_pct_within*100, 1))+'%, sells='+str(np.round(sell_pct_within*100, 1))+'%')
        
    all_users = df.groupby('user')['score'].sum().sort_values(ascending=False).index
    top10users = all_users[:10]

    with tabs[1]:
        all_stats = []
        score_emas = {}
        # top10users = df.groupby('user')['score'].sum().sort_values(ascending=False).head(10).index
        st.title('mm leaderboard')
        metmet1, metemet2 = st.columns(2)
        users = st.multiselect('users:', list(all_users), list(top10users))
        [lbtable] = st.columns([1])
        [eslid, s2] = st.columns([1, 1])
        [echart] = st.columns([1])

        rolling_window = eslid.slider('rolling window:', 1, 200, 100, 5)
        use_ts_est = s2.selectbox('index:', ['snap_slot', 'est_utc_timestamp'], key='tab1', help='est utc uses "slot ref:" and "unix_timestamp_ref" to deduce timestamps')
        for user in users:
            bbo_user, bbo_user_stats = get_mm_stats(df, user, oracle, bbo2snippet)
            score_emas[str(user)] = (bbo_user['score'].fillna(0).rolling(rolling_window, min_periods=min(rolling_window, 1)).mean())
            all_stats.append(bbo_user_stats)

        all_stats_df = pd.concat(all_stats, axis=1).T.sort_values('avg score', ascending=False)
        metmet1.metric('total avg score:', np.round(all_stats_df['avg score'].sum(),2))
        lbtable.dataframe(all_stats_df)
        # print(topmm)
        dat = pd.concat(score_emas, axis=1)[users].fillna(0)
        if use_ts_est == 'est_utc_timestamp':
            dat.index = [pd.to_datetime(slot_to_timestamp_est(x, ts_for_latest_slot, latest_slot)*1e9, utc=True) for x in dat.index]
        echart.plotly_chart(dat.plot(), use_container_width=True)

    with tabs[2]:

        st.title('individual mm lookup')

        o0, o1, o2, o3 = st.columns(4)
        o0.write('look up your user account address using your authority + subaccount number')
        authority_l = o1.text_input('wallet address')
        sub_id_l = o2.number_input('subaccount id')
        o3.write(get_user_account_public_key(clearing_house.program_id,
                    authority_l,
                    sub_id_l))
        
        s1, s2 = st.columns(2)
        user = s1.selectbox('individual maker', all_users, 0, help='maker address is drift user account address (not wallet address)')
        use_ts_est = s2.selectbox('index:', ['snap_slot', 'est_utc_timestamp'], help='est utc uses "slot ref:" and "unix_timestamp_ref" to deduce timestamps')
        bbo_user, bbo_user_stats = get_mm_stats(df, user, oracle, bbo2)

        # st.text('user bestbid time='+str(np.round(offer_best_pct*100, 2))+'%')
        # st.text('user bid within 3bps of best time='+str(np.round(offer_within_best_pct*100, 2))+'%')
        # st.text('user bestoffer time='+str(np.round(offer_best_pct*100, 2))+'%')
        # st.text('user offer within 3bps of best time='+str(np.round(offer_within_best_pct*100, 2))+'%')
        # st.text('user uptime='+str(np.round(uptime_pct*100, 2))+'%')
        bbo_user['score'] = bbo_user['score'].fillna(0)
        bbo_user['ema_score'] = bbo_user['score'].fillna(0).ewm(100).mean()
        dat = bbo_user.loc[values[0]:values[1]]
        if use_ts_est == 'est_utc_timestamp':
            dat.index = [pd.to_datetime(slot_to_timestamp_est(x, ts_for_latest_slot, latest_slot)*1e9, utc=True) for x in dat.index]
        st.plotly_chart(dat.plot(title=market_type+' market index='+str(market_index)))

        best_slot = bbo_user.score.idxmax()
        worst_slot = bbo_user.score.idxmin()
        st.write('best slot:'+str(best_slot), ' | worst slot:'+str(worst_slot))

    def cooling_highlight(val):
        color = '#ACE5EE' if val=='long' else 'pink'
        return f'background-color: {color}'

    def level_highlight(val):
        kk = str(val).split('-')[0]
        gradient = {
            'A': '#FFA500',  # orange
            'B': '#FFB347',
            'C': '#FFC58B',
            'D': '#FFE0AD',
            'E': '#FFFFE0',
            'F': '#FFFF00'   # yellow
        }

        color = gradient.get(kk, None)

        return f'background-color: {color}'

    with tabs[3]:
        st.title('individual snapshot lookup')
        # print(df.columns)

        sol1, sol10, sol11, sol0, sol2 = st.columns([2,1,1, 1,2])
        slot = sol1.select_slider('individual snapshot', df.snap_slot.unique().tolist(), df.snap_slot.max())
        slot = sol10.number_input('slot:', value=slot)
        score_color = sol11.radio('show score highlights:', [True, False], 0)

        slippage = sol2.select_slider('slippage (%)', list(range(1, 100)), 5)
        toshow = df[['level', 'score', 'price', 'priceRounded', 'baseAssetAmountLeft', 'direction', 'user', 'status', 'orderType',
        'marketType', 'baseAssetAmount', 'marketIndex',  'oraclePrice', 'slot', 'snap_slot', 'orderId', 'userOrderId', 
        'baseAssetAmountFilled', 'quoteAssetAmountFilled', 'reduceOnly',
        'triggerPrice', 'triggerCondition', 'existingPositionDirection',
        'postOnly', 'immediateOrCancel', 'oraclePriceOffset', 'auctionDuration',
        'auctionStartPrice', 'auctionEndPrice', 'maxTs', 
        ]]

        toshow_snap = toshow[df.snap_slot.astype(int)==int(slot)]

        st.metric('total score:', np.round(toshow_snap.score.sum(),2))

        bids1 = toshow_snap[toshow_snap.direction=='long'].groupby('price').sum().sort_index(ascending=False)
        bids = bids1['baseAssetAmountLeft'].cumsum()
        bscores = bids1['score'].dropna()#.cumsum()
        asks1 = toshow_snap[toshow_snap.direction=='short'].groupby('price').sum().sort_index(ascending=True)
        asks = asks1['baseAssetAmountLeft'].cumsum()
        ascores = asks1['score'].dropna()#.cumsum()
        markprice = (bids.index.max()+asks.index.min()) /2

        oh = pd.concat({'bid scores': bscores, 'ask scores': ascores},axis=1).sort_index().replace(0, np.nan).dropna(how='all')
        fig2 = oh.plot(kind='line', title='book scores')

        ddd = pd.concat({'bids':bids, 'asks':asks},axis=1).sort_index().loc[markprice*(1-slippage/100): markprice*(1+slippage/100)].replace(0, np.nan)
        fig = ddd.plot(kind='line', title='book depth')

        
        [plotly1, plotly2] = st.columns(2)
        plotly1.plotly_chart(fig, use_container_width=True)
        plotly2.plotly_chart(fig2, use_container_width=True)

        if score_color:
            toshow_snap = toshow_snap.replace(0, np.nan).dropna(subset=['score'],axis=0)


        st.dataframe(toshow_snap.style.applymap(cooling_highlight, subset=['direction'])
                .applymap(level_highlight, subset=['level'])
                    #  ,(heating_highlight, subset=['Heating inputs', 'Heating outputs'])             
                     , use_container_width=True)


    with tabs[4]:
        warning_text = ''
        st_tscore, liveness_switch = st.columns([4, 1])
        st_tscore.button('refresh')
        is_live = liveness_switch.radio('liveness', ['on', 'off'], index=1, horizontal=True)
        def count_down(ts, live=False):
            def do_it():
                now = datetime.datetime.now()
                try:
                    df = load_realtime_book(market_index)
                except Exception as e:
                    warning_text = 'load_realtime_book error: ' + str(e)
                    if warning_text:
                        st.warning(warning_text)
                    df = pd.DataFrame()
                df['snap_slot'] = 'current'
                scored_df = get_mm_score_for_snap_slot(df)
                scored_df = scored_df.dropna(subset=['score'])
                toshow = scored_df[['level', 'score', 'price', 'priceRounded', 'baseAssetAmountLeft', 'direction', 'user', 'status', 'orderType',
                'marketType', 'baseAssetAmount', 'marketIndex',  'oraclePrice', 'slot', 'snap_slot', 'orderId', 'userOrderId', 
                'baseAssetAmountFilled', 'quoteAssetAmountFilled', 'reduceOnly',
                'triggerPrice', 'triggerCondition', 'existingPositionDirection',
                'postOnly', 'immediateOrCancel', 'oraclePriceOffset', 'auctionDuration',
                'auctionStartPrice', 'auctionEndPrice', 'maxTs', 
                ]]
                st.metric('total score:', np.round(toshow['score'].sum(), 2))
                st.write('as of: ' + str(now))
                st.table(toshow.groupby('user')['score'].sum().sort_values(ascending=False))
                st.dataframe(toshow.style.applymap(cooling_highlight, subset=['direction'])
                        .applymap(level_highlight, subset=['level'])  , use_container_width=True)

            if live:
                with st.empty():
                     while ts:
                        # mins, secs = divmod(ts, 60)
                        # time_now = '{:02d}:{:02d}'.format(mins, secs)
                        # st.header(f"{time_now}")
                        do_it()
                        time.sleep(10)
                        ts -= 1
            else:
                do_it()
        time_in_seconds = 60
        if is_live == 'on':
            count_down(int(time_in_seconds), True)
        else:
            count_down(int(time_in_seconds))

    with tabs[5]:
        url = st.text_input('url to csv:', '')
        if url!='':
            df = pd.read_csv(url)
            st.dataframe(df)

    with tabs[6]:
        dcol = 3
        acol = 6
        hist_df = pd.read_csv('https://raw.githubusercontent.com/drift-labs/transaction-tables/master/mm_rewards.csv',
                              
                              parse_dates=[dcol], index_col=[dcol]).sort_index()
        
        st.metric('total mm program tx:', '$' + str(hist_df.iloc[:, acol].sum()))
        st.plotly_chart(hist_df.iloc[:, acol].cumsum().plot())