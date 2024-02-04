
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
from driftpy.constants.spot_markets import mainnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import mainnet_perp_market_configs, PerpMarketConfig
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
    # market_to_oracle_map

    df = pd.DataFrame([order['order'] for order in x['orders']])
    user = pd.DataFrame([order['user'] for order in x['orders']], columns=['user'])
    df = pd.concat([df, user],axis=1)
    if 'oraclePrice' not in df.columns:
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



def get_mm_score_for_slot(df):
    d = df
    #df[(df.orderType=='limit')
    # & (df.postOnly)
    #]
    d['baseAssetAmountLeft'] = d['baseAssetAmount'] - d['baseAssetAmountFilled']
    assert(len(d.slot.unique())==1)

    market_index = d.marketIndex.max()
    best_bid = d[d.direction=='long']['price'].max()
    best_ask = d[d.direction=='short']['price'].min()
    oracle = (best_ask+best_bid)/2

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
    all_slots = sorted(list(df.slot.unique()))
    df = df[df.user == user]
    print(df.columns)

    bbo = df.groupby(['direction', 'slot']).agg({'price':['min', 'max']}).unstack(0)['price'].swaplevel(axis=1)
    ll = {}
    if 'long' in bbo.columns:
        lmax = bbo['long']['max']
        ll['best dlob bid'] = (lmax)
    ll['oracle'] = (oracle)
    if 'short' in bbo.columns:
        smin = bbo['short']['min']
        ll['best dlob offer'] = (smin)

    bbo_user = pd.concat(ll,axis=1).reindex(ll['oracle'].index)
    bbo_user = bbo_user.reindex(all_slots).loc[bbo2.index[0]:bbo2.index[-1]]
    # bbo_user['score'] = df.groupby(['direction', 'slot'])['score'].sum()
    bbo_user_score = df.groupby('slot')[['score', 'baseAssetAmountLeft']].sum()\
        .reindex(all_slots).loc[bbo2.index[0]:bbo2.index[-1]]
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

def load_scored_snapshot_file(x, source):
    #liq-score-${marketType}-${marketIndex}-${slot}.csv.g
    ff = source+x
    # st.warning(ff)
    try:
        df = pd.read_csv(ff)
    except:
        st.warning('couldnt load: ' + ff)
        df = pd.DataFrame()

    if 'oraclePrice' not in df.columns and 'price' in df.columns:
        df['oraclePrice'] = df[df['price']!=0].dropna().median() # todo

    return df

def load_snapshot_file(x, source):
    json_list = requests.get(source+x).json()
    rows = []
    for i, r in enumerate(json_list['orders']):
        # st.write(r)
        df = pd.DataFrame(r['order'], index=[i])
        df['user'] = r['user']
        rows.append(df)

    df = pd.concat(rows)
    df['slot'] = int(x.split('/')[-1].split('.')[0])

    # df['oraclePrice'] = df['price'].astype(float) # todo
    # df['oracle'] = df['price'].astype(float) # todo

    for col in ['price', 'baseAssetAmount', 'baseAssetAmountFilled']:
        df[col] = df[col].astype(float)

    df = get_mm_score_for_slot(df)
    return df


@st.cache_data(ttl=3600*2)  # 2 hr TTL this time
def get_s3_slots_for_dates(source, env, dates):
    all_slots = []
    for date in dates:
        url = f'{source}{env}/{date}/index.json.gz'
        res = get_index_json_from_s3(url)
        slots = [slot.split('/')[-1].split('.')[0] for slot in res]
        all_slots.extend(slots)

    return all_slots


@st.cache_data(ttl=3600*2)  # 2 hr TTL this time
def get_index_json_from_s3(url):
    return requests.get(url).json()


@st.cache_data(ttl=3600*2)  # 2 hr TTL this time
def get_data_by_market_index(market_type, market_index, env, date, source, url_agg):
    dfs = []
    tt = market_type+str(market_index)
    # ggs = glob('../drift-v2-orderbook-snap/'+tt+'/*.csv')
    url = source+env+'/'+date+'/index.json.gz'
    # liq-score-index.json.gz
    ggs = get_index_json_from_s3(url)
    with st.expander('data source: ' + url):
        st.write(url_agg)
        st.json(ggs, expanded=False)

    df = None
    if True:
        print('building new data file with', len(ggs), 'records!')
        for x in sorted(ggs)[-10:]:
            new_x = x.split('/')
            new_x[-1] = f'liq-score-{market_type}-{str(market_index)}-'+new_x[-1]
            new_x = '/'.join(new_x).replace('json.gz', 'csv.gz')
            # st.write(new_x)s
            df = load_scored_snapshot_file(new_x, source)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        # df.to_csv('data/'+tt+'.csv.gz', index=False, compression='gzip')
    else:
        print('reading csv')
        if source == '':
            source = 'data/'+tt+'.csv.gz'
        else:
            source = source+tt+'.csv.gz'
        # df = pd.read_csv('data/'+tt+'.csv.gz')
        df = pd.read_csv(source)
    df = df.reset_index(drop=True)
    return df, ggs


async def show_slot_source_settings(clearing_house):
    best_default_source = 'https://dlob-data.s3.eu-west-1.amazonaws.com/'
    ss1, ss2, ss3 = st.columns([3,1,1])
    source = ss1.text_input('source:', best_default_source)
    ss = json.loads((await clearing_house.program.provider.connection.get_slot()).to_json())['result']
    tt = json.loads((await clearing_house.program.provider.connection.get_block_time(ss)).to_json())['result']
    SLOT1 = ss2.number_input('slot ref:', min_value=0, value=ss)
    ss2.write('https://explorer.solana.com/block/'+str(SLOT1))
    TS1 = ss3.number_input('unix_timestamp ref:', min_value=0, value=tt)

    return source, ss, tt, SLOT1, TS1


async def mm_program_page(clearing_house: DriftClient, env): 
    mol00, mol1, molselect, mol0, mol2 = st.columns([2, 3, 3, 3, 10])

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


    latest_slot = None
    ts_for_latest_slot = None

    market_type = mol00.selectbox('market type', ['perp', 'spot'])
    market_indexes = list(range(1,12))
    if market_type == 'perp':
        market_indexes = list(range(0,25))
    market_index = mol1.selectbox(market_type+' market index', market_indexes)
    now = datetime.datetime.now()
    current_ts = time.mktime(now.timetuple())
    tt = market_type+str(market_index)

    active_market_symbol = ''
    if market_type == 'perp':
        active_market_symbol = mainnet_perp_market_configs[market_index].symbol
    else:
        active_market_symbol = mainnet_spot_market_configs[market_index].symbol
    st.write(active_market_symbol)

    st.write('subject to change liquidity score component of maker scores. these numbers are relative standings per market. final score are calculated over larger interval and aggregated across markets.')
    tabs = st.tabs(['mm leaderboard', 'mm individual', 'individual snapshot', 'settings'])

    with tabs[3]:
        source, ss, tt, SLOT1, TS1 = await show_slot_source_settings(clearing_house)


    url_agg = source+'mainnet-beta/aggregate-liq-score/'+market_type+'-'+str(market_index)+'.csv.gz'
    total_agg_df = pd.read_csv(url_agg).dropna()
    tots = total_agg_df.groupby('slot')['score'].sum()
    for x in total_agg_df.slot.unique():
        total_agg_df.loc[(total_agg_df.slot==x), 'score'] /= tots.loc[x]/100
    total_agg_df['approx_date'] = [pd.to_datetime(slot_to_timestamp_est(x, ts_for_latest_slot, latest_slot)*1e9, utc=True) 
                               for x in total_agg_df.slot]
    
   
    # st.write(ffs[-1])
    # all_slots = df_full.slot.unique()
    oldest_slot = total_agg_df.slot.min()
    newest_slot = total_agg_df.slot.max()

    # load_volumes([now], 'SOL-PERP')
    # latest_slot = s2.number_input('latest slot:', value=newest_slot)
    # ts_for_latest_slot = s3.number_input('latest slot ts:')
    # s3.write('current ts='+ str(int(current_ts)))
    # if ts_for_latest_slot == 0:
        # ts_for_latest_slot = None

    week_ago_slot = total_agg_df[total_agg_df.slot>newest_slot-(2*60*60*24*7)].slot.min()
    # print(base_trade_size)

    tzInfo = pytz.timezone('UTC')
    latest_slot_full = int(total_agg_df.slot.max())
    # st.write(total_agg_df)

    lastest_date = pd.to_datetime(slot_to_timestamp_est(latest_slot_full, ts_for_latest_slot, latest_slot)*1e9, utc=True)
    date = lastest_date
    date_range = None
    range_selected = molselect.selectbox('range select:', ['daily', 'range', 'weekly', 'last month'], 0)
    if range_selected == 'daily':
        date = mol0.date_input('select approx. date:', lastest_date, min_value=datetime.datetime(2022,11,4), max_value=lastest_date) #(datetime.datetime.now(tzInfo)))
        values = get_slots_for_date(date)
    elif range_selected == 'range':
        start_slot = mol2.number_input('start slot:', oldest_slot, newest_slot, oldest_slot)
        end_slot = mol2.number_input('end slot:', oldest_slot, newest_slot, newest_slot)
        values = [start_slot, end_slot]
    elif range_selected == 'weekly':
        values = mol2.slider(
        'Select a range of slot values',
        int(week_ago_slot), int(newest_slot), (int(week_ago_slot), int(newest_slot)))
    elif range_selected == 'last month':
        values = mol2.slider(
        'Select a range of slot values',
        int(oldest_slot), int(newest_slot), (int(oldest_slot), int(newest_slot)))

    date_range = pd.to_datetime([slot_to_timestamp_est(x, ts_for_latest_slot, latest_slot)*1e9 for x in values])
    if date_range is not None:
        mol2.write(f'approx date range: `{date_range[0].strftime("%Y-%m-%d %H:%M:%S")}` to `{date_range[1].strftime("%Y-%m-%d %H:%M:%S")}`')

    total_agg_df = total_agg_df[(total_agg_df.slot >= values[0]) & (total_agg_df.slot <= values[1])]

    smallest_slot = total_agg_df.slot.min()
    largest_slot = total_agg_df.slot.max()
    approx_start_date = total_agg_df.approx_date.min()
    approx_end_date = total_agg_df.approx_date.max()

    st.markdown(f"above dataframe start slot: [{(smallest_slot)}](https://explorer.solana.com/block/{(smallest_slot)}) (approx: {approx_start_date})")
    st.markdown(f"above dataframe end slot:   [{(largest_slot)}](https://explorer.solana.com/block/{(latest_slot)}) (approx: {approx_end_date})")

    date_strr = date.strftime("%Y-%m-%d")
    df_full, ffs = get_data_by_market_index(market_type, market_index, env, date_strr,
                                       source if 'amazonaws' in source else '', url_agg)

    df = df_full[(df_full.currentSlot>=values[0]) & (df_full.currentSlot<=values[1])]
    st.write(df_full.currentSlot.max())
    if df_full.currentSlot.min() > values[0]:
        missing_slots = df_full.currentSlot.min() - values[0]
        st.write('missing slots end start:', missing_slots, 'approx', )

    if df_full.currentSlot.max() < values[1]:
        st.write('missing slots from desired end:',  values[1] - df_full.currentSlot.max())

    df['price'] = df['price'].astype(float)
    # st.write('not full', df)

    bbo = df.groupby(['direction', 'currentSlot']).agg(
        baseAssetAmount=("baseAssetAmount", "sum"),
        max_price=("price", 'max'), 
        min_price=("price", 'min')
    ).unstack(0)
    # print(bbo)
    bbo = bbo.swaplevel(axis=1)
    # st.write(bbo)
    all_users = df.groupby('user')['score'].sum().sort_values(ascending=False).index
    top10users = all_users[:10].tolist()

    with tabs[0]:
        total_agg_df['slot'] = total_agg_df['slot'].astype(int)
        piv_agg = total_agg_df.pivot_table(index='slot', columns='user', values='score')
        # st.plotly_chart(piv_agg.cumsum().plot())
        df = total_agg_df
        # st.write(piv_agg)
        all_users2 = df.groupby('user')['score'].sum().sort_values(ascending=False).index
        top10users2 = all_users2[:10].tolist()

        all_stats = []
        score_emas = {}
        # top10users = df.groupby('user')['score'].sum().sort_values(ascending=False).head(10).index
        # st.title('mm leaderboard')
        metmet1, metemet2 = st.columns(2)
        users = st.multiselect('users:', list(all_users2), list(top10users2))
        [lbtable] = st.columns([1])
        [eslid, s2] = st.columns([1, 1])
        [echart] = st.columns([1])

        rolling_window = eslid.slider('rolling window:', 1, 200, 100, 5)
        use_ts_est = s2.selectbox('index:', ['slot', 'est_utc_timestamp'], key='tab1', help='est utc uses "slot ref:" and "unix_timestamp_ref" to deduce timestamps')
        # for user in users:
            # bbo_user, bbo_user_stats = get_mm_stats(df, user, oracle, bbo2snippet)
            # score_emas[str(user)] = (bbo_user['score'].fillna(0)
            # all_stats.append(bbo_user_stats)
        # all_stats = df
        all_stats_df = df.sort_values('score', ascending=False)
        metmet1.metric('total avg score:', np.round(piv_agg.sum(axis=1).mean(),2), help='this is relative standing normalized across slot')
        # lbtable.dataframe(all_stats_df)
        # print(topmm)
        dat = piv_agg[users].fillna(0)
        if use_ts_est == 'est_utc_timestamp':
            dat.index = [pd.to_datetime(slot_to_timestamp_est(x, ts_for_latest_slot, latest_slot)*1e9, utc=True) for x in dat.index]
        echart.plotly_chart(dat.rolling(rolling_window,
                                        min_periods=1).mean().plot(), use_container_width=True)
        # st.header('total aggregate dataframe')
        # st.write(total_agg_df)

    with tabs[1]:
        o0, o1, o2, o3 = st.columns(4)
        o0.write('look up your user account address using your authority + subaccount number')
        authority_l = o1.text_input('wallet address')
        sub_id_l = o2.number_input('subaccount id')
        if len(authority_l) > 10:
            o3.write(get_user_account_public_key(clearing_house.program_id,
                        Pubkey.from_string(authority_l),
                        int(sub_id_l)))
        
        s1, s2 = st.columns(2)
        user = s1.selectbox('individual maker', all_users, 0, help='maker address is drift user account address (not wallet address). You can use the above to find your account address.')
        use_ts_est = s2.selectbox('index:', ['slot', 'est_utc_timestamp'], help='est utc uses "slot ref:" and "unix_timestamp_ref" to deduce timestamps')
        
        indiv_maker_df = total_agg_df[total_agg_df.user==user]
        df_to_plt = indiv_maker_df.set_index('slot')['score']
        if use_ts_est == 'est_utc_timestamp':
            df_to_plt.index = [pd.to_datetime(slot_to_timestamp_est(x, ts_for_latest_slot, latest_slot)*1e9, utc=True) 
                               for x in df_to_plt.index]

        st.plotly_chart(df_to_plt.plot())
        # bbo_user, bbo_user_stats = get_mm_stats(df, user, oracle, bbo2)

        # st.text('user bestbid time='+str(np.round(offer_best_pct*100, 2))+'%')
        # st.text('user bid within 3bps of best time='+str(np.round(offer_within_best_pct*100, 2))+'%')
        # st.text('user bestoffer time='+str(np.round(offer_best_pct*100, 2))+'%')
        # st.text('user offer within 3bps of best time='+str(np.round(offer_within_best_pct*100, 2))+'%')
        # st.text('user uptime='+str(np.round(uptime_pct*100, 2))+'%')
        # bbo_user['score'] = bbo_user['score'].fillna(0)
        # bbo_user['ema_score'] = bbo_user['score'].fillna(0).ewm(100).mean()
        # dat = bbo_user.loc[values[0]:values[1]]
               # st.plotly_chart(dat.plot(title=market_type+' market index='+str(market_index)))

        # best_slot = bbo_user.score.idxmax()
        # worst_slot = bbo_user.score.idxmin()
        # st.write('best slot:'+str(best_slot), ' | worst slot:'+str(worst_slot))

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

    with tabs[2]:
        st.title('individual snapshot lookup')
        # print(df.columns)
        # st.write(ffs)
        sol1, sol10, sol11, sol0, sol2 = st.columns([2,1,1, 1,2])
        slot = sol1.select_slider('individual snapshot', df.slot.unique().tolist(), df.slot.max())
        slot = sol10.number_input('slot:', value=slot)
        score_color = sol11.radio('show score highlights:', [True, False], 0)
        dol1, dol2 = st.columns([2,2])
        snapshot_type = dol1.radio('snapshot_type:', ['raw', 'scored'], index=1, horizontal=True)
        snapshot_display = dol2.radio('snapshot_display:', ['raw', 'filtered'], index=1, horizontal=True)

        outf = [ff for ff in ffs if str(slot) in ff]

        toshow_snap = pd.DataFrame()
        df = pd.DataFrame()
        if snapshot_type == 'raw':
            st.write('loading:', source+str(outf))
            outf = [ff for ff in ffs if str(slot) in ff]
            if len(outf):
                dfr = load_snapshot_file(outf[0], source)
                df = dfr[(dfr.marketIndex==market_index)
                            & (dfr.marketType==market_type)
                            & (dfr.orderType=='limit')
                            & (dfr.price != 0)
                            ]
                st.write(df)
                for col in ['price', 'oraclePrice', 'oraclePriceOffset']:
                    df[col] = df[col].astype(int)
                    df[col] /= 1e6
                    
                for col in ['quoteAssetAmountFilled']:
                    df[col] = df[col].astype(int)
                    df[col] /= 1e6 

                for col in ['baseAssetAmount', 'baseAssetAmountFilled']:
                    df[col] = df[col].astype(int)
                    df[col] /= 1e9
                
                df['score'] = np.nan
                toshow = df
                toshow_snap = toshow
            else:
                st.warning(ffs)
        else:
            path = 'mainnet-beta/'+date_strr+'/liq-score-'+market_type+'-'+str(market_index)\
                +'-' + str(slot) \
                + '.csv.gz'
            st.write('loading:', source+ path)
            df = load_scored_snapshot_file(path, source)
            if df.empty:
                st.error(f'{source + path} does not exist, no individual snapshots')
            else:
                toshow = df[['level', 'score', 'price', 'priceRounded', 'baseAssetAmountLeft', 'direction', 'user', 'status', 'orderType',
            'marketType', 'baseAssetAmount', 'marketIndex',  'oraclePrice', 'slot', 'orderId', 'userOrderId',
            'baseAssetAmountFilled', 'quoteAssetAmountFilled', 'reduceOnly',
            'triggerPrice', 'triggerCondition', 'existingPositionDirection',
            'postOnly', 'immediateOrCancel', 'oraclePriceOffset', 'auctionDuration',
            'auctionStartPrice', 'auctionEndPrice', 'maxTs',
            ]]
                toshow_snap = toshow[toshow.orderType=='limit']

        if not toshow_snap.empty:
            slippage = sol2.select_slider('slippage (%)', list(range(1, 100)), 5)

            # toshow_snap = toshow[df.slot.astype(int)==int(slot)]
            # st.write(toshow_snap)
            if 'score' in df.columns:
                st.metric('total score:', np.round(df.score.sum(),2))

            # st.write(toshow_snap)
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

            if snapshot_display == 'raw':
                st.write(df)
            else:
                if score_color:
                    toshow_snap = toshow_snap.replace(0, np.nan).dropna(subset=['score'],axis=0)
                st.dataframe(toshow_snap.style.applymap(cooling_highlight, subset=['direction'])
                        .applymap(level_highlight, subset=['level'])
                            #  ,(heating_highlight, subset=['Heating inputs', 'Heating outputs'])
                            , use_container_width=True)