import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"
from solana.rpc.types import MemcmpOpts

import time
# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
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

import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import *
from driftpy.addresses import * 
from driftpy.constants.numeric_constants import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import asyncio

async def slot_ts_fetch(_ch):
    ss = json.loads((await _ch.program.provider.connection.get_slot()).to_json())['result']
    tt = json.loads((await _ch.program.provider.connection.get_block_time(ss)).to_json())['result']
    st.write(ss, tt)
    return (ss, tt)

@st.cache_data
def slot_ts_fetch1(_ch):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(slot_ts_fetch(_ch))        

@st.cache_data
def cached_get_orders_data(_ch: DriftClient, depth_slide, market_type, market_index, order_filter):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(get_orders_data(_ch, depth_slide, market_type, market_index, order_filter))

@st.cache_data
def cached_get_price_data(market_type, market_index):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(get_price_data(market_type, market_index))


async def get_price_data(market_type, market_index):
    assert(market_type in ['spot', 'perp'])
    url = f'https://mainnet-beta.api.drift.trade/trades?marketIndex={market_index}&marketType={market_type}'
    dat = [json.loads(x) for x in requests.get(url).json()['data']['trades']]
    return pd.DataFrame(dat)

async def get_orders_data(_ch: DriftClient, depth_slide, market_type, market_index, order_filter):
    # try:
    all_users = await _ch.program.account['User'].all(filters=[MemcmpOpts(offset=4352, bytes='2')])
        # st.warning('len: '+str(len(all_users)))
    # except Exception as e:
    #     st.warning("ERROR: '"+str(e)+"', and cannot load ['User'].all() with current rpc")
    #     return

    st.sidebar.text('cached on: ' + _ch.time)
    
    # long orders 
    # short orders 
    # current oracle price 
    # [bids/longs] [asks/shorts]
    # [price, baa], [price, baa]
    # (dec price)    (inc price)
    from driftpy.accounts.oracle import get_oracle_price_data_and_slot

    state = await get_state_account(_ch.program)
    mm = state.number_of_markets if market_type == 'perp' else state.number_of_spot_markets
    for perp_idx in range(mm):
        if perp_idx != market_index:
            continue
        market = None
        oracle_pubkey = None
        oracle_source = None
        if market_type == 'perp':
            market = await get_perp_market_account(
                _ch.program, perp_idx
            )
            oracle_pubkey = market.amm.oracle
            oracle_source = market.amm.oracle_source
        else:
            market = await get_spot_market_account(
                _ch.program, perp_idx
            )
            oracle_pubkey = market.oracle
            oracle_source = market.oracle_source

        oracle_data = (await get_oracle_price_data_and_slot(_ch.program.provider.connection, oracle_pubkey, oracle_source))
        oracle_price = oracle_data.data.price
        # oracle_price = 14 * 1e6

        order_type_dict = {'PositionDirection.Long()': [], 'PositionDirection.Short()': []}
        for x in all_users: 
            user: User = x.account
            orders = user.orders
            for order in orders:
                # st.write(order.market_type, market_type)
                order_market_type = str(order.market_type).split('.')[-1].replace('()','').lower()
                if str(order.status) == 'OrderStatus.Open()' and order.market_index == perp_idx and  order_market_type == market_type:
                    order.owner = str(x.public_key)
                    order.authority = str(x.account.authority)                        

                    # if order.trigger_price != 0 and order.price == 0: 
                    #     order.price = order.trigger_price
                    
                    if order.oracle_price_offset != 0 and order.price == 0: 
                        order.price = oracle_price + order.oracle_price_offset

                    # oracle offset orders for now 
                    if order.price != 0:
                        order_type = str(order.direction)
                        order_type_dict[order_type] = order_type_dict.get(order_type, []) + [order]

                        # print(order_type, order.price/PRICE_PRECISION, user.authority)

        longs = order_type_dict['PositionDirection.Long()']
        longs.sort(key=lambda order: order.price)
        if order_filter != 'All':
            if order_filter == 'Limit':
                longs = [x for x in longs if 'Trigger' not in str(x.order_type)]
            else:
                longs = [x for x in longs if order_filter in str(x.order_type)]
        longs = longs[::-1] # decreasing price 

        shorts = order_type_dict['PositionDirection.Short()']
        shorts.sort(key=lambda order: order.price) # increasing price
        if order_filter != 'All':
            if order_filter == 'Limit':
                shorts = [x for x in shorts if 'Trigger' not in str(x.order_type)]
            else:
                shorts = [x for x in shorts if order_filter in str(x.order_type)]

        def format_order(order: Order):
            price = order.price/PRICE_PRECISION
            size = (order.base_asset_amount - order.base_asset_amount_filled)/AMM_RESERVE_PRECISION
            return (price, size)

        d_longs_authority = [str(order.authority) for order in longs]
        d_longs_order_id = [order.order_id for order in longs]
        d_longs_owner = [str(order.owner) for order in longs]
        d_longs_order_type = [str(order.order_type).split('.')[-1].split('()')[0] for order in longs]
        d_longs_order_type = [x+' [POST]' if longs[idx].post_only else x for idx, x in enumerate(d_longs_order_type)]
        d_longs_order_type = ['Oracle'+x if longs[idx].oracle_price_offset != 0 else x for idx, x in enumerate(d_longs_order_type)]
        d_longs = [format_order(order) for order in longs]
        d_shorts = [format_order(order) for order in shorts]

        d_shorts_order_type = [str(order.order_type).split('.')[-1].split('()')[0] for order in shorts]
        d_shorts_order_type = [x+' [POST]' if shorts[idx].post_only else x for idx, x in enumerate(d_shorts_order_type)]
        d_shorts_order_type = ['Oracle'+x if shorts[idx].oracle_price_offset != 0 else x for idx, x in enumerate(d_shorts_order_type)]
        d_shorts_owner = [str(order.owner) for order in shorts]
        d_shorts_authority = [str(order.authority) for order in shorts]
        d_shorts_order_id = [order.order_id for order in shorts]

        # st.write(f'number of bids: {len(d_longs)}')
        # st.write(f'number of asks: {len(d_shorts)}')

        # col1, col2, col3 = st.columns(3)
        # col1.metric("best bid", d_longs[0][0], str(len(d_longs))+" orders total")
        # col2.metric("best ask",  d_shorts[0][0], "-"+str(len(d_shorts))+" orders total")

        pad = abs(len(d_longs) - len(d_shorts))
        if len(d_longs) > len(d_shorts):
            d_shorts += [""] * pad
            d_shorts_owner += [""] * pad
            d_shorts_order_type += [""] * pad
            d_shorts_authority += [""] * pad
            d_shorts_order_id += [""] * pad
        else:
            d_longs += [""] * pad
            d_longs_owner  += [""] * pad
            d_longs_order_type += [""] * pad
            d_longs_authority += [""] * pad
            d_longs_order_id += [""] * pad

        market_name = bytes(market.name).decode('utf-8')

        order_data = {
            'market': market_name,
            'bids order id': d_longs_order_id,
            'bids authority': d_longs_authority,
            'bids owner': d_longs_owner,
            'bids order type': d_longs_order_type,
            'bids (price, size)': d_longs,
            'asks (price, size)': d_shorts,
            'asks order type': d_shorts_order_type,
            'asks owner': d_shorts_owner,
            'asks authority': d_shorts_authority,
            'asks order id': d_shorts_order_id,
        }


        price_min = float(oracle_price/1e6)*float(1-depth_slide*.001)
        price_max = float(oracle_price/1e6)*float(1+depth_slide*.001)
        drift_depth = pd.DataFrame(columns=['bids', 'asks'])
        drift_order_depth = pd.DataFrame(columns=['bids', 'asks'])
        if market_type == 'perp':
            drift_order_depth, drift_depth = calc_drift_depth(oracle_price/1e6, market.amm.long_spread/1e6, 
            
            market.amm.short_spread/1e6,
            market.amm.base_asset_reserve/1e9, price_max, order_data)
        else:
            dd = requests.get('https://openserum.io/api/serum/market/8BnEgHoWFysVcuFFX7QztDmzuH8r5ZFvyP3sYwn1XTh6/depth').json()
            lenn = 13
            bb = pd.DataFrame(dd['bids']).set_index(0)[[2]].iloc[-lenn:].sort_index(ascending=False).cumsum()#.reset_index()
            aa = pd.DataFrame(dd['asks']).set_index(0)[[2]].iloc[:lenn].sort_index(ascending=True).cumsum()#.reset_index()
            bb.columns = ['bids']
            aa.columns = ['asks']
            # df = pd.DataFrame(index=range(10))
            # df['bids'] = [tuple(x) for x in bb.values]
            # df['asks'] = [tuple(x) for x in aa.values]
            drift_depth = pd.concat([bb, aa])

            drift_order_bids = pd.DataFrame(order_data['bids (price, size)'], columns=['price', 'bids'])\
            .set_index('price').dropna()
            drift_order_bids['bids'] = drift_order_bids['bids'].astype(float).cumsum()
            drift_order_bids.rename_axis(None, inplace=True)

            drift_order_asks = pd.DataFrame(columns=['asks'])
            if order_data['asks (price, size)'] != ['']:
                drift_order_asks = pd.DataFrame(order_data['asks (price, size)'], columns=['price', 'asks'])\
                    .set_index('price').dropna()
                drift_order_asks['asks'] = drift_order_asks['asks'].astype(float).cumsum()
                drift_order_asks.rename_axis(None, inplace=True)
                drift_order_asks = drift_order_asks.loc[:price_max]
                

            drift_order_depth = pd.concat([drift_order_bids, drift_order_asks]).replace(0, np.nan).sort_index()
            _idxs = []
            for v in drift_order_depth.index: 
                # add a small value
                while v in _idxs: 
                    v += 1e-5
                _idxs.append(v)
            drift_order_depth.index = pd.Index(_idxs)

            ss = [drift_depth.index.min()]+[x for x in drift_order_depth.index.to_list() if x > drift_depth.index.min()]+[drift_depth.index.max()]
            drift_order_depth = drift_order_depth.reindex(ss, method='ffill').sort_index()
            drift_order_depth['bids'] = drift_order_depth['bids'].bfill()

        return (pd.DataFrame(order_data), oracle_data, drift_order_depth, drift_depth)


def calc_drift_depth(mark_price, l_spr, s_spr, base_asset_reserve, price_max, order_data):
        def calc_slip(x):
            f = x/base_asset_reserve
            slippage = 1/(1-f)**2 - 1
            return slippage
        def calc_slip_short(x):
            f = x/base_asset_reserve
            slippage = 1 - 1/(1+f)**2
            return slippage

        max_f = np.sqrt(price_max)/np.sqrt(mark_price) - 1

        # st.table(pd.Series([max_f, mark_price, price_max, base_asset_reserve]))
        quantities_max = max(1, int(max_f*base_asset_reserve))
        quantities = list(range(1, quantities_max, int(max(1, quantities_max/100))))
        if quantities_max <= 10:
            quantities = [x/100 for x in quantities]
        drift_asks = pd.DataFrame(quantities, 
        columns=['asks'],
        index=[mark_price*(1+l_spr)*(1+calc_slip(x)) for x in quantities])
        drift_bids = pd.DataFrame(quantities, 
        columns=['bids'],
        index=[mark_price*(1-s_spr)*(1-calc_slip_short(x)) for x in quantities])

        # print(order_data['bids (price, size)'])
        drift_order_bids = pd.DataFrame(order_data['bids (price, size)'], columns=['price', 'bids'])\
            .set_index('price').dropna()
        drift_order_bids['bids'] = drift_order_bids['bids'].astype(float).cumsum()
        drift_order_bids.rename_axis(None, inplace=True)
            
        drift_order_asks = pd.DataFrame(columns=['asks'])
        if order_data['asks (price, size)'] != []:
            # print(order_data)
            drift_order_asks = pd.DataFrame(order_data['asks (price, size)'], columns=['price', 'asks'])\
                .set_index('price').dropna()
            drift_order_asks['asks'] = drift_order_asks['asks'].astype(float).cumsum()
            drift_order_asks.rename_axis(None, inplace=True)
            drift_order_asks = drift_order_asks.loc[:price_max]

        drift_depth = pd.concat([drift_bids, drift_asks]).replace(0, np.nan).sort_index()

        drift_order_depth = pd.concat([drift_order_bids, drift_order_asks]).replace(0, np.nan).sort_index()
        _idxs = []
        for v in drift_order_depth.index: 
            # add a small value
            while v in _idxs: 
                v += 1e-5
            _idxs.append(v)
        drift_order_depth.index = pd.Index(_idxs)

        ss = [drift_depth.index.min()]+[x for x in drift_order_depth.index.to_list() if x > drift_depth.index.min()]+[drift_depth.index.max()]
        drift_order_depth = drift_order_depth.reindex(ss, method='ffill').sort_index()


        return drift_order_depth, drift_depth


def orders_rpc_page(ch: DriftClient):

        # time.sleep(3)
        # oracle_price = 13.5 * 1e6 

        depth_slide = st.slider('depth:', 1, 1000, 10)

        s1, s2, s3 = st.columns(3)
        market_type = s1.selectbox('marketType:', list(['perp', 'spot']))
        market_index = s2.selectbox('marketIndex:', list(range(21)))
        # market = col1.radio('select market:', ['SOL-PERP', 'BTC-PERP', 'ETH-PERP', 'SOL-USDC'], horizontal=True)
        order_filter = s3.radio('order filter:', ['Limit', 'Trigger', 'All'], horizontal=True)

        # if market == 'SOL-PERP':
        #     market_type = 'perp'
        #     market_index= 0
        # elif market == 'BTC-PERP':
        #     market_type = 'perp'
        #     market_index= 1        
        # elif market == 'ETH-PERP':
        #     market_type = 'perp'
        #     market_index= 2
        # else:
        #     market_type = 'spot'
        #     market_index = 1

        data, oracle_data, drift_order_depth, drift_depth  = cached_get_orders_data(ch, depth_slide, market_type, market_index, order_filter)
        # if len(data):
        #     st.write(f'{data.market.values[0]}')

        zol2, zol3 = st.columns([6,20])

        # zol1.image("https://app.drift.trade/assets/icons/markets/sol.svg", width=33)
        # print(oracle_data)
        # oracle_data.slot
        zol2.metric('Oracle Price', f'${oracle_data.data.price/PRICE_PRECISION}', f'±{oracle_data.data.confidence/PRICE_PRECISION} (slot={oracle_data.slot})',
        delta_color="off")
        tabs = st.tabs(['OrderBook', 'Depth', 'Recent Trades'])

        with tabs[0]:


            correct_order = data.columns.tolist()
            cols = st.multiselect(
                            "Choose columns", data.columns.tolist(), 
                            ['bids order id', 'bids order type', 'bids (price, size)', 'asks (price, size)', 'asks order type',  'asks order id']
                        )
            subset_ordered = [x for x in correct_order if x in cols]
            df = pd.DataFrame(data)[subset_ordered]

            market_nom = data['market'].unique()[0]

            def make_clickable(link):
                # target _blank to open new window
                # extract clickable text to display for your link
                text = link.split('=')[1]
                text = text[:4]+'..'+text[-4:]
                return f'<a target="_blank" href="{link}">{text}</a>'

            # link is the column with hyperlinks
            # df['link'] = df['bids authority'].apply(lambda x: f'https://app.drift.trade/?authority={x}')
            # df['link'] = df['link'].apply(make_clickable)

            def highlight_survived(s):
                res = []
                for _ in range(len(s)):
                    if 'bids (price' in s.name:
                        res.append('background-color: lightgreen')
                    elif 'asks (price' in s.name:
                        res.append('background-color: pink')
                    else:
                        res.append('')
                return res

            
            bids_quote = np.round(df['bids (price, size)'].apply(lambda x: x[0]*x[1] if x!='' else 0).sum(), 2)
            bids_base = np.round(df['bids (price, size)'].apply(lambda x: x[1] if x!='' else 0).sum(), 2)

            asks_quote = np.round(df['asks (price, size)'].apply(lambda x: x[0]*x[1] if x!='' else 0).sum(), 2)
            asks_base = np.round(df['asks (price, size)'].apply(lambda x: x[1] if x!='' else 0).sum(), 2)
            col1, col2, _ = st.columns([5,5, 10])
            col1.metric(f'bids:', f'${bids_quote:,.2f}', f'{bids_base} {market_nom}')
            col2.metric(f'asks:', f'${asks_quote:,.2f}', f'{-asks_base} {market_nom}')
            if len(df):
                st.dataframe(df.astype(str).style.apply(highlight_survived, axis=0))
            else:
                st.dataframe(df)

        with tabs[1]: 
            # depth_slide = st.slider("Depth", 1, int(1/.01), 10, step=5)

            ext_depth_nom = 'Drift vAMM' if market_type == 'perp' else 'Openbook'
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                subplot_titles=[ext_depth_nom+' depth', 'Drift DLOB depth'])

            fig.add_trace( go.Scatter(x=drift_depth.index, y=drift_depth['bids'],  name='bids', fill='tozeroy'),  row=1, col=1)
            fig.add_trace( go.Scatter(x=drift_depth.index, y=drift_depth['asks'],  name='asks', fill='tozeroy'),  row=1, col=1)
            fig.add_vline(x=oracle_data.data.price/PRICE_PRECISION, line_width=3, line_dash="dot", line_color="blue")

            fig.add_trace( go.Scatter(x=drift_order_depth.index, y=drift_order_depth['bids'], name='bids',  fill='tozeroy'),  row=2, col=1)
            fig.add_trace( go.Scatter(x=drift_order_depth.index, y=drift_order_depth['asks'], name='asks',  fill='tozeroy'), row=2, col=1)

            st.plotly_chart(fig)


        with tabs[2]:
            price_df = cached_get_price_data(market_type, market_index)
            if len(price_df):
                # st.write(price_df)
                odf = (price_df.set_index('ts'))
                odf['tradePrice'] = (odf['quoteAssetAmountFilled'].astype(float))/ odf['baseAssetAmountFilled'].astype(float) *1e3
                odf['oraclePrice'] = odf['oraclePrice'].astype(float)/1e6
                odf['baseAssetAmountFilled'] = odf['baseAssetAmountFilled'].astype(float)/1e9
                can_cols = ['oraclePrice', 'tradePrice', 'baseAssetAmountFilled', 'taker', 'maker', 'actionExplanation', 'takerOrderDirection']
                can_cols = [x for x in can_cols if x in odf.columns]
                odf = odf[can_cols]
                odf.index = pd.to_datetime([int(x) for x in odf.index.astype(int) * 1e9])
                layout = go.Layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    grid=None,
                xaxis_showgrid=False, yaxis_showgrid=False
                )

                fig = odf[['tradePrice', 'oraclePrice']].plot()
                fig = fig.update_layout(layout)

                col1, col3 = st.columns(2, gap='large')
                col1.plotly_chart(fig, use_container_width=True)

                can_cols2 = [x for x in ['tradePrice', 'baseAssetAmountFilled', 'taker', 'maker', 'actionExplanation', 'takerOrderDirection'] if x in can_cols]
                tbl = odf.reset_index(drop=True)[can_cols2].fillna('vAMM')

                renom_cols = ['Price', 'Size', 'Taker', 'Maker', 'ActionExplanation', 'takerOrderDirection']
                if len(can_cols2) == len(renom_cols):
                    tbl.columns = renom_cols
                else:
                    tbl.columns = ['Price', 'Size', 'Taker', 'ActionExplanation', 'takerOrderDirection']


                def highlight_survived(s):
                    return ['background-color: lightgreen']*len(s) if s.takerOrderDirection=='long' else ['background-color: pink']*len(s)

                def color_survived(val):
                    color = 'green' if val else 'red'
                    return f'background-color: {color}'

                col3.dataframe(tbl.style.apply(highlight_survived, axis=1), use_container_width=True)




def orders_page(ch: DriftClient):
        slot_ts_fetch1(ch)

        data_source = st.radio(horizontal=True, label='Data Source:', options=[None, 
                                                                               'SERVER-BATCHL2', 
                                                                               'SERVER-L3',
                                                                               'RPC'])

        if data_source == 'SERVER-BATCHL2':
            url = 'https://mainnet-beta.api.drift.trade/dlob/batchL2'
            fields = '?marketType=perp,perp,perp,perp,perp,perp,perp,perp,perp,perp,perp,perp,perp,perp,perp,perp,perp,perp,perp,spot,spot,spot,spot,spot,spot,spot,spot,spot&marketIndex=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,0,1,2,3,4,5,6,7,8&depth=100,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20&includeVamm=true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,false,false,false,false,false,false,false,false,false&includePhoenix=false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,true,true,true,true,true&includeSerum=false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,true,true,true,true,true,true,true,true,true&includeOracle=true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,true'
            endpoint = st.text_input('endpoint:', value=url+fields)
            result = requests.get(endpoint).json()

            perp_df = pd.DataFrame(result['l2s'][:19])
            perp_df['bids'] = perp_df['bids'].apply(lambda x: float(x[0]['price'])/PRICE_PRECISION)
            perp_df['asks'] = perp_df['asks'].apply(lambda x: float(x[0]['price'])/PRICE_PRECISION)
            perp_df['oracle'] = perp_df['oracle'].astype(float) /PRICE_PRECISION
            perp_df['spread'] = ((perp_df['asks'] - perp_df['bids'])/((perp_df['bids']+perp_df['asks'])/2)) * 100
            perp_df = perp_df[['bids', 'asks', 'oracle', 'spread', 'slot']]

            st.dataframe(perp_df)
            st.json(result, expanded=False)


        if data_source == 'SERVER-L3':
            url = 'https://mainnet-beta.api.drift.trade/dlob/l3'
            s1, s2, s3 = st.columns(3)
            mt = s1.selectbox('marketType:', list(['perp', 'spot']))
            mi = s2.selectbox('marketIndex:', list(range(21)))
            fields = f'?marketIndex={mi}&marketType={mt}&depth=1'
            endpoint = s3.text_input('endpoint:', value=url+fields)
            result = requests.get(endpoint).json()

            import plotly.graph_objects as go

            # Extracting bid and ask data
            bids = result["bids"]
            asks = result["asks"]
            st.write('server slot:', result['slot'])
            # Convert price and size to floats and scale
            for bid in bids:
                bid["price"] = float(bid["price"]) / 1e6
                bid["size"] = float(bid["size"]) / 1e9

            for ask in asks:
                ask["price"] = float(ask["price"]) / 1e6
                ask["size"] = float(ask["size"]) / 1e9
            # Sort bids and asks by price
            bids.sort(key=lambda x: x["price"], reverse=True)
            asks.sort(key=lambda x: x["price"])

            # Calculate cumulative bid and ask depths
            cumulative_bid_depth = [sum(bid['size'] for bid in bids[:i + 1]) for i in range(len(bids))]
            cumulative_ask_depth = [sum(ask['size'] for ask in asks[:i + 1]) for i in range(len(asks))]

            # Extracting price, size, and maker information for hover data
            bid_hover_text = [f"Price: {bid['price']}, Cumulative Size: {cumulative_bid_depth[i]}, Maker: {bid['maker']}" for i, bid in enumerate(bids)]
            ask_hover_text = [f"Price: {ask['price']}, Cumulative Size: {cumulative_ask_depth[i]}, Maker: {ask['maker']}" for i, ask in enumerate(asks)]

            # Creating the cumulative area plot chart with markers
            fig = go.Figure()

            # Plotting cumulative bid depth with markers
            fig.add_trace(go.Scatter(x=[bid['price'] for bid in bids],
                                    y=cumulative_bid_depth,
                                    mode='lines+markers',
                                    marker=dict(color='green'),
                                    line=dict(color='green'),
                                    hovertext=bid_hover_text,
                                    name='Bid Depth'))

            # Plotting cumulative ask depth with markers
            fig.add_trace(go.Scatter(x=[ask['price'] for ask in asks],
                                    y=cumulative_ask_depth,
                                    mode='lines+markers',
                                    marker=dict(color='red'),
                                    line=dict(color='red'),
                                    hovertext=ask_hover_text,
                                    name='Ask Depth'))

            # Setting layout
            fig.update_layout(title='Market Liquidity Depth',
                            xaxis=dict(title='Price'),
                            yaxis=dict(title='Cumulative Depth'),
                            showlegend=True)
            
            mid_price = (bids[0]['price'] + asks[0]['price']) / 2

            so1, so2 = st.columns(2)
            zoom_percentage = so1.selectbox('zoom:', [1,5,10,None])
            # so2.multiselect()

            if zoom_percentage is not None:
                fig.update_xaxes(range=[mid_price * (1 - zoom_percentage / 100), mid_price * (1 + zoom_percentage / 100)])

            # Show the plot
            st.plotly_chart(fig)

            st.json(result, expanded=False)


        if data_source == 'RPC':
            orders_rpc_page(ch)