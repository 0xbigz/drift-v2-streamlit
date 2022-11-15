import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"

import time
# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
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

import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import *
from driftpy.addresses import * 
from driftpy.constants.numeric_constants import *

import asyncio

@st.experimental_memo
def cached_get_orders_data(rpc: str, _ch: ClearingHouse):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(get_orders_data(rpc, _ch))

async def get_orders_data(rpc: str, _ch: ClearingHouse):
    all_users = await _ch.program.account['User'].all()
    
    # long orders 
    # short orders 
    # current oracle price 
    # [bids/longs] [asks/shorts]
    # [price, baa], [price, baa]
    # (dec price)    (inc price)
    from driftpy.clearing_house_user import get_oracle_data

    state = await get_state_account(_ch.program)
    for perp_idx in range(state.number_of_markets):
        market = await get_perp_market_account(
            _ch.program, perp_idx
        )
        oracle_data = (await get_oracle_data(_ch.program.provider.connection, market.amm.oracle))
        oracle_price = oracle_data.price
        # oracle_price = 14 * 1e6
        st.write(f"Market: {bytes(market.name).decode('utf-8')}")

        order_type_dict = {}
        for x in all_users: 
            user: User = x.account
            orders = user.orders
            for order in orders:
                if str(order.status) == 'OrderStatus.Open()' and order.market_index == perp_idx:
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

        longs = order_type_dict['PositionDirection.Long()']
        longs.sort(key=lambda order: order.price)
        longs = longs[::-1] # decreasing price 

        shorts = order_type_dict['PositionDirection.Short()']
        shorts.sort(key=lambda order: order.price) # increasing price

        def format_order(order: Order):
            price = float(f'{order.price/PRICE_PRECISION:,.4f}')
            size = (order.base_asset_amount - order.base_asset_amount_filled)/AMM_RESERVE_PRECISION
            return (price, size)

        d_longs_authority = [str(order.authority) for order in longs]
        d_longs_order_id = [order.order_id for order in longs]
        d_longs_owner = [str(order.owner) for order in longs]
        d_longs_order_type = [str(order.order_type).split('.')[-1].split('()')[0] for order in longs]
        d_longs = [format_order(order) for order in longs]
        d_shorts = [format_order(order) for order in shorts]
        d_shorts_order_type = [str(order.order_type).split('.')[-1].split('()')[0] for order in shorts]
        d_shorts_owner = [str(order.owner) for order in shorts]
        d_shorts_authority = [str(order.authority) for order in shorts]
        d_shorts_order_id = [order.order_id for order in shorts]

        st.write(f'number of bids: {len(d_longs)}')
        st.write(f'number of asks: {len(d_shorts)}')

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

        data = {
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
        return (pd.DataFrame(data), oracle_price)

def orders_page(rpc: str, ch: ClearingHouse):

        # time.sleep(3)
        # oracle_price = 13.5 * 1e6 

        data, oracle_price = cached_get_orders_data(rpc, ch)

        st.write(f'last oracle price: {oracle_price/PRICE_PRECISION}')

        correct_order = data.columns.tolist()
        cols = st.multiselect(
                        "Choose columns", data.columns.tolist(), 
                        ['bids order id', 'bids (price, size)', 'asks (price, size)',  'asks order id']
                    )
        subset_ordered = [x for x in correct_order if x in cols]
        df = pd.DataFrame(data)[subset_ordered]
        st.dataframe(df)