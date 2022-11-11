import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"

print(driftpy.__dir__())
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

async def orders_page(rpc: str, ch: ClearingHouse):
    all_users = await ch.program.account['User'].all()
    
    # long orders 
    # short orders 
    # current oracle price 
    # [bids/longs] [asks/shorts]
    # [price, baa], [price, baa]
    # (dec price)    (inc price)
    from driftpy.clearing_house_user import get_oracle_data

    state = await get_state_account(ch.program)
    for perp_idx in range(state.number_of_markets):
        market = await get_perp_market_account(
            ch.program, perp_idx
        )
        oracle_price = (await get_oracle_data(ch.program.provider.connection, market.amm.oracle)).price 

        st.write(f"Market: {bytes(market.name).decode('utf-8')}")

        order_type_dict = {}
        for x in all_users: 
            user: User = x.account
            orders = user.orders
            for order in orders:
                if str(order.status) == 'OrderStatus.Open()' and order.market_index == perp_idx:
                    if order.trigger_price != 0 and order.price == 0: 
                        order.price = order.trigger_price
                    
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

        d_longs = [format_order(order) for order in longs]
        d_shorts = [format_order(order) for order in shorts]

        pad = abs(len(d_longs) - len(d_shorts))
        if len(d_longs) > len(d_shorts):
            d_shorts += [""] * pad
        else:
            d_longs += [""] * pad

        data = {
            'bids (price, size)': d_longs,
            'asks (price, size)': d_shorts,
        }
        st.write(f'number of bids: {len(d_longs)}')
        st.write(f'number of asks: {len(d_shorts)}')
        st.write(f'last oracle price: {oracle_price/PRICE_PRECISION}')

        df = pd.DataFrame(data=data)
        st.write(df)