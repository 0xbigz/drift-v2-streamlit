
from typing import Tuple
from driftpy.constants.numeric_constants import *
from driftpy.types import PerpMarket
PERCENTAGE_PRECISION = 10**6
DEFAULT_REVENUE_SINCE_LAST_FUNDING_SPREAD_RETREAT = 100

def cap_to_max_spread(
    long_spread: int,
    short_spread: int,
    max_spread: int
) -> Tuple[int, int]:
    total_spread = long_spread + short_spread

    if total_spread > max_spread:
        if long_spread > short_spread:
            long_spread = (long_spread * max_spread + total_spread - 1) // total_spread
            short_spread = max_spread - long_spread
        else:
            short_spread = (short_spread * max_spread + total_spread - 1) // total_spread
            long_spread = max_spread - short_spread

    new_total_spread = long_spread + short_spread

    assert new_total_spread <= max_spread, f"new_total_spread({new_total_spread}) > max_spread({max_spread})"

    return long_spread, short_spread

def calculate_spread_revenue_retreat_amount(
    base_spread: int,
    max_spread: int,
    net_revenue_since_last_funding: int
) -> int:
    revenue_retreat_amount = 0

    if net_revenue_since_last_funding < DEFAULT_REVENUE_SINCE_LAST_FUNDING_SPREAD_RETREAT:
        max_retreat = max_spread // 10

        if net_revenue_since_last_funding >= DEFAULT_REVENUE_SINCE_LAST_FUNDING_SPREAD_RETREAT * 1000:
            revenue_retreat_amount = min(
                max_retreat,
                (base_spread * abs(net_revenue_since_last_funding)) // abs(DEFAULT_REVENUE_SINCE_LAST_FUNDING_SPREAD_RETREAT)
            )
        else:
            revenue_retreat_amount = max_retreat

    return revenue_retreat_amount

def calculate_spread_inventory_scale(
    base_asset_amount_with_amm: int,
    base_asset_reserve: int,
    min_base_asset_reserve: int,
    max_base_asset_reserve: int,
    directional_spread: int,
    max_spread: int
) -> int:
    if base_asset_amount_with_amm == 0:
        return BID_ASK_SPREAD_PRECISION

    amm_inventory_pct = calculate_inventory_liquidity_ratio(
        base_asset_amount_with_amm,
        base_asset_reserve,
        min_base_asset_reserve,
        max_base_asset_reserve
    )

    inventory_scale_max = max(MAX_BID_ASK_INVENTORY_SKEW_FACTOR, max_spread * BID_ASK_SPREAD_PRECISION // max(directional_spread, 1))

    inventory_scale_capped = min(
        inventory_scale_max,
        BID_ASK_SPREAD_PRECISION + (inventory_scale_max * amm_inventory_pct // PERCENTAGE_PRECISION)
    )

    return inventory_scale_capped

def calculate_spread_leverage_scale(
    quote_asset_reserve: int,
    terminal_quote_asset_reserve: int,
    peg_multiplier: int,
    base_asset_amount_with_amm: int,
    reserve_price: int,
    total_fee_minus_distributions: int
) -> int:
    AMM_TIMES_PEG_TO_QUOTE_PRECISION_RATIO_I128 = 1e9 * 1e3 / 1e6
    AMM_TO_QUOTE_PRECISION_RATIO_I128 = 1e3
    net_base_asset_value = (
        (quote_asset_reserve - terminal_quote_asset_reserve) *
        peg_multiplier *
        AMM_TIMES_PEG_TO_QUOTE_PRECISION_RATIO_I128 // AMM_TO_QUOTE_PRECISION_RATIO_I128
    )

    local_base_asset_value = (
        base_asset_amount_with_amm *
        reserve_price *
        AMM_TO_QUOTE_PRECISION_RATIO_I128 *
        PRICE_PRECISION // AMM_TO_QUOTE_PRECISION_RATIO_I128
    )

    effective_leverage = max(0, local_base_asset_value - net_base_asset_value) * BID_ASK_SPREAD_PRECISION // (max(0, total_fee_minus_distributions) + 1)

    effective_leverage_capped = min(
        MAX_BID_ASK_INVENTORY_SKEW_FACTOR,
        BID_ASK_SPREAD_PRECISION + int(max(0, effective_leverage) + 1)
    )

    return effective_leverage_capped

def calculate_long_short_vol_spread(
    last_oracle_conf_pct: int,
    reserve_price: int,
    mark_std: int,
    oracle_std: int,
    long_intensity_volume: int,
    short_intensity_volume: int,
    volume_24h: int
) -> Tuple[int, int]:
    market_avg_std_pct = (oracle_std + mark_std) * PERCENTAGE_PRECISION // (2 * reserve_price)

    vol_spread = max(last_oracle_conf_pct, market_avg_std_pct // 2)

    factor_clamp_min = PERCENTAGE_PRECISION // 100  # .01
    factor_clamp_max = 16 * PERCENTAGE_PRECISION // 10  # 1.6

    long_vol_spread_factor = max(
        factor_clamp_min,
        min(factor_clamp_max, long_intensity_volume * PERCENTAGE_PRECISION // max(volume_24h, 1))
    )
    short_vol_spread_factor = max(
        factor_clamp_min,
        min(factor_clamp_max, short_intensity_volume * PERCENTAGE_PRECISION // max(volume_24h, 1))
    )

    return (
        max(last_oracle_conf_pct, (vol_spread * long_vol_spread_factor) // PERCENTAGE_PRECISION),
        max(last_oracle_conf_pct, (vol_spread * short_vol_spread_factor) // PERCENTAGE_PRECISION)
    )


def _calculate_market_open_bids_asks(base_asset_reserve, min_base_asset_reserve, max_base_asset_reserve):
    bids = max_base_asset_reserve-base_asset_reserve
    asks = base_asset_reserve - min_base_asset_reserve
    return bids,asks

def calculate_inventory_liquidity_ratio(
    base_asset_amount_with_amm: int,
    base_asset_reserve: int,
    min_base_asset_reserve: int,
    max_base_asset_reserve: int
) -> int:
    max_bids, max_asks = _calculate_market_open_bids_asks(base_asset_reserve, min_base_asset_reserve, max_base_asset_reserve)

    min_side_liquidity = min(max_bids, abs(max_asks))

    if abs(base_asset_amount_with_amm) < min_side_liquidity:
        amm_inventory_pct = (abs(base_asset_amount_with_amm) * PERCENTAGE_PRECISION) // max(min_side_liquidity, 1)
        amm_inventory_pct = min(amm_inventory_pct, PERCENTAGE_PRECISION)
    else:
        amm_inventory_pct = PERCENTAGE_PRECISION  # 100%

    return amm_inventory_pct

def calculate_spread(
    base_spread: int,
    last_oracle_reserve_price_spread_pct: int,
    last_oracle_conf_pct: int,
    max_spread: int,
    quote_asset_reserve: int,
    terminal_quote_asset_reserve: int,
    peg_multiplier: int,
    base_asset_amount_with_amm: int,
    reserve_price: int,
    total_fee_minus_distributions: int,
    net_revenue_since_last_funding: int,
    base_asset_reserve: int,
    min_base_asset_reserve: int,
    max_base_asset_reserve: int,
    mark_std: int,
    oracle_std: int,
    long_intensity_volume: int,
    short_intensity_volume: int,
    volume_24h: int
) -> Tuple[int, int]:
    
    long_vol_spread, short_vol_spread = calculate_long_short_vol_spread(
        last_oracle_conf_pct,
        reserve_price,
        mark_std,
        oracle_std,
        long_intensity_volume,
        short_intensity_volume,
        volume_24h
    )

    long_spread = max(base_spread // 2, long_vol_spread)
    short_spread = max(base_spread // 2, short_vol_spread)

    max_target_spread = max(max_spread, abs(last_oracle_reserve_price_spread_pct))

    if last_oracle_reserve_price_spread_pct < 0:
        long_spread = max(long_spread, abs(last_oracle_reserve_price_spread_pct) + long_vol_spread)
    elif last_oracle_reserve_price_spread_pct > 0:
        short_spread = max(short_spread, abs(last_oracle_reserve_price_spread_pct) + short_vol_spread)

    inventory_scale_capped = calculate_spread_inventory_scale(
        base_asset_amount_with_amm,
        base_asset_reserve,
        min_base_asset_reserve,
        max_base_asset_reserve,
        long_spread if base_asset_amount_with_amm > 0 else short_spread,
        max_target_spread
    )

    if base_asset_amount_with_amm > 0:
        long_spread = (long_spread * inventory_scale_capped) // BID_ASK_SPREAD_PRECISION
    elif base_asset_amount_with_amm < 0:
        short_spread = (short_spread * inventory_scale_capped) // BID_ASK_SPREAD_PRECISION

    if total_fee_minus_distributions <= 0:
        long_spread = (long_spread * DEFAULT_LARGE_BID_ASK_FACTOR) // BID_ASK_SPREAD_PRECISION
        short_spread = (short_spread * DEFAULT_LARGE_BID_ASK_FACTOR) // BID_ASK_SPREAD_PRECISION
    else:
        effective_leverage_capped = calculate_spread_leverage_scale(
            quote_asset_reserve,
            terminal_quote_asset_reserve,
            peg_multiplier,
            base_asset_amount_with_amm,
            reserve_price,
            total_fee_minus_distributions
        )

        if base_asset_amount_with_amm > 0:
            long_spread = (long_spread * effective_leverage_capped) // BID_ASK_SPREAD_PRECISION
        elif base_asset_amount_with_amm < 0:
            short_spread = (short_spread * effective_leverage_capped) // BID_ASK_SPREAD_PRECISION

    revenue_retreat_amount = calculate_spread_revenue_retreat_amount(
        base_spread,
        max_target_spread,
        net_revenue_since_last_funding
    )

    if revenue_retreat_amount != 0:
        if base_asset_amount_with_amm > 0:
            long_spread += revenue_retreat_amount
            short_spread += revenue_retreat_amount // 2
        elif base_asset_amount_with_amm < 0:
            long_spread += revenue_retreat_amount // 2
            short_spread += revenue_retreat_amount
        else:
            long_spread += revenue_retreat_amount // 2
            short_spread += revenue_retreat_amount // 2

    long_spread, short_spread = cap_to_max_spread(long_spread, short_spread, max_target_spread)

    return int(long_spread), int(short_spread)




from datetime import datetime as dt
import sys
from tokenize import tabsize
import driftpy
import pandas as pd
import numpy as np
from driftpy.math.oracle import *
from constants import ALL_MARKET_NAMES

import plotly.express as px

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.accounts import (
    get_perp_market_account,
    get_spot_market_account,
    get_user_account,
    get_state_account,
)
from driftpy.constants.numeric_constants import *
from driftpy.clearing_house_user import get_token_amount
import os
import json
import streamlit as st
from driftpy.types import MarginRequirementType
from driftpy.constants.banks import devnet_banks, Bank
from driftpy.constants.markets import devnet_markets, Market
from driftpy.addresses import *
from dataclasses import dataclass
from solana.publickey import PublicKey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import time
from enum import Enum
from driftpy.math.margin import MarginCategory, calculate_asset_weight
import datetime

async def vamm(ch: ClearingHouse):
    # Create sliders for each variable
    mi1 = st.selectbox('perp market index:', [0,1,2])
    market: PerpMarket = await get_perp_market_account(ch.program, mi1)
    nom = bytes(market.name).decode('utf-8')
    market_index = market.market_index
    st.write(nom)

    tabs = st.tabs(['liquidity', 'spread'])
    with tabs[0]:
        st.text(f'vAMM Liquidity (bids= {(market.amm.max_base_asset_reserve-market.amm.base_asset_reserve) / 1e9} | asks={(market.amm.base_asset_reserve-market.amm.min_base_asset_reserve) / 1e9})')
        t0, t1, t2 = st.columns([1,1,5])
        dir = t0.selectbox('direction:', ['buy', 'sell'], key='selectbox-'+str(market_index))
        ba = t1.number_input('base amount:', value=1, key='numin-'+str(market_index))
        bid_price = market.amm.bid_quote_asset_reserve/market.amm.bid_base_asset_reserve * market.amm.peg_multiplier/1e6
        ask_price = market.amm.ask_quote_asset_reserve/market.amm.ask_base_asset_reserve * market.amm.peg_multiplier/1e6
        reserve_price = market.amm.quote_asset_reserve/market.amm.base_asset_reserve * market.amm.peg_multiplier/1e6
        def px_impact(dir, ba):
            f = ba / (market.amm.base_asset_reserve/1e9)
            if dir == 'buy':
                pct_impact = (1/((1-f)**2) - 1) * 100
            else:
                pct_impact = (1 - 1/((1+f)**2)) * 100
            return pct_impact
        px_impact = px_impact(dir, ba)
        price = (ask_price * (1+px_impact/100)) if dir=='buy' else (bid_price * (1-px_impact/100))
        t2.text(f'vAMM stats: \n px={price} \n px_impact={px_impact}%')

    with tabs[1]:
        c1,c2,c3 = st.columns(3)
        base_spread = c1.slider("Base Spread", min_value=0.0, max_value=1.0/100, value=market.amm.base_spread/1e6)
        max_spread = c2.slider("Max Spread", min_value=0.0, max_value=1.0, value=market.amm.max_spread /1e6)

        d1,d2,d3 = st.columns(3)

        last_oracle_reserve_price_spread_pct = d1.slider("Last Oracle Reserve Price Spread %", value=market.amm.last_oracle_reserve_price_spread_pct/1e6)
        last_oracle_conf_pct = d2.slider("Last Oracle Conf %",  value=market.amm.last_oracle_conf_pct/1e6)

        with st.expander('amm'):
            a1, a2, a3, a4 = st.columns(4)
            n1, n2, n3, n4 = st.columns(4)

            quote_asset_reserve = a1.number_input("Quote Asset Reserve", value=market.amm.quote_asset_reserve/1e9)
            terminal_quote_asset_reserve = a2.number_input("Terminal Quote Asset Reserve", value=market.amm.terminal_quote_asset_reserve/1e9)
            peg_multiplier = a4.number_input("Peg Multiplier", value=market.amm.peg_multiplier/1e3)
            reserve_price = n1.number_input("Reserve Price",  value=reserve_price)
            total_fee_minus_distributions = n4.number_input("Total Fee Minus Distributions", value=market.amm.total_fee_minus_distributions/1e6)
            base_asset_reserve = a3.number_input("Base Asset Reserve",  value=market.amm.base_asset_reserve/1e9)
            min_base_asset_reserve = n2.number_input("Min Base Asset Reserve",value=market.amm.min_base_asset_reserve/1e9)
            max_base_asset_reserve = n3.number_input("Max Base Asset Reserve",  value=market.amm.max_base_asset_reserve/1e9)

        p1, p2, p3 = st.columns(3)
        net_revenue_since_last_funding = p1.number_input("Net Revenue Since Last Funding", value=market.amm.net_revenue_since_last_funding/1e6)
        base_asset_amount_with_amm = p2.slider("Base Asset Amount with AMM",
                                                # min_value=-1000, max_value=1000, 
                                                                                            # value=1)

                                                value=market.amm.base_asset_amount_with_amm/1e9)


        v1, v2, v3 = st.columns(3)
        mark_std = v1.number_input("Mark Standard Deviation", value=market.amm.mark_std/1e6)
        oracle_std = v2.number_input("Oracle Standard Deviation", value=market.amm.oracle_std/1e6)

        s2,s3,s4 = st.columns(3)

        long_intensity_volume = s2.slider("Long Intensity Volume", value=market.amm.long_intensity_volume/1e6)
        short_intensity_volume = s3.slider("Short Intensity Volume", value=market.amm.short_intensity_volume/1e6) 
        volume_24h = s4.slider("Volume 24h", value=market.amm.volume24h/1e6) 

        # Call the calculate_spread function with the slider values
        result = calculate_spread(
            base_spread * 1e6,
            last_oracle_reserve_price_spread_pct  * 1e6,
            last_oracle_conf_pct * 1e6,
            max_spread * 1e6,
            quote_asset_reserve * 1e9,
            terminal_quote_asset_reserve * 1e9,
            peg_multiplier * 1e3,
            base_asset_amount_with_amm * 1e9,
            reserve_price * 1e6,
            total_fee_minus_distributions * 1e6,
            net_revenue_since_last_funding * 1e6,
            base_asset_reserve * 1e9,
            min_base_asset_reserve * 1e9,
            max_base_asset_reserve * 1e9,
            mark_std * 1e6,
            oracle_std * 1e6,
            long_intensity_volume * 1e6,
            short_intensity_volume * 1e6,
            volume_24h * 1e6,
        )

        # Display the result
        st.write("Result:", result)