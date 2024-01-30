
from typing import Tuple
from driftpy.constants.numeric_constants import *
from driftpy.types import PerpMarketAccount

from driftpy.math.amm import calculate_spread
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

def calculate_spread_local(
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
from driftpy.accounts.oracle import *
from constants import ALL_MARKET_NAMES

import plotly.express as px

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.accounts import (
    get_perp_market_account,
    get_spot_market_account,
    get_user_account,
    get_state_account,
)
from driftpy.constants.numeric_constants import *
from driftpy.drift_user import get_token_amount
import os
import json
import streamlit as st
from driftpy.types import MarginRequirementType
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from driftpy.addresses import *
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import time
from enum import Enum
from driftpy.math.margin import MarginCategory, calculate_asset_weight
import datetime


def calculate_market_open_bid_ask(base_asset_reserve, min_base_asset_reserve, max_base_asset_reserve, step_size=None):
    # open orders
    if min_base_asset_reserve < base_asset_reserve:
        open_asks = (base_asset_reserve - min_base_asset_reserve) * -1

        if step_size and abs(open_asks) // 2 < step_size:
            open_asks = 0
    else:
        open_asks = 0

    if max_base_asset_reserve > base_asset_reserve:
        open_bids = max_base_asset_reserve - base_asset_reserve

        if step_size and open_bids // 2 < step_size:
            open_bids = 0
    else:
        open_bids = 0

    return open_bids, open_asks

def calculate_inventory_liquidity_ratio(base_asset_amount_with_amm, base_asset_reserve, min_base_asset_reserve, max_base_asset_reserve):
    # inventory skew
    open_bids, open_asks = calculate_market_open_bid_ask(base_asset_reserve, min_base_asset_reserve, max_base_asset_reserve)

    min_side_liquidity = min(abs(open_bids), abs(open_asks))

    inventory_scale_bn = min(
        base_asset_amount_with_amm * PERCENTAGE_PRECISION // max(min_side_liquidity, 1),
        PERCENTAGE_PRECISION
    )
    return inventory_scale_bn


def clamp(value, min_value, max_value):
        return max(min(value, max_value), min_value)



def calculate_reservation_price_offset(
    reserve_price,
    last_24h_avg_funding_rate,
    liquidity_fraction,
    oracle_twap_fast,
    mark_twap_fast,
    oracle_twap_slow,
    mark_twap_slow,
    max_offset_pct
):
    offset = 0
    # max_offset_pct = (1e6 / 400) 
    # base_inventory_threshold = min_order_size * 5
    # calculate quote denominated market premium

    max_offset_in_price = int(max_offset_pct * reserve_price / PERCENTAGE_PRECISION)

    # Calculate quote denominated market premium
    mark_premium_minute = clamp(mark_twap_fast - oracle_twap_fast, -max_offset_in_price, max_offset_in_price)
    mark_premium_hour = clamp(mark_twap_slow - oracle_twap_slow, -max_offset_in_price, max_offset_in_price)

    # convert last_24h_avg_funding_rate to quote denominated premium
    mark_premium_day = clamp(last_24h_avg_funding_rate / FUNDING_RATE_BUFFER * 24, -max_offset_in_price, max_offset_in_price)

    mark_premium_avg = (mark_premium_day + mark_premium_hour + mark_premium_minute ) / 3
    

    mark_offset = mark_premium_avg * PRICE_PRECISION / reserve_price
    inv_offset = liquidity_fraction * max_offset_pct / PERCENTAGE_PRECISION
    offset = mark_offset + inv_offset

    if np.sign(inv_offset) != np.sign(mark_offset):
        offset = 0

    clamped_offset = clamp(offset, -max_offset_in_price, max_offset_in_price)

    return clamped_offset


async def vamm(ch: DriftClient):
    tabs = st.tabs(['overview', 'vamm fills', 'risks', 'per market', 'spread calculator'])
    await ch.account_subscriber.update_cache()
    with tabs[0]:
        curve_override = st.number_input('curve_override', -1, 200, -1)
        num_m = ch.get_state_account().number_of_markets
        res = []
        profit_res = []
        for i in range(num_m):
            market: PerpMarketAccount = ch.get_perp_market_account(i)
            spread = (market.amm.long_spread + market.amm.short_spread)/1e6 * 100
            op = market.amm.historical_oracle_data.last_oracle_price/1e6
            bids = (market.amm.max_base_asset_reserve-market.amm.base_asset_reserve) / 1e9 * op
            asks = (market.amm.base_asset_reserve-market.amm.min_base_asset_reserve) / 1e9 * op
            amm_owned = (1 - market.amm.user_lp_shares / market.amm.sqrt_k) * 100
            
            if curve_override < 0:
                curve_i = market.amm.curve_update_intensity
            else:
                curve_i = curve_override
            max_offset_pct = max(market.amm.max_spread / 5, PERCENTAGE_PRECISION / 10000 * (curve_i-100))
            if curve_i <= 100:
                max_offset_pct = 0
            liquidity_fraction = calculate_inventory_liquidity_ratio(market.amm.base_asset_amount_with_amm,
                                                                     market.amm.base_asset_reserve,
                                                                     market.amm.min_base_asset_reserve,
                                                                    market.amm.max_base_asset_reserve,
                                                                     )
            reserve_price = market.amm.quote_asset_reserve/market.amm.base_asset_reserve * market.amm.peg_multiplier
            offset = calculate_reservation_price_offset(reserve_price,
                                                        market.amm.last24h_avg_funding_rate, 
                                                        liquidity_fraction *np.sign(market.amm.base_asset_amount_with_amm+market.amm.base_asset_amount_with_unsettled_lp),
                                                        market.amm.historical_oracle_data.last_oracle_price_twap5min,
                                                        market.amm.last_mark_price_twap5min,
                                                        market.amm.historical_oracle_data.last_oracle_price_twap,
                                                        market.amm.last_mark_price_twap,
                                                        max_offset_pct
                                                        )
            # offset /= op
            offset *= 100

            oi = max(market.amm.base_asset_amount_long, -market.amm.base_asset_amount_short)
            max_oi = market.amm.max_open_interest
            res.append((bytes(market.name).decode('utf-8'), 
                        market.contract_tier,
                        offset/1e6, spread, 
                        market.amm.base_spread/1e6 * 100, 
                        market.amm.max_spread/1e6 * 100, 
                         bids, asks, amm_owned,
                         oi/1e9,
                         max_oi/1e9,
                         oi/max_oi * 100,
                         market.amm.curve_update_intensity
                         ))
            
            profit_res.append({'market': bytes(market.name).decode('utf-8'), 
                               'tier': market.contract_tier, 
                                'ex_fee': market.amm.total_exchange_fee / QUOTE_PRECISION,
                                'mm_fee': market.amm.total_mm_fee/ QUOTE_PRECISION,
                                'tfmd': market.amm.total_fee_minus_distributions/ QUOTE_PRECISION,
                               'liq_fee': market.amm.total_liquidation_fee/ QUOTE_PRECISION,
                               'withdrawn_fee': market.amm.total_fee_withdrawn/ QUOTE_PRECISION,
                               'revenue_per_period': market.insurance_claim.revenue_withdraw_since_last_settle/QUOTE_PRECISION,
                               'max_revenue_withdraw_per_period': market.insurance_claim.max_revenue_withdraw_per_period/QUOTE_PRECISION,
                               'quote_settled_insurance': market.insurance_claim.quote_settled_insurance/QUOTE_PRECISION,
                               'quote_max_insurance': market.insurance_claim.quote_max_insurance/QUOTE_PRECISION,
            })
        
        df = pd.DataFrame(res,  
                              columns=['market', 'tier', 'offset (%)', 'spread (%)', 'base spread (%)', 'max spread (%)',
                                        'bids ($)', 'asks ($)', 'amm owned (%)',
                                        'OI',
                                        'Max OI',
                                        'OI %',
                                        'curve intensity'
                                        ]
                              )
        st.write(df)

        df = pd.DataFrame(profit_res)
        st.write(df)  

    with tabs[1]:
        res = []
        for i in range(num_m):
            market: PerpMarketAccount = ch.get_perp_market_account(i)
            op = market.amm.historical_oracle_data.last_oracle_price/1e6
            bids = (market.amm.max_base_asset_reserve-market.amm.base_asset_reserve) / 1e9
            asks = (market.amm.base_asset_reserve-market.amm.min_base_asset_reserve) / 1e9 
            res.append((bytes(market.name).decode('utf-8'), 
                        market.amm.max_fill_reserve_fraction,
                        bids / 2,
                        asks / 2,
                        market.amm.base_asset_reserve / market.amm.max_fill_reserve_fraction / 1e9,
                        market.amm.base_asset_reserve / market.amm.max_fill_reserve_fraction / 1e9 * op,
            ))


        df = pd.DataFrame(res,  
                              columns=['market', 'market.amm.max_fill_reserve_fraction', 
                                       'max_sell', 'max_buy',
                                       'max_base_fill', 
                                       'max_base_fill ($)',
                                        ]
                              )
        st.write(df)
    with tabs[2]:
        st.write('risk is important')
        ff = st.text_input('file source:', '')#'driftv2_user_snap1.csv.gz')
        try:
            res = pd.read_csv(ff)
            st.plotly_chart(res['leverage'].plot(kind='box'))
        except:
            st.warning('couldnt load file source')

    with tabs[3]:
        # Create sliders for each variable
        num_m = ch.get_state_account().number_of_markets
        mi1 = st.selectbox('perp market index:', range(num_m))
        market: PerpMarketAccount = ch.get_perp_market_account(mi1)
        nom = bytes(market.name).decode('utf-8')
        market_index = market.market_index
        st.write(nom)

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

    with tabs[4]:
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
        result = calculate_spread_local(
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

        op = ch.get_oracle_price_data_for_perp_market(0)
        result2 = calculate_spread(market.amm, oracle_price_data=op)
        st.write("Result2:", result2)

