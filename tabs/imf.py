
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"
from driftpy.math.margin import MarginCategory, calculate_market_margin_ratio, calculate_asset_weight

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
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
from glob import glob


async def imf_page(clearing_house: DriftClient):   
    tabs = st.tabs(['overview', 'calculator']) 
    ch = clearing_house
    if ch.account_subscriber.cache is None:
        await ch.account_subscriber.update_cache()
    state = ch.get_state_account()

    with tabs[0]:
        overview = {}
        not_size = st.number_input('notional size:', value=100_000)
        for i in range(state.number_of_markets):
            market = ch.get_perp_market_account(i)
            nom = bytes(market.name).decode('utf-8')
            size = not_size * BASE_PRECISION/(market.amm.historical_oracle_data.last_oracle_price/PRICE_PRECISION)
            res = [
                calculate_market_margin_ratio(market, 0, MarginCategory.INITIAL),
                calculate_market_margin_ratio(market, size, MarginCategory.INITIAL),
                calculate_market_margin_ratio(market, 0, MarginCategory.MAINTENANCE),
                calculate_market_margin_ratio(market, size, MarginCategory.MAINTENANCE),
            ]
            res = [1/(x/MARGIN_PRECISION) for x in res]
            res = [market.imf_factor/1e6] + res
            overview[nom] = res

        s1, s2 = st.columns(2)


        s1.write('perps')
        s1.write(pd.DataFrame(overview, index=['imf factor', 'init leverage', 
                                               'init leverage (size)', 
                                               'maint leverage',
                                               'maint leverage (size)']).T.reset_index())


        overview2 = {}
        for i in range(state.number_of_spot_markets):
            market = ch.get_spot_market_account(i)
            nom = bytes(market.name).decode('utf-8')
            oracle_price = market.historical_oracle_data.last_oracle_price
            size = not_size * (10 ** market.decimals)/(oracle_price/PRICE_PRECISION)
            res = [
                calculate_asset_weight(0, oracle_price, market, MarginCategory.INITIAL),
                calculate_asset_weight(size, oracle_price, market, MarginCategory.INITIAL),
                calculate_asset_weight(size, oracle_price, market, MarginCategory.MAINTENANCE),
            ]
            # 1 is liability weight of usdc
            res = [1/(1 - x/MARGIN_PRECISION + 1e-6) for x in res]
            res = [market.imf_factor/1e6] + res
            overview2[nom] = res

        s2.write('spot')
        s2.write(pd.DataFrame(overview2, index=['imf factor', 'init leverage', 'init leverage (size)', 'maint leverage']).T.reset_index())

    with tabs[1]:
        col1, col2 = st.columns(2)
        dd = col1.selectbox('mode:', ['custom', 'perp', 'spot'])
        oo = 0

        if dd != 'custom':
            n = state.number_of_markets
            s = 0
            if dd == 'spot':
                s = 1
                n = state.number_of_spot_markets
            oo = col2.selectbox('market:', range(s, n), 0)
            if dd == 'perp':
                market = ch.get_perp_market_account(oo)
                init_liability_wgt = market.margin_ratio_initial/1e4
                maint_liability_wgt = market.margin_ratio_maintenance/1e4
                ogimf = market.imf_factor/1e6
                oracle_px = market.amm.historical_oracle_data.last_oracle_price/1e6

            else:
                market = ch.get_spot_market_account(oo)
                init_liability_wgt = (market.initial_liability_weight/1e4 - 1)
                maint_liability_wgt = (market.maintenance_liability_weight/1e4 - 1)
                ogimf = market.imf_factor/1e6
                oracle_px = market.historical_oracle_data.last_oracle_price/1e6

            st.write(bytes(market.name).decode('utf-8'))
            max_imf = .01 
            if dd == 'spot':
                max_imf = .5
            imf = st.slider('imf', 0.0, max_imf, ogimf, step=.00005)
            st.write('imf factor=', ogimf, '->', imf)
            st.write('init_liability_wgt=', init_liability_wgt)
            st.write('maint_liability_wgt=', maint_liability_wgt)

        else:
            init_liability_wgt = st.select_slider('liability wgt', [.02, .03, .05, .1, .2, .5, 1])
            maint_liability_wgt = st.select_slider('maint liability wgt', [.02, .03, .05, .1, .2, .5, 1])

            imf = st.select_slider('imf', [0, .000005, .00001, .00005, .0001, .0005, .001, .005, .01, .05, .1])
            oracle_px = st.number_input('oracle price')
        base = st.number_input('base asset amount', value=1)

        st.text('notional='+str(oracle_px * base))

        liability_wgt_n = init_liability_wgt

        
        if(imf!=0):
            liability_wgt_n = liability_wgt_n * .8 

        ddsize = np.sqrt(np.abs(base))
        res = max(init_liability_wgt, liability_wgt_n + imf * ddsize)
        st.text('max('+str(liability_wgt_n)+' + '+ str(imf) + ' * ' + str(ddsize)+', '+str(init_liability_wgt)+')')
        st.text('='+str(res))
        st.text('(max leverage= ' + str(1/init_liability_wgt) + ' -> ' + str(1/res)+'x )')

        is_spot = False
        if dd == 'spot':
            is_spot = True
        st.write('is_spot', is_spot)

        def calc_size_liab(wgt, imf, base, is_spot):
            if is_spot:
                wgt += 1

            liability_wgt_n = wgt
            if(imf != 0):
                liability_wgt_n = wgt * .8 #/(1/(imf/10000))
            dd = np.sqrt(np.abs(base))
            res = max(wgt, liability_wgt_n + imf * dd)

            if is_spot:
                res -= 1

            return res

        if oracle_px != 0:
            oo = max(10000/oracle_px, base)
        else:
            oo = base

        step = 1000 #int(np.round(oo/1000) * 1000)
        if oo > 100000:
            step *= 10
        if oo > 1000000:
            step *= 10


        index = np.linspace(0, oo, step)
        df = pd.Series([
            1/calc_size_liab(init_liability_wgt, imf, x, is_spot) 
            for x in index
            ]
        )
        df.index = index

        df2 = pd.Series([
            1/calc_size_liab(maint_liability_wgt, imf, x, is_spot) 
            for x in index
            ]
        )
        df2.index = index

        imf_df = pd.concat({'init': df, 'maint': df2},axis=1)
        imf_df.index *= oracle_px

        fig = imf_df.plot(kind='line')

        fig.update_layout(
        title="IMF Discount",
        xaxis_title="$ Notional",
        yaxis_title="Leverage",
        )

        st.plotly_chart(fig)

