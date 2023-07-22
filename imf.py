
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


async def imf_page(clearing_house: ClearingHouse):    
    ch = clearing_house
    state = await get_state_account(ch.program)
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
            market = await get_perp_market_account(ch.program, oo)
            init_liability_wgt = market.margin_ratio_initial/1e4
            maint_liability_wgt = market.margin_ratio_maintenance/1e4
            ogimf = market.imf_factor/1e6
            oracle_px = market.amm.historical_oracle_data.last_oracle_price/1e6

        else:
            market = await get_spot_market_account(ch.program, oo)
            init_liability_wgt = (market.initial_liability_weight/1e4 - 1)
            maint_liability_wgt = (market.maintenance_liability_weight/1e4 - 1)
            ogimf = market.imf_factor/1e6
            oracle_px = market.historical_oracle_data.last_oracle_price/1e6

        st.write(bytes(market.name).decode('utf-8'))
        max_imf = .1 
        if dd == 'spot':
            max_imf = .01
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

