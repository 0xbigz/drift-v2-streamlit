
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


def imf_page(clearing_house: ClearingHouse):    
    init_liability_wgt = st.select_slider('liability wgt', [.02, .03, .05, .1, .2, .5, 1])
    maint_liability_wgt = st.select_slider('maint liability wgt', [.02, .03, .05, .1, .2, .5, 1])

    imf = st.select_slider('imf', [0, .000005, .00001, .00005, .0001, .0005, .001, .005, .01, .05, .1])
    base = st.number_input('base asset amount')
    oracle_px = st.number_input('oracle price')
    st.text('notional='+str(oracle_px * base))

    liability_wgt_n = init_liability_wgt


    if(imf!=0):
        liability_wgt_n = init_liability_wgt - init_liability_wgt/(1/ imf)
    dd = np.sqrt(np.abs(base))
    res = max(init_liability_wgt, liability_wgt_n + imf * dd)
    st.text('max('+str(liability_wgt_n)+' + '+ str(imf) + ' * ' + str(dd)+', '+str(init_liability_wgt)+')')
    st.text('='+str(res))
    st.text('(max leverage= ' + str(1/init_liability_wgt) + ' -> ' + str(1/res)+'x )')



    def calc_size_liab(wgt, imf, base):
        liability_wgt_n = wgt
        if(imf != 0):
            liability_wgt_n = wgt * .8 #/(1/(imf/10000))
        dd = np.sqrt(np.abs(base))
        res = max(wgt, liability_wgt_n + imf * dd)
        return res

    index = np.linspace(0, max(10000, base * oracle_px), 1000)
    df = pd.Series([
        1/calc_size_liab(init_liability_wgt, imf, x) 
        for x in index
        ]
    )
    df.index = index

    df2 = pd.Series([
        1/calc_size_liab(maint_liability_wgt, imf, x) 
        for x in index
        ]
    )
    df2.index = index

    st.plotly_chart(pd.concat({'init': df, 'maint': df2},axis=1).plot(kind='line'))

