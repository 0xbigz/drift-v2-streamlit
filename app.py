from math import exp
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import json
from urllib.request import urlopen
import numpy as np
import os

import asyncio
import datetime

from logs import log_page
from simulations import sim_page
from pid import show_pid_positions
from driftpy.constants.config import configs

from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from fees import fee_page
from if_stakers import insurance_fund_page
from userstats import show_user_stats
from orders import orders_page
from platyperps import show_platyperps
from liquidity import mm_page

def main():
    st.set_page_config(
        'Drift v2',
        layout='wide',
        page_icon="👾"
    )

    current_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    if st.sidebar.button('Clear Cache'):
        st.experimental_memo.clear()

    env = st.sidebar.radio('env', ('mainnet-beta', 'devnet'))

    url = "🤫"
    if env == 'devnet':
        url = 'https://api.'+env+'.solana.com'
    else:
        url = "🤫"

    rpc = st.sidebar.text_input('rpc', url)
    if env == 'mainnet-beta' and (rpc == '🤫' or  rpc == ''):
        rpc = os.environ['ANCHOR_PROVIDER_URL']

    def query_string_callback():
        st.experimental_set_query_params(**{'tab': st.session_state.query_key})
    query_p = st.experimental_get_query_params()
    query_tab = query_p.get('tab', ['Overview'])[0]

    tab_options = ('Overview', 'Simulations', 'Logs', 'Fee-Schedule', 'IF-Stakers', 'User-Stats', 'DLOB', 'MM', 'Config', 'Social', 'PlatyPerps')
    query_index = 0
    for idx, x in enumerate(tab_options):
        if x.lower() == query_tab.lower():
            query_index = idx

    tab = st.sidebar.radio(
        "Select Tab:",
        tab_options,
        query_index,
        on_change=query_string_callback,
        key='query_key'
        )

    if env == 'mainnet-beta':
        config = configs['mainnet']
    else:
        config = configs[env]

    kp = Keypair() # random wallet
    wallet = Wallet(kp)
    connection = AsyncClient(rpc)
    provider = Provider(connection, wallet)
    clearing_house: ClearingHouse = ClearingHouse.from_config(config, provider)

    clearing_house.time = current_time
    
    st.title(f'Drift v2: {tab}')

    if tab.lower() == 'overview':
        loop = asyncio.new_event_loop()
        loop.run_until_complete(show_pid_positions(clearing_house))

    elif tab.lower() == 'config':  
        with st.expander(f"pid={clearing_house.program_id} config"):
            st.json(config.__dict__)

    elif tab.lower() == 'logs':
        loop = asyncio.new_event_loop()
        loop.run_until_complete(log_page(rpc, clearing_house))

    elif tab.lower() == 'simulations':
        sim_page()

    elif tab.lower() == 'fee-schedule':
        loop = asyncio.new_event_loop()
        loop.run_until_complete(fee_page(clearing_house))

    elif tab.lower() == 'if-stakers':
        loop = asyncio.new_event_loop()
        loop.run_until_complete(insurance_fund_page(clearing_house))

    elif tab.lower() == 'user-stats':
        loop = asyncio.new_event_loop()
        loop.run_until_complete(show_user_stats(clearing_house))
    
    elif tab.lower() == 'dlob':
        orders_page(clearing_house)
    elif tab.lower() == 'mm':
        mm_page(clearing_house)
    elif tab.lower() == 'social':

        repo = "https://github.com/drift-labs/protocol-v2"
        st.markdown('['+repo+']('+repo+') | [@driftprotocol](https://twitter.com/@driftprotocol)')

        tweets = {
            'cindy': 'https://twitter.com/cindyleowtt/status/1569713537454579712',
            '0xNineteen': 'https://twitter.com/0xNineteen/status/1571926865681711104',
        }
        st.header('twitter:')
        st.table(pd.Series(tweets))

    elif tab.lower() == 'platyperps':
        loop = asyncio.new_event_loop()
        loop.run_until_complete(show_platyperps(clearing_house))

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

if __name__ == '__main__':
    main()