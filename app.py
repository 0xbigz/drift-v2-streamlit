from math import exp
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import json
from urllib.request import urlopen
import numpy as np

import aiohttp
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

from if_stakers import insurance_fund_page

async def main():
    st.set_page_config(
        'Drift v2',
        layout='wide',
        page_icon="👾"
    )

    last_update = st.sidebar.text('updated: ' + datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    env = st.sidebar.radio('env', ('mainnet-beta', 'devnet'))
    rpc = st.sidebar.text_input('rpc', 'https://api.'+env+'.solana.com')
    tab = st.sidebar.radio(
        "Select Tab:",
        ('Overview', 'Simulations', 'Tweets', 'Logs', 'IF-Stakers'))

    if env == 'mainnet-beta':
        config = configs['mainnet']
    else:
        config = configs[env]

    kp = Keypair() # random wallet
    wallet = Wallet(kp)
    connection = AsyncClient(rpc)
    provider = Provider(connection, wallet)
    clearing_house: ClearingHouse = ClearingHouse.from_config(config, provider)
    
    st.title(f'Drift v2: {tab}')

    repo = "https://github.com/drift-labs/protocol-v2"
    st.markdown('['+repo+']('+repo+') | [@driftprotocol](https://twitter.com/@driftprotocol)')
    
    with st.expander(f"pid={clearing_house.program_id} config"):
        st.json(config.__dict__)

    if tab == 'Overview':
        await show_pid_positions(rpc, clearing_house)

    elif tab == 'Logs':
        await log_page(rpc, clearing_house)

    elif tab == 'Simulations':
        sim_page()

    elif tab == 'IF-Stakers':
        await insurance_fund_page(clearing_house)

    elif tab == 'Tweets':
        tweets = {
            'cindy': 'https://twitter.com/cindyleowtt/status/1569713537454579712',
            '0xNineteen': 'https://twitter.com/0xNineteen/status/1571926865681711104',
        }
        st.table(pd.Series(tweets))

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())