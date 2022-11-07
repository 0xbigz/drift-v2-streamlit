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

from logs import view_logs
from simulations import sim_page
from pid import show_pid_positions
from driftpy.constants.config import configs

async def main():

    st.set_page_config(
        'Drift v2 Simulations',
        layout='wide',
        page_icon="👾"
    )
    env = st.sidebar.radio('env', ('mainnet-beta', 'devnet'))
    rpc = st.sidebar.text_input('rpc', 'https://api.'+env+'.solana.com')
    tab = st.sidebar.radio(
        "Select Tab:",
        ('Overview', 'Simulations', 'Tweets', 'Logs'))

    if tab == 'Overview':
        st.title('Drift v2')
        repo = "https://github.com/drift-labs/protocol-v2"
        st.markdown('['+repo+']('+repo+') | [@driftprotocol](https://twitter.com/@driftprotocol)')
        env = 'mainnet-beta'
        await show_pid_positions('', rpc)

    elif tab == 'Logs':
        st.title('Drift v2: Logs')
        repo = "https://github.com/drift-labs/protocol-v2"
        st.markdown('['+repo+']('+repo+') | [@driftprotocol](https://twitter.com/@driftprotocol)')
        await view_logs(rpc)

    elif tab == 'Simulations':
        st.title('Drift v2 Simulations')
        repo = "https://github.com/drift-labs/drift-sim"
        st.markdown('['+repo+']('+repo+') | [@driftprotocol](https://twitter.com/@driftprotocol)')
        sim_page()

    elif tab == 'Tweets':
        tweets = {
            'cindy': 'https://twitter.com/cindyleowtt/status/1569713537454579712',
            '0xNineteen': 'https://twitter.com/0xNineteen/status/1571926865681711104',
        
        }
        st.table(pd.Series(tweets))

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())