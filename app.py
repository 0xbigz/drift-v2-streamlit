from math import exp
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import json
from urllib.request import urlopen
import numpy as np
from simulations import sim_page


st.set_page_config(
    'Drift v2 Simulations',
    layout='wide',
)

tab = st.sidebar.radio(
    "Select Tab:",
    ('Overview', 'Simulations', 'Tweets'))

if tab == 'Overview':
    st.title('Drift v2')
    repo = "https://github.com/drift-labs/protocol-v2"
    st.markdown('['+repo+']('+repo+') | [@driftprotocol](https://twitter.com/@driftprotocol)')

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
