from math import exp
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import json
from urllib.request import urlopen

root = "https://raw.githubusercontent.com/drift-labs/drift-sim/master/backtest/"

st.title('Drift v2 Simulations')
st.markdown('[https://github.com/drift-labs/drift-sim](https://github.com/drift-labs/drift-sim) | [@driftprotocol](https://twitter.com/@driftprotocol)')


experiments = ['lunaCrash']
experiment = st.selectbox(
        "Choose experiment", list(experiments), 
    )
st.text('1. simulate: simulate agent interactions based on oracle price action')
events_df = pd.read_csv(root+"/"+experiment+"/"+"events.csv")
with st.expander(experiment+" events sequence"):
    st.table(events_df)

with st.expander(experiment+" run_info.json"):
    run_info_ff = root+"/"+experiment+"/"+"run_info.json"

    ff = urlopen(run_info_ff)
    data = pd.DataFrame(json.loads(ff.read()), index=[0]).T
    st.markdown('[github](https://github.com/drift-labs/drift-sim/tree/master/backtest/'+experiment+')')
    st.table(data)


st.text('')
st.text('')
st.text('')
st.text('')
st.text('')
st.text('')

st.text('2. backtest: runs events against drift smart contract (each iteration is a trial)')
st.markdown('[https://github.com/drift-labs/protocol-v2](https://github.com/drift-labs/protocol-v2)')

trials = ['trial_no_oracle_guards', 'trial_spread_250', 'trial_spread_1000', 'trial_oracle_guards']
trial = st.selectbox(
        "Choose trial", list(trials), 
    )

user_selected = st.selectbox('user', options=['all'] + [str(x) for x in list(range(0,20))], index=0)

if user_selected == 'all':
    all_user_stats = pd.read_csv(
        f"https://raw.githubusercontent.com/drift-labs/drift-sim/master/backtest/{experiment}/{trial}/all_user_stats.csv")
    all_user_stats /= 1e6 #todo

else:
    all_user_stats = pd.read_csv(
        f"https://raw.githubusercontent.com/drift-labs/drift-sim/master/backtest/{experiment}/{trial}/result_user_{user_selected}.csv")
fig = px.line( all_user_stats, 
        title='user tvl'+' ('+experiment+':'+trial+')'
    )
st.plotly_chart(fig)
 


for ff in ['perp_market0', 'spot_market0']:
    df = pd.read_csv(root+"/"+experiment+"/"+trial+"/"+ff+".csv")
    df_types = df.dtypes.reset_index().groupby([0]).agg(list)
    df_types.index = [str(x) for x in df_types.index]
    print(df_types)
    dd_types = ['numerics', 'other']
    columns = {dd_type: [] for dd_type in dd_types}

    with st.expander(ff+" column selection"):
        for dd_type in dd_types:
            df_types_map = {'numerics':['int64', 'float64'], 'other':['bool','object']}[dd_type]
            typed_cols = [df_types.loc[str(df_type)].values[0] for df_type in df_types_map if df_type in df_types.index]
            typed_cols = [item for sublist in typed_cols for item in sublist]
            default_cols = []

            if 'perp' in ff and str(dd_type) == 'numerics':
                default_cols= [
                'market.expiry_price', 
                'market.amm.historical_oracle_data.last_oracle_price',
                'market.amm.historical_oracle_data.last_oracle_price_twap'
                ]

            columns[dd_type] = st.multiselect(
                    "Choose "+str(dd_type)+" columns", typed_cols, default_cols
                )


    fig = px.line(df[[item for sublist in columns.values() for item in sublist]], 
        title=ff+' ('+experiment+':'+trial+')'
    )
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-1,
    xanchor="right",
    x=1
))
    st.plotly_chart(fig)
