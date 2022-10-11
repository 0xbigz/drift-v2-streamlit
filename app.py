import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

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



st.text('')
st.text('')
st.text('')
st.text('')
st.text('')
st.text('')

st.text('2. backtest: runs events against drift smart contract (each iteration is a trial)')

trials = ['trial_no_oracle_guards', 'trial_spread', 'trial_oracle_guards']
trial = st.selectbox(
        "Choose trial", list(trials), 
    )


df = pd.read_csv(root+"/"+experiment+"/"+trial+"/result_market0.csv")

columns = st.multiselect(
        "Choose columns", list(df.columns), ["last_oracle_price", "last_oracle_price_twap"]
    )


st.markdown('[https://github.com/drift-labs/protocol-v2](https://github.com/drift-labs/protocol-v2)')
fig = px.line(df[columns], title='Backtest of '+experiment+':'+trial)
st.plotly_chart(fig)