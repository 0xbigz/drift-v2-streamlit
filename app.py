from math import exp
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import json
from urllib.request import urlopen

st.set_page_config(
    'Drift v2 Simulations',
    layout='wide',

)

root = "https://raw.githubusercontent.com/drift-labs/drift-sim/master/backtest/"

st.title('Drift v2 Simulations')
st.markdown('[https://github.com/drift-labs/drift-sim](https://github.com/drift-labs/drift-sim) | [@driftprotocol](https://twitter.com/@driftprotocol)')

st.subheader('Step 1 : simulate')
st.text('simulate agent interactions based on oracle price action')

experiments = ['lunaCrash']
experiment = st.selectbox(
        "Choose experiment", list(experiments), 
    )

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

st.subheader('Step 2: backtest')
st.text('runs events against drift smart contract (each iteration is a trial)')
st.markdown('[https://github.com/drift-labs/protocol-v2](https://github.com/drift-labs/protocol-v2)')

trials = ['trial_no_oracle_guards', 'trial_spread_250', 'trial_spread_1000', 'trial_oracle_guards']
trial = st.selectbox(
        "Choose trial", list(trials), 
    )

st.text('')
st.text('')
st.text('')
st.text('')
st.text('')
st.text('')

st.subheader('Step 3 : explore')
st.text('explore the data to find anomalies and make sure invariants are in check')
st.markdown("- do funds in vault after everyone enters and exits add up?")
st.markdown("- was the cumulative social loss reasonable?")


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


    with st.expander(ff+" more plots"): 

        from plotly.subplots import make_subplots
        # plotly setup
        plot_rows=len(df.columns)//3
        plot_cols=3
        fig = make_subplots(rows=plot_rows, cols=plot_cols, shared_xaxes=True, subplot_titles=df.columns)

        # add traces
        x = 0
        for i in range(1, plot_rows + 1):
            for j in range(1, plot_cols + 1):
                if not(df.columns[x] in ['market.pubkey', 'market.padding', 'market.name', 'market.amm.oracle'
                
                'spot_market.mint','spot_market.pubkey', 'spot_market.value', 'spot_market.mint'
                ]):
                    #print(str(i)+ ', ' + str(j))
                    fig.add_trace(go.Scatter(x=df.index, y=df[df.columns[x]].values,
                                            name = df.columns[x],
                                            mode = 'lines'),
                                row=i,
                                col=j)

                x=x+1

        # Format and show fig
        fig.update_layout(height=1200, width=1200)
        fig.update_annotations(font_size=8)

        st.plotly_chart(fig)



        if 'perp' in ff:
            df['market.amm.long_cost_basis'] = -df['market.amm.quote_asset_amount_long']/df['market.amm.base_asset_amount_long']
            df['market.amm.short_cost_basis'] = -df['market.amm.quote_asset_amount_short']/df['market.amm.base_asset_amount_short']

            fig = px.line(
                df[['market.amm.'+x for x in 
                ['base_asset_amount_long', 'base_asset_amount_short', 'long_cost_basis', 'short_cost_basis']
                ]], 
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



            fig = px.line(
                df[['market.amm.'+x for x in 
                ['base_spread', 'long_spread', 'short_spread', 'max_spread']
                ]], 
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


            mark_price = df['market.amm.quote_asset_reserve']/df['market.amm.base_asset_reserve'] * df['market.amm.peg_multiplier']
            fig = px.line(
                pd.concat([df['market.amm.peg_multiplier'], mark_price],axis=1),
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