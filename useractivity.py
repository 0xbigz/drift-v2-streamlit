

import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
from driftpy.math.oracle import *
import datetime

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.clearing_house_user import get_token_amount
import os
import json
import streamlit as st
from driftpy.types import MarginRequirementType
from driftpy.constants.banks import devnet_banks, Bank
from driftpy.constants.markets import devnet_markets, Market
from driftpy.addresses import *
from dataclasses import dataclass
from solana.publickey import PublicKey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import time
from enum import Enum
from driftpy.math.margin import MarginCategory, calculate_asset_weight
import plotly.graph_objs as go


async def show_user_activity(clearing_house: ClearingHouse):
    # connection = clearing_house.program.provider.connection
    addycol, rpc_overcol, limcol, pagecol = st.columns([4,2,1,1])
    rpc_override = rpc_overcol.text_input('rpc override:', 'https://api.mainnet-beta.solana.com')
    connection = AsyncClient(rpc_override)

    addy = addycol.text_input('userAccount:', value='CJfc9nPHgrZohPWEsBnvYJeTs3LxBA5j8ZoZuZPugSQb')
    limit = limcol.number_input('limit:', 0, 1000, 0)
    before_sig = pagecol.text_input('before sig:', value="")
    before_sig1 = None
    if before_sig != '':
        before_sig1 = str(before_sig)

    res2 = []
    first_try = True
    
    tabs = st.tabs(['heatmap', 'dataframe', 'transaction details'])

    while (first_try and len(res2) % 1000 == 0) and len(res2)< 1000 * 20:
        try:
            if len(res2):
                bbs = res2[-1]['signature']
            else:
                bbs = before_sig1
            res = await connection.get_signatures_for_address(addy, before=bbs, limit=limit)
            if 'result' not in res:
                st.warning('bad get_signatures_for_address' + str(res))
                first_try = False
            else:
                res2.extend(res['result'])
        except Exception as e:
            st.warning('exception:'+str(e))
            first_try = False

    t = pd.DataFrame(res2)
    # st.write(t)

    t['date'] = pd.to_datetime(t['blockTime']*1e9)
    t['day'] = t['date'].apply(lambda x: x.date())
    
    with tabs[1]:
        st.dataframe(t)
    
    with tabs[2]:
        parser = EventParser(clearing_house.program.program_id, clearing_house.program.coder)
        all_sigs = t['signature'].values.tolist()
        c1, c2 = st.columns([1,8])
        ra = c1.radio('run all:', [True, False], index=1)
        signature = c2.selectbox('tx signatures:', all_sigs)
        st.write(t[t.signature==signature])

        theset = all_sigs if ra else [signature]
        idx = 0
        txs = []
        sigs = []
        # try:
        while idx < len(theset):
            sig = t['signature'].values[idx]
            ff = 'transactions/'+sig+'.json'

            if not os.path.exists(ff) or not ra:
                transaction_got = await connection.get_transaction(sig)

                if ra:
                    os.makedirs('transactions', exist_ok=True)
                    with open(ff, 'w') as f:
                        json.dump(transaction_got, f)
            else:
                with open(ff, 'r') as f:
                    transaction_got = json.load(f)
            txs.append(transaction_got)
            sigs.append(sig)
            
            st.json(transaction_got, expanded=False)
            idx+=1
        # except Exception as e:
        #     st.warning('rpc failed: '+str(e))

        # txs = [transaction_got]
        # sigs = all_sigs[:idx]
        logs = {}
        for tx, sig in zip(txs, sigs):
            def call_b(evt): 
                logs[sig] = logs.get(sig, []) + [evt]
            # likely rate limited
            if 'result' not in tx: 
                st.write(tx['error'])
                break 
            parser.parse_logs(tx['result']['meta']['logMessages'], call_b)
        st.write(logs)
        # st.write(transaction_got['result']['meta']['logMessages'])

    tgrp = t.groupby('day').count()['blockTime']
    weeks = pd.to_datetime(tgrp.reset_index().iloc[:,0]).dt.to_period('W-SUN').dt.start_time #tgrp.reset_index().iloc[:,0].apply(lambda x: (datetime.datetime(x.isocalendar()[1]))
    dow = pd.to_datetime(tgrp.reset_index().iloc[:,0]).apply(lambda x: (x.dayofweek))

    with tabs[0]:
        dates = tgrp.reset_index().iloc[:,0].values.tolist()
        fig = go.Figure(data=go.Heatmap(
            z=tgrp,
            x=weeks,
            y=dow,
            text=['transactions on '+str(x) for x in dates],
            hoverinfo='z+text',
            colorscale='greens'))
        layout = go.Layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    grid=None,
                    xaxis_showgrid=False, yaxis_showgrid=False
                )
        fig.update_layout(layout)
        fig['layout']['yaxis']['autorange'] = "reversed"
        num_tx = int(sum(tgrp.tolist()))
        first_day = dates[0]
        last_day = dates[-1]
        st.write(str(num_tx)+' transactions from '+str(first_day) + ' to ' + str(last_day))
        st.plotly_chart(fig, use_container_width=True)
