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
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.clearing_house_user import get_token_amount

import os
import json
import streamlit as st
from driftpy.constants.banks import devnet_banks, Bank
from driftpy.constants.markets import devnet_markets, Market
from driftpy.clearing_house_user import get_token_amount
from dataclasses import dataclass
from solana.publickey import PublicKey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import datetime

import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import InsuranceFundStake, SpotMarket
from driftpy.addresses import * 
# using time module
import time
import plotly.express as px
from anchorpy import Program, Context, Idl, Wallet, Provider
from datafetch.transaction_fetch import transaction_history_for_account



async def competitions(ch: ClearingHouse, env):

    tabs = st.tabs(['summary', 'accounts', 'transactions', 'settings'])
    idl = None
    pid = None
    preset_programs = None
    with tabs[3]:
        preset_programs = st.radio('presets:', ['drift-competitions', 'drift-vaults', 'metadaoproject', 'custom'],
                                   horizontal=True)
        g1, g2 = st.columns(2)
        
        if preset_programs == 'drift-competitions':
            #'HjMa8sytpmBvf1Qr6UAJxYMtTfc3Qw8Z2cHD3nY1w2Nq', '8yCtqd9UetSttHhbGJFRk3MpcQny3bNvEfv5YrADuHEp'
            default_pid = 'DraWMeQX9LfzQQSYoeBwHAgM5JcqFkgrX7GbTfjzVMVL'
            default_url = 'https://raw.githubusercontent.com/drift-labs/drift-competitions/master/ts/sdk/src/idl/drift_competitions.json'
        elif preset_programs == 'metadaoproject':
            default_pid = 'Ctt7cFZM6K7phtRo5NvjycpQju7X6QTSuqNen2ebXiuc'
            default_url = 'https://gist.githubusercontent.com/metaproph3t/eb73908720f3286a2c9baf9ce200b2e9/raw/1a0b736baf88a42e3ca935b05f1f12075c44fc4f/autocrat_idl.json'
        else:
            default_pid = ''
            default_url = ''
        pid = g1.text_input('program id:', 
                            default_pid,
        )
        url = g2.text_input('idl:', default_url)
        response = requests.get(url)
        
        st.write('https://solscan.io/account/'+pid)
        data = response.json()
        idl = data
        provider = ch.program.provider
        # st.json(idl, expanded=False)

    program = Program(
            Idl.from_json(idl),
            PublicKey(pid),
            provider,
        )
    
    with tabs[2]:
        txns = await transaction_history_for_account(provider.connection, pid, None, None, 1000)
        st.write(f'last {len(txns)} transactions')

        df = pd.DataFrame(txns)
        df['datetime'] = pd.to_datetime((df['blockTime'].astype(int)*1e9).astype(int))
        st.dataframe(df)
        if len(df):
            parser = EventParser(program.program_id, program.coder)
            all_sigs = df['signature'].values.tolist()
            c1, c2 = st.columns([1,8])
            do_check = c1.radio('load tx:', [True, False], index=1)
            signature = c2.selectbox('tx signatures:', all_sigs)
            # st.write(df[df.signature==signature])
            if do_check:
                theset = [signature]
                idxes = [i for i,x in enumerate(all_sigs) if x==signature]
                txs = []
                sigs = []
                # try:
                for idx in idxes:
                    sig = df['signature'].values[idx]
                    transaction_got = await provider.connection.get_transaction(sig)
                    txs.append(transaction_got)
                    sigs.append(sig)
                # except Exception as e:
                #     st.warning('rpc failed: '+str(e))

                # txs = [transaction_got]
                # sigs = all_sigs[idx]
                logs = {}
                for tx, sig in zip(txs, sigs):
                    def call_b(evt): 
                        logs[sig] = logs.get(sig, []) + [evt]
                    # likely rate limited
                    # if 'result' not in tx: 
                    #     st.write(tx['error'])
                    #     break 
                    st.write(tx['result']['meta']['logMessages'], expanded=True)
                    parser.parse_logs(tx['result']['meta']['logMessages'], call_b)
                    st.write(logs)
                    st.json(transaction_got, expanded=False)
            
        
        # st.write(transaction_got['result']['meta']['logMessages'])


    with tabs[3]:
        # with st.expander('vault fields'):
        #     st.write(program.account.keys())
        #     st.write(program.type.keys())
        fields = ['instructions', 'accounts', 'types', 'errors']
        tts = st.tabs(fields)
        for idx,field in enumerate(fields) :
            tts[idx].json((idl[field]), expanded=False)
            tts[idx].dataframe(pd.DataFrame(idl[field]).astype(str))

    accounts = [(await program.account[act['name']].all()) for act in idl['accounts']]
    vault_df = pd.DataFrame()
    vault_dep_def = pd.DataFrame()
    with tabs[1]:
        for acts in accounts:
            res = []
            for x in acts:
                dd = x.account.__dict__
                if 'name' in dd:
                   dd['name'] = bytes(dd['name']).decode('utf-8').strip(' ')
                ser = pd.Series(dd, name=x.public_key)
                res.append(ser)
            if len(res):
                res = pd.concat(res,axis=1)
                vault_df = res
                vault_df = vault_df.T#.sort_values(by='next_expiry_ts', ascending=False).T
                st.dataframe(res)
   
    with tabs[0]:
        s1, s2, s3 = st.columns(3)
        s1.metric('number of '+idl['accounts'][0]['name']+'(s):', len(accounts[0]), str(len(accounts[1]))+' '+idl['accounts'][1]['name']+'(s)')
        

        if preset_programs == 'drift-competitions':
            every_user_stats = await ch.program.account['UserStats'].fetch_multiple(vault_df['user_stats'].values)
            # mode = st.radio('mode:', ['current state', 'extrapolated'], horizontal=True)
            derived_snap_name = 'latest_snapshot(wb)'
            vault_df[derived_snap_name] = [x.fees.total_fee_paid//100 for x in every_user_stats]

            comp_acct = accounts[0][0].account
            vsum = vault_df.sum()
            st.write('round number:', comp_acct.round_number, ' | number of winners:', comp_acct.number_of_winners)
            prize_ev = s3.number_input('expected prize ($):', step=1.0, min_value=0.0, value=2000.0, max_value=1e9)
            vault_df['entries'] = vault_df[derived_snap_name] + vault_df['bonus_score'] - vault_df['previous_snapshot_score']
            vault_df['entries'] =  vault_df['entries'].apply(lambda x: min(x, comp_acct.max_entries_per_competitor))
            vault_df['EV ($)'] = vault_df['entries']/vault_df['entries'].sum() * prize_ev
            vault_df['chance of a win'] = 1 - ((1- vault_df['entries']/vault_df['entries'].sum()) ** 33)

            s2.metric('total entries:', f"{vault_df['entries'].sum():,.0f}",
                    f"{vsum['bonus_score']} from bonus_score")
            d1, d2 = st.columns([3,1])        
            
            vault_df1 = vault_df.reset_index().set_index('authority')
            d1.dataframe(vault_df1[['entries', 'EV ($)', 'chance of a win']].sort_values(by='entries', ascending=False), use_container_width=True)
