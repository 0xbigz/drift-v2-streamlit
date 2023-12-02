import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.drift_user import get_token_amount
from anchorpy.provider import Signature
import os
import json
import streamlit as st
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from driftpy.drift_user import get_token_amount
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import datetime

import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import InsuranceFundStakeAccount, SpotMarketAccount
from driftpy.addresses import * 
# using time module
import time
import plotly.express as px
from anchorpy import Program, Context, Idl, Wallet, Provider
from datafetch.transaction_fetch import transaction_history_for_account, load_token_balance


async def tab_openbookv2(ch, env):
    st.write('gm')

    tabs = st.tabs(['summary', 'accounts', 'transactions', 'settings'])
    idl = None
    pid = None
    preset_programs = None
    with tabs[3]:
        preset_programs = st.radio('presets:', ['openbookv2', 'metadaoproject', 'custom'],
                                   horizontal=True)
        g1, g2 = st.columns(2)
        
        if preset_programs == 'openbookv2':
            default_pid = 'opnb2LAfJYbRMAHHvqjCwQxanZn7ReEHp1k81EohpZb'
            default_url = 'https://raw.githubusercontent.com/openbook-dex/openbook-v2/master/idl/openbook_v2.json'
        elif preset_programs == 'metadaoproject':
            default_pid = 'meta3cxKzFBmWYgCVozmvCQAS3y9b3fGxrG9HkHL7Wi'
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
        data = response.text
        idl_raw: str = data
        idl = json.loads(idl_raw)
        provider = ch.program.provider
        # st.json(idl, expanded=False)

    program = Program(
            Idl.from_json(idl_raw),
            Pubkey.from_string(pid),
            provider,
        )
    
    
    with tabs[2]:
        s1, s2, s3 = st.columns(3)
        addysource = s1.text_input('address:', pid)
        filter_errs = s2.radio('filter errors:', [True, False])
        targetNum = s3.number_input('limit:', value=1000)
        txns = await transaction_history_for_account(provider.connection, Pubkey.from_string(addysource), None, None, targetNum)

        df = pd.DataFrame(txns)
        df['datetime'] = pd.to_datetime((df['blockTime'].astype(int)*1e9).astype(int))
        if filter_errs:
            df = df[df.err.astype(str)=='None']
        st.write(f'last {len(df)} transactions')

        st.dataframe(df)
        if len(df):
            parser = EventParser(program.program_id, program.coder)
            all_sigs = df['signature'].values.tolist()
            c1, c2 = st.columns([1,8])
            do_check = c1.radio('load tx:', [True, False], index=1, key='hihihi')
            signature = c2.selectbox('tx signatures:', all_sigs)
            # st.write(df[df.signature==signature])
            if do_check:
                theset = [signature]
                idxes = [i for i,x in enumerate(all_sigs) if x==signature]
                txs = []
                sigs = []
                # try:
                # from solders.rpc.responses import GetSignaturesForAddressResp, GetTokenAccountBalanceResp, Signature

                for idx in idxes:
                    sig = df['signature'].values[idx]
                    transaction_got = (await provider.connection.get_transaction(Signature.from_string(sig)))
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
                    tt = json.loads(tx.to_json())
                    st.write(tt['result']['meta']['logMessages'], expanded=True)
                    parser.parse_logs(tt['result']['meta']['logMessages'], call_b)
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
            tts[idx].json(([field]), expanded=False)
            tts[idx].dataframe(pd.DataFrame(idl[field]).astype(str))

    accounts = [(await program.account[act['name']].all()) for act in idl['accounts']]
    vault_df = pd.DataFrame()
    vault_dep_def = pd.DataFrame()
    with tabs[1]:
        for nom, acts in zip(idl['accounts'], accounts):
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
                for col in vault_df.columns:
                    vault_df[col] = vault_df[col].astype(str)

                st.write(nom['name'], res.shape)
                try:
                    if len(vault_df) < 3:
                        st.dataframe(vault_df.T)
                    else:
                        st.dataframe(vault_df)
                except Exception as e:
                    st.warning('cannot write: ' + str(e))