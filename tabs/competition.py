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
    with tabs[3]:
        preset_programs = st.radio('presets:', ['drift-competitions', 'drift-vaults', 'metadaoproject', 'custom'],
                                   horizontal=True)
        g1, g2 = st.columns(2)
        
        if preset_programs == 'drift-competitions':
            default_pid = 'HjMa8sytpmBvf1Qr6UAJxYMtTfc3Qw8Z2cHD3nY1w2Nq'
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
        s1, s2 = st.columns(2)
        s1.metric('number of '+idl['accounts'][0]['name']+'(s):', len(accounts[0]), str(len(accounts[1]))+' '+idl['accounts'][1]['name']+'(s)')
        chu = None
        return 0
        if len(vault_df):
            st.write(vault_df.T)
            vu = [PublicKey(x) for x in vault_df.T.loc['competition_authority'].values]
            # st.write(vu)
            vault_users = await ch.program.account["User"].fetch_multiple(vu)
            from driftpy.clearing_house_user import ClearingHouseUser
            fuser = vault_users[0]
            # st.write(vault_users)
            # st.write(fuser)
            if fuser is not None:
                chu = ClearingHouseUser(
                    ch, 
                    authority=fuser.authority, 
                    subaccount_id=fuser.sub_account_id, 
                    use_cache=True
                )
                await chu.set_cache()
                cache = chu.CACHE

                for i, vault_user in enumerate(vault_users):
                    chu = ClearingHouseUser(
                        ch, 
                        authority=vault_user.authority, 
                        subaccount_id=vault_user.sub_account_id, 
                        use_cache=False
                    )
                    chu.CACHE = cache

                    nom = bytes(vault_user.name).decode('utf-8').strip(' ')

                    dat = vault_df.iloc[:,i]

                    max_tokens = None
                    if float(dat.loc['spot_market_index']) == 0:
                        max_tokens = float(dat.loc['max_tokens'])/1e6

                    equity = (await chu.get_spot_market_asset_value())/1e6
                    with st.expander(f"> {nom}: ${equity:,.2f}/{max_tokens:,.2f}"):
                        st.markdown(f'balance: `${equity:,.2f}` [[web ui](https://app.drift.trade/?authority={vault_user.authority})]')

                        prof_share = float(dat.loc['profit_share'])/1e4
                        mang_share = float(dat.loc['management_fee'])/1e4
                        st.write(f'fee structure: `{mang_share:,.2f}% / {prof_share:,.2f}%` (management / profit fee)')
                        if float(dat.loc['total_shares']) != 0:
                            user_owned_frac = float(dat.loc['user_shares'])/float(dat.loc['total_shares'])
                            st.write(f'user-owned: `${equity*user_owned_frac:,.2f}`')


                            this_vault_deps = vault_dep_def.T[vault_dep_def.T['vault'].astype(str)==str(vault_user.authority)]\
                                .sort_values('vault_shares', ascending=False).T
                            this_vault_deps.index = this_vault_deps.index.astype(str)
                            # st.write(this_vault_deps)
                            for j in range(len(this_vault_deps.columns)):
                                tv_dep = this_vault_deps.iloc[:,j]
                                # st.write(tv_dep)
                                tv_nom = tv_dep.loc['authority'][:5]+'...'

                                tuser_owned_frac = float(tv_dep.loc['vault_shares'])/float(dat.loc['total_shares'])
                                tcur_val = equity*tuser_owned_frac
                                tpnl = tcur_val - float(tv_dep.loc['net_deposits'])/1e6
                                tprof_fees = float(tv_dep.loc['cumulative_profit_share_amount'])/1e6 * prof_share/1e2
                                st.write(f'> {tv_nom}-owned: `${tcur_val:,.3f}` | pnl = `${tpnl:,.3f}` | u.b. profit fees: `${tprof_fees:,.3f}`')

                            st.write(f'manager-owned: `${equity - equity*user_owned_frac:,.2f}` ')

                            dd = float(dat.loc['manager_total_profit_share'])/1e6
                            dd2 = float(dat.loc['manager_total_fee'])/1e6

                            mdep = float(dat.loc['manager_total_deposits'])/1e6
                            mwit = float(dat.loc['manager_total_withdraws'])/1e6

                            st.write(f'> manager net: `${mdep} - ${mwit}`')
                            st.write(f'> manager gains: `${dd2} + ${dd}`')
                    # st.write(vault_user)
                    # st.write(f"fees: {vault_df.iloc[:,i]}%")# / {vault_df.iloc['profit_share',i]/1e4}% ")
        # st.write(all_vaults[0])