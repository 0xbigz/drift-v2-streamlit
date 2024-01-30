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
from driftpy.drift_client import DriftClient, AccountSubscriptionConfig
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.drift_user import get_token_amount, UserAccount

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



async def vaults(ch: DriftClient, env):

    tabs = st.tabs(['summary', 'vaults', 'vault depositors', 'settings'])
    idl = None
    pid = None
    with tabs[3]:
        g1, g2 = st.columns(2)
        pid = g1.text_input('program id:', 'vAuLTsyrvSfZRuRB3XgvkPwNGgYSs9YRYymVebLKoxR',
        #'VAULtLeTwwUxpwAw98E6XmgaDeQucKgV5UaiAuQ655D' # old one
        )
        default_url = 'https://raw.githubusercontent.com/drift-labs/drift-vaults/master/ts/sdk/src/idl/drift_vaults.json'
        url = g2.text_input('idl:', default_url)
        response = requests.get(url)
        
        st.write('https://solscan.io/account/'+pid)
        data = response.text
        idl_raw = data
        idl = json.loads(idl_raw)
        provider = ch.program.provider
        st.json(idl, expanded=False)

    program = Program(
            Idl.from_json(idl_raw),
            Pubkey.from_string(pid),
            provider,
        )
    
    with tabs[3]:
        # with st.expander('vault fields'):
        #     st.write(program.account.keys())
        #     st.write(program.type.keys())
        st.json((idl['instructions']), expanded=False)
        st.dataframe(pd.DataFrame(idl['instructions']).astype(str))

    all_vaults = await program.account["Vault"].all()
    all_vault_deps = await program.account["VaultDepositor"].all()
    vault_df = pd.DataFrame()
    vault_dep_def = pd.DataFrame()
    with tabs[1]:
        res = []
        for x in all_vaults:
            dd = x.account.__dict__
            dd['name'] = bytes(dd['name']).decode('utf-8').strip(' ')
            ser = pd.Series(dd, name=x.public_key)
            res.append(ser)
        if len(res):
            res = pd.concat(res,axis=1)
            vault_df = res
            vault_df = vault_df.T.sort_values(by='init_ts', ascending=False).T
            st.dataframe(res)

    
    with tabs[2]:
        res = []
        for x in all_vault_deps:
            dd = x.account.__dict__
            # dd['name'] = bytes(dd['name']).decode('utf-8').strip(' ')
            ser = pd.Series(dd, name=x.public_key)
            res.append(ser)
        if len(res):
            res = pd.concat(res, axis=1)
            vault_dep_def = res
            vault_dep_def.loc['authority'] = vault_dep_def.loc['authority'].astype(str)
            st.dataframe(res) 
   
    with tabs[0]:
        s1, s2 = st.columns(2)
        s1.metric('number of vaults:', len(all_vaults), str(len(all_vault_deps))+' vault depositor(s)')
        chu = None
        if len(vault_df):
            user_keys = vault_df.astype(str).T.sort_values(by='init_ts', ascending=False).T.loc['user'].values
            vu = [Pubkey.from_string(x) for x in user_keys]
            # st.write(vu)
            vault_users = await ch.program.account["User"].fetch_multiple(vu)
            from driftpy.drift_user import DriftUser
            fuser: UserAccount = vault_users[0]
            # st.write(vault_users)
            # st.write(fuser)

            chu = DriftUser(
                ch, 
                user_public_key=Pubkey.from_string(user_keys[0]),
                # authority=fuser., 
                # sub_account_id=fuser.sub_account_id, 
                # use_cache=True
                account_subscription=AccountSubscriptionConfig('cached')
            )
            await chu.account_subscriber.update_cache()
            await chu.drift_client.account_subscriber.update_cache()
            cache = chu.drift_client.account_subscriber.cache

            for i, vault_user in enumerate(vault_users):
                chu = DriftUser(
                    ch, 
                    user_public_key=Pubkey.from_string(user_keys[i]), 
                    # sub_account_id=vault_user.sub_account_id, 
                    account_subscription=AccountSubscriptionConfig('cached')
                    # use_cache=False
                )
                await chu.account_subscriber.update_cache()
                # chu.drift_client.account_subscriber.cache = cache

                nom = bytes(vault_user.name).decode('utf-8').strip(' ')

                dat = vault_df.iloc[:,i]

                max_tokens = 0
                if float(dat.loc['spot_market_index']) == 0:
                    max_tokens = float(dat.loc['max_tokens'])/1e6

                equity = (chu.get_spot_market_asset_value())/1e6
                if equity is None:
                    continue
                # st.write(nom, equity, max_tokens)
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