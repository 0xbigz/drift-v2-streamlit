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



async def vaults(ch: ClearingHouse, env):

    tabs = st.tabs(['summary', 'vaults', 'vault depositors', 'settings'])
    idl = None
    pid = None
    with tabs[3]:
        g1, g2 = st.columns(2)
        pid = g1.text_input('program id:', 'VAULtLeTwwUxpwAw98E6XmgaDeQucKgV5UaiAuQ655D')
        default_url = 'https://raw.githubusercontent.com/drift-labs/drift-vaults/master/ts/sdk/src/idl/drift_vaults.json'
        url = g2.text_input('idl:', default_url)
        response = requests.get(url)
        
        st.write('https://solscan.io/account/'+pid)
        data = response.json()
        idl = data
        provider = ch.program.provider
        st.json(idl, expanded=False)

    program = Program(
            Idl.from_json(idl),
            PublicKey(pid),
            provider,
        )
    
    with tabs[3]:
        with st.expander('vault fields'):
            st.write(program.account.keys())
            st.write(program.type.keys())
        st.dataframe(pd.DataFrame(idl['instructions']))

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
            res = pd.concat(res)
            vault_df = res
            st.dataframe(res)

    
    with tabs[2]:
        res = []
        for x in all_vault_deps:
            dd = x.account.__dict__
            # dd['name'] = bytes(dd['name']).decode('utf-8').strip(' ')
            ser = pd.Series(dd, name=x.public_key)
            res.append(ser)
        if len(res):
            res = pd.concat(res)
            vault_dep_def = res
            st.dataframe(res) 
   
    with tabs[0]:
        s1, s2 = st.columns(2)
        s1.metric('number of vaults:', len(all_vaults), str(len(all_vault_deps))+' vault depositor(s)')
        chu = None
        if len(vault_df):
            vu = [PublicKey(x) for x in vault_df.loc[['user']].tolist()]
            # st.write(vu)
            vault_users = await ch.program.account["User"].fetch_multiple(vu)
            from driftpy.clearing_house_user import ClearingHouseUser
            fuser = vault_users[0]

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
                equity = (await chu.get_spot_market_asset_value())/1e6
                st.markdown(f"> [{nom}](https://app.drift.trade/?authority={vault_user.authority}): ${equity}")
                # st.write(vault_user)
                # st.write(f"fees: {vault_df.iloc[:,i]}%")# / {vault_df.iloc['profit_share',i]/1e4}% ")
        # st.write(all_vaults[0])