import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"

print(driftpy.__dir__())
# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
import os
import json
import streamlit as st
from driftpy.constants.banks import devnet_banks, Bank
from driftpy.constants.markets import devnet_markets, Market
from dataclasses import dataclass
from solana.publickey import PublicKey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio

import requests
from aiocache import Cache
from aiocache import cached
from config import configs

def get_account_txs(pk, limit=10, last_hash=None):
    if last_hash is None:
        resp = requests.get(f'https://public-api.solscan.io/account/transactions?account={pk}&limit={limit}')
    else: 
        resp = requests.get(f'https://public-api.solscan.io/account/transactions?account={pk}&limit={limit}&beforeHash={last_hash}')
    return resp

def get_last_n_tx_sigs(program_id, limit):
    sigs = []
    last_hash = None
    # sol scan only allows 50 at a time
    for i in range(0, limit, 50):
        amount = 50 if i < limit else limit % 50 
        resp = get_account_txs(program_id, amount, last_hash)
        resp = json.loads(resp.text)
        sigs += [r['txHash'] for r in resp]
        last_hash = sigs[-1]
    
    return sigs

@cached(ttl=None, cache=Cache.MEMORY)
async def get_all(ch, limit):
    connection = ch.program.provider.connection
    sigs = get_last_n_tx_sigs(ch.program_id, limit)

    promises = []
    for sig in sigs:
        promises.append(connection.get_transaction(sig))
    txs = await asyncio.gather(*promises)
    n_logs = len(txs)
    
    parser = EventParser(ch.program.program_id, ch.program.coder)
    
    logs = {}
    for tx, sig in zip(txs, sigs):
        def call_b(evt): 
            logs[sig] = logs.get(sig, []) + [evt]
        parser.parse_logs(tx['result']['meta']['logMessages'], call_b)

    log_times = list(set([event.data.ts for events in logs.values() for event in events]))
    max_log = max(log_times)
    min_log = min(log_times)

    log_names = list(set([event.name for events in logs.values() for event in events]))
    type_to_log = {}
    for log_name in log_names:
        for sig, events in logs.items(): 
            for event in events:
                if event.name == log_name:
                    if log_name not in type_to_log: 
                        type_to_log[log_name] = {}
                    type_to_log[log_name][sig] = type_to_log[log_name].get(sig, []) + [event.data]

    return log_names, type_to_log, n_logs, (max_log, min_log)

async def log_page(url: str, ch):
    # log liquidations
    limit = st.number_input('tx look up limit', value=10)
    log_names, type_to_log, n_logs, (max_ts, min_ts) = await get_all(ch, limit)
    
    import datetime
    min_ts = datetime.datetime.fromtimestamp(min_ts)
    max_ts = datetime.datetime.fromtimestamp(max_ts)
    st.write(f'min <-> max ts :: {min_ts} <-> {max_ts}')

    options = st.multiselect('Log types', log_names, log_names)
    st.write(f'### number of logs found: {n_logs}')

    view_logs = {}
    for name in options: 
        view_logs[name] = type_to_log[name]

    st.write(view_logs)