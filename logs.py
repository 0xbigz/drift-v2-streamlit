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

def get_account_txs(pk, limit=10, last_hash=None):
    if last_hash is None:
        resp = requests.get(f'https://public-api.solscan.io/account/transactions?account={pk}&limit={limit}')
    else: 
        resp = requests.get(f'https://public-api.solscan.io/account/transactions?account={pk}&limit={limit}&beforeHash={last_hash}')
    return resp

# 150 requests/ 30 seconds
def get_last_n_tx_sigs(program_id, limit):
    sigs = []
    last_hash = None
    n_requests = 0
    # sol scan only allows 50 at a time
    for i in range(0, limit, 50):
        amount = 50 if i+50 < limit else limit % 50 
        resp = get_account_txs(program_id, amount, last_hash)
        resp = json.loads(resp.text)
        sigs += [r['txHash'] for r in resp]
        if len(resp) == 0:
            return sigs, len(sigs)

        last_hash = sigs[-1]
        n_requests += 1
    
    return sigs, n_requests

def get_tx_request(sig):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            sig,
            "json"
        ]
    }

def batch_get_txs(sigs):
    data = [get_tx_request(sig) for sig in sigs]

    resps = []
    step = 100
    from tqdm import tqdm 
    for i in tqdm(range(0, len(data), step)):
        batch = data[i: i+step]
        print(f'requesting {len(batch)}')
        resp = requests.post(
            "https://drift-cranking.rpcpool.com", 
            headers={
                "Content-Type": "application/json",
                "origin": "https://app.drift.trade",
            }, 
            json=batch
        )
        try:
            resp = json.loads(resp.text)
        except Exception as e: 
            print(resp.text)
            raise e
        
        # expecting json list as response
        resps += resp
    return resp

# async def query_server(ch: ClearingHouse):
#     all_users = await ch.program.account['User'].all()
#     page_index = 0 
#     page_size = 1000

#     jsons = []
#     from tqdm import tqdm
#     for x in tqdm(all_users):
#         key = str(x.public_key)
#         query = f'https://mainnet-beta.api.drift.trade/liquidations/userAccount/?userPublicKey={key}&pageIndex={page_index}&pageSize={page_size}'
#         resp = requests.get(query)
#         resp = json.loads(resp.text)
#         if len(resp['data']['liquidations']) > 0:
#             print(resp)
#             jsons.append(resp)
#             break
#     st.json(jsons)

@cached(ttl=None, cache=Cache.MEMORY)
async def get_all(ch, limit):
    sigs, n_req = get_last_n_tx_sigs(ch.program_id, limit)
    st.write(f'n of solscan responses: {n_req}')
    if n_req == 0: 
        st.write('likely solscan rate limit...')
        return False, None

    txs = batch_get_txs(sigs)

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

    return True, (log_names, type_to_log, n_logs, (max_log, min_log))

async def log_page(url: str, ch):
    # log liquidations
    limit = st.number_input('tx look up limit', value=10)
    s, r = await get_all(ch, limit)
    if not s: 
        return
    log_names, type_to_log, n_logs, (max_ts, min_ts) = r 

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