import sys
import driftpy
import pandas as pd 
import numpy as np 

pd.options.plotting.backend = "plotly"

from driftpy.constants.numeric_constants import * 
import json
import streamlit as st
from anchorpy import EventParser

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
        if n_requests > 148:
            st.write('reached max sol scan requests...')
            break
    
    return sigs, n_requests

def get_tx_request(sig):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            sig,
            {"encoding": "json", "maxSupportedTransactionVersion": 0}
        ]
    }

def batch_get_txs(url, sigs):
    data = [get_tx_request(sig) for sig in sigs]

    resps = []
    step = 100
    from tqdm import tqdm 
    for i in tqdm(range(0, len(data), step)):
        batch = data[i: i+step]
        resp = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
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
    return resps

@cached(ttl=None, cache=Cache.MEMORY)
async def get_all(url, ch, limit):
    sigs, n_req = get_last_n_tx_sigs(ch.program_id, limit)
    st.write(f'n of solscan responses: {n_req}')
    if n_req == 0: 
        st.write('likely solscan rate limit...')
        return False, None

    txs = batch_get_txs(url, sigs)
    n_logs = len(txs)
    parser = EventParser(ch.program.program_id, ch.program.coder)
    
    logs = {}
    for tx, sig in zip(txs, sigs):
        def call_b(evt): 
            logs[sig] = logs.get(sig, []) + [evt]
        # likely rate limited
        if 'result' not in tx: 
            st.write(tx['error'])
            break 
        parser.parse_logs(tx['result']['meta']['logMessages'], call_b)
    
    if len(logs) == 0: 
        st.write('likely rpc rate limited...')
        return False, None

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

from driftpy.drift_client import DriftClient
async def log_page(url: str, ch: DriftClient):
    # log liquidations
    st.write("getting recent txs from [https://public-api.solscan.io/docs/#/](https://public-api.solscan.io/docs/#/)...")
    limit = st.number_input('tx look up limit', value=0)
    if limit > 0:
        s, r = await get_all(url, ch, limit)
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

    parser = EventParser(ch.program.program_id, ch.program.coder)
    connection = ch.program.provider.connection
    tx_sig = st.text_input("lookup tx sig:")
    if tx_sig != '':
        # tx = await connection.get_transaction(tx_sig)
        tx = batch_get_txs(url, [tx_sig])[0]
        logs = []
        def call_b(evt): logs.append(evt)
        parser.parse_logs(tx['result']['meta']['logMessages'], call_b)
        st.json(logs)
