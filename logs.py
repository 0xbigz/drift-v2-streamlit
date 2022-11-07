
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

@dataclass
class Config:
    env: str
    pyth_oracle_mapping_address: PublicKey
    clearing_house_program_id: PublicKey
    usdc_mint_address: PublicKey
    markets: list[Market]
    banks: list[Bank]

configs = {
    "devnet": Config(
        env='devnet',
        pyth_oracle_mapping_address=PublicKey('BmA9Z6FjioHJPpjT39QazZyhDRUdZy2ezwx4GiDdE2u2'),
		clearing_house_program_id=PublicKey('dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH'),
		usdc_mint_address=PublicKey('8zGuJQqwhZafTah7Uc7Z4tXRnguqkn5KLFAP8oV6PHe2'),
		markets=devnet_markets,
		banks=devnet_banks,
    ),
    "mainnet-beta": Config(
        env='mainnet-beta',
        pyth_oracle_mapping_address=PublicKey('BmA9Z6FjioHJPpjT39QazZyhDRUdZy2ezwx4GiDdE2u2'),
		clearing_house_program_id=PublicKey('dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH'),
		usdc_mint_address=PublicKey('8zGuJQqwhZafTah7Uc7Z4tXRnguqkn5KLFAP8oV6PHe2'),
		markets=devnet_markets,
		banks=devnet_banks,
    )
}

def get_account_txs(pk, limit=10, last_hash=None):
    if last_hash is None:
        resp = requests.get(f'https://public-api.solscan.io/account/transactions?account={pk}&limit={limit}')
    else: 
        resp = requests.get(f'https://public-api.solscan.io/account/transactions?account={pk}&limit={limit}&beforeHash={last_hash}')
    return resp

import requests
from aiocache import Cache
from aiocache import cached

@cached(ttl=None, cache=Cache.MEMORY)
async def get_all(limit, endpoint):
    config = configs['mainnet-beta']
    kp = Keypair()
    wallet = Wallet(kp)
    connection = AsyncClient(endpoint)
    provider = Provider(connection, wallet)
    ch: ClearingHouse = ClearingHouse.from_config(config, provider)

    sigs = []
    last_hash = None
    for _ in range(0, limit, 50):
        resp = get_account_txs(ch.program_id, 50, last_hash)
        resp = json.loads(resp.text)
        sigs += [r['txHash'] for r in resp]
        last_hash = sigs[-1]

    promises = []
    for sig in sigs:
        promises.append(connection.get_transaction(sig))
    txs = await asyncio.gather(*promises)
    n_logs = len(txs)
    
    # txs = []
    # for sig in sigs:
    #     tx = await connection.get_transaction(sig)
    #     txs.append(tx)

    parser = EventParser(ch.program.program_id, ch.program.coder)
    logs = {}
    for tx, sig in zip(txs, sigs):
        def call_b(evt): logs[sig] = evt
        parser.parse_logs(tx['result']['meta']['logMessages'], call_b)
 
    log_names = list(set([log.name for log in logs.values()]))
    type_to_log = {}
    for log_name in log_names:
        for sig, log in logs.items(): 
            if log.name == log_name:
                if log_name not in type_to_log: 
                    type_to_log[log_name] = {}
                type_to_log[log_name][sig] = log.data

    return log_names, type_to_log, n_logs

async def view_logs(url):
    config = configs['mainnet-beta']

    kp = Keypair()
    wallet = Wallet(kp)
    connection = AsyncClient(url)
    provider = Provider(connection, wallet)
    ch: ClearingHouse = ClearingHouse.from_config(config, provider)

    with st.expander(f'pid={ch.program_id} config'):
        st.text(str(config))

    # log liquidations
    # ...
    limit = st.number_input('tx look up limit', value=10)
    log_names, type_to_log, n_logs = await get_all(limit, url)

    options = st.multiselect('Log types', log_names, log_names)
    st.write(f'### number of logs found: {n_logs}')

    view_logs = {}
    for name in options: 
        view_logs[name] = type_to_log[name]

    st.write(view_logs)