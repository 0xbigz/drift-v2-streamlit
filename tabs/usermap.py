import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
import copy
import plotly.express as px
pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet, AccountClient
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import MemcmpOpts
from driftpy.drift_client import DriftClient
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.drift_user import DriftUser, get_token_amount
from datafetch.transaction_fetch import transaction_history_for_account, load_token_balance

from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.constants.spot_markets import mainnet_spot_market_configs

import os
import json
import streamlit as st
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from dataclasses import dataclass
from solders.pubkey import Pubkey
from helpers import serialize_perp_market_2, serialize_spot_market, all_user_stats
from anchorpy import EventParser
import asyncio
from driftpy.math.margin import MarginCategory
import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import InsuranceFundStakeAccount, SpotMarketAccount
from driftpy.addresses import * 
import time
from driftpy.user_map.user_map import UserMap, UserMapConfig, PollingConfig
import datetime
# @st.cache_resource
async def load_user_map(_drift_client, user_map_settings):
    start_time = time.time()
    st.write("cached usermap at:", int(start_time), datetime.datetime.fromtimestamp(start_time))
    include_idle = True if user_map_settings in ['all', 'idle']  else False
    user_map = UserMap(UserMapConfig(_drift_client, PollingConfig(0), skip_initial_load=False, include_idle=include_idle))
    if user_map_settings == 'all':
        await user_map.subscribe()
    elif user_map_settings == 'active':
        await user_map.subscribe()
    elif user_map_settings == 'idle':
        await user_map.subscribe()
    elif user_map_settings == 'whales':
        # over ~100k in value
        whales = '''BRksHqLiq2gvQw1XxsZq6DXZjD3GB5a9J63tUBgd6QS9
7tDm5mxdUcqW423mX9a3yC9MkPQccaTYuPmAxZBkGNxn
ETsaTf7tsaYEChnLfu2iwzFXWxJbCzwD6bPkMGCyB42d
9e3dJLadqDVdunTLbQwW5rPyq2284pXJ6aWQtKD6DZHJ
7SeykJkVT24ZkuNwyYNACWj5JaLdNRjxVeBD5LAJbMmB
2aMcirYcF9W8aTFem6qe8QtvfQ22SLY6KUe6yUQbqfHk
4oTeSjNig62yD4KCehU4jkpNVYowLfaTie6LTtGbmefX
8RLb4ys91TjY3EqWHEVEbNnzzSbQGsdwn6Dj3dtY8Z67
3KdSdkCjSPvpthQScANfNRkSfX8xpqcngTTUnnZJoxUm
8LsnV5g8xtKcjiwyBBUNZonyaiC91GcbX2sXazVefVyX
2oJLLjUnoq6dvwdQKBWAoY2cUJKCEgMBs49HxJiRKwDY
2Crr5X53x36xcBUpEV52wT1Xbnhknwaz2XVjeTXTJEUY
BsZ7DBAqNiTC9PN5St4cb8aUoaCzGB23NRq6Xcv2CfUW
DQmcci1JpEPHkXYoAn3kKkDB7ZNpEQjZPUbQd97z1aXM
73ZwKGCsaMitEcTQV4W5tBnyht9XayUmShA9xbzG8EUG
DiDsuvdxGUQdvLBToYN4Bmk2GfFSR2bFFMRu7GhmMkJG
HmrXYdAiraCKSjyuTuzX24Jb6QDNj36oTD1BwWb369Gd
EHQ2YfkMQ9UTV3yCLTuMXqWNfoLSuwBnrwNZ2iUPGHcF
HpFLzDwaKqfHWZGGHgUJfVRB3D2RF4EsY9NKaFXcTxRr
GPmMKkE3cuvoDmQmyPdqqoxhWXeZZAcbJ726r1zbBEkE
8us4LosdD7Ct2jQEHFsfZdF3Uaj9mtwYbBwpxeWvcaGp
EU7uEtXojSPz1FtmaJnUrLJkx9RTwayS3TyfveBZ5qDC
HbwQHQ52NZZTuSgpWdSrm3zHKFB4VKSrufWVH5bV7b8C
BsVi8TcSsfdZuoksw1enwrDTYEVRnPBze4ZSPSqmafqh
F1vmkcTwgC7AVSQkVMpRhjws1Ni3gmz4Cu55Xe1TWXvA
G4iis9dLtpKwFkERDUBUqjbWPY4uJwYkGnWcZQUa1mpv
3hH27WiZV2QmM6bK5Rz7bkrqHK2PX7wNo4EogPzmGadN
56BnDPDBv8yhJdsVjkuH53QjvjiMA4qURKuDy7FYWX2X
AM1zueGrHg7zQDYUfVWtzFgn2G6EAuX161KMrkSgtkkh
22n9s2gQQN5T5t1i8RrHpKju5RkiJsw2LrfmgBsdPhmn
2EUQ6kxy2cM558ZJN4D31x6RGgLwgK1eo9BAEFjochVT
FgbsjcwKyUYLKLoYyq3piHW3xfU7cemPS8GniM1eeoUs
FeCV4X2uSehvYRSUHU7eVqtiJdnFJ2fGShC98CNW8gwG
Daz7CYfAym86pf6MmzsfZFz22wuLNLqECu1wx7rKLxsn
Er5x5oryNKNwvCgPBz2481m82NutLWMHivHBRGJzV9uv
5z3wiSDCPnkxywMj1VJ64ZbvcVPoUK8ydCp74rkeqzZ8
Lshvy96HmpRUyUhx1KP5F1xL2LvJerEpbHSh2xqAmg1
4EUfWFFpxckhBiCRN35wLqQFH6XeT4sRy5NkBF3249Fu
4b7qSeHYDqN3byBgbtsQNVfNcJ7x2P9BLFNk9yQfQqLq
2Mc7TScL3xN4iRRFRpaGJP3zy1Zaq2ZRV21epuobQMHg
EEBjRSo6kNeQXpRYmxuuBcBNB4MqxmRsQaarNUJTHBL5
GC3hdnsPZVkVsXCceF5HyTWxu7ZD7sj7CMiVyPW7ew9y
HvSYehyjXAcL9Dq8DW9sYaUieRWpz5wozB3U5ajCRvqc
4rv4bGgCuKpakw2BTCasnkzSgqi7QFGmWEEkGEw9modm
C13FZykQfLXKuMAMh2iuG7JxhQqd8otujNRAgVETU6id
74obHhSMzZLyyWpbz9SfbrFtKGmmTmAfp13FEV26PDgB
DUfFQC9uXXmSc7amJPr7pwf75LV2oNtSkW5wV669b1LQ
4yN82JynHtd57JB63CepKmCs7nvsiqk4KShjPd9xtpDU
FAmFxKSzAdtDkb8eoHNyGf9Qrk7qHGo6BAZujRe5r1fc
3ZHrooUPNAv7NuajRW4N95t6Pm5Xeyps15q9oeSPt5Q4
3gvf9XBYcs453b1UXDnaj3CgNcz5p9dT9aiYDhxBePiC
7WrK1Ku9qXv2Lpum7osN7n7vrdDPdRorM4LYVDZ2Qsts
8hiFmuXjwWpmUpSJfnZb9jfSCx8gPfqwQUVcNV8WmS8K
46BbA4BYke2wYZsqRXiyXcz67LtSBR911KXoVTuRrDg2
8LNHw4qZGmXAKMJ6qL3PS45yCpJQre49oCzrVVCc76sW
74tj4HwRvWY3yHc7PNRzAuDss4z9udeSygKdfB9HSBtY
EaETWptCSuVjLeMe2c2exYt4CzHLruUZQF3LXTgR87de
G8ipCTW4dn1KwYyMBuTaXWp5stYbtrjYeLA6JxrAgbQe
DwtySoEwL2QSArfmUzVXaVFXW8ZLCHEDuXQCpqpq5V3j
7cBRvpTLiLi6cp3bLaZ8u9AerjrwsqPo37H2jKChnE3c
AMz5hbWe2B2ohM3LhVk5eRRtSYVUgANdUKR3M5Xb4t7s
EcY7jYAZAdgRzM8wBsR2qP3faL25w7C3Uy18HtzWdfZQ
EZe9hRFsUuSFD9kVeAEa8cmKZHsnBvKdGLiPspHBFrB
AQ66FNXphVFfTR8mFmv1PmCgkQzRKUP3Giw4bRYdHERe
6xjDCa6fpehWBQzEdDQW3Gv1kum3VzVKzZpXG4AKckJK
FjnB7KNmjGjYg2vRAQymnePW9wMuaTcP4EiinRcNdZtr
Eh87smZiFo797bjFc5RreRGrmKxvBs2vQM71oNfSxnfA
Gc2wTwfKgr17VzaEaGKHQFGDfqWo3miAshHzNhHRMGPJ
HFk6YDhctF45YcxP4U3JbEMVt4YkmhGz7mNU9nyXMpLe
BdehTirPRjv7iAakdzL3vD41VMYrizaEYYCSA2gVnWv
6X4aNRKmxeh6BLKhzGm8RedaspAigHZJPMAaM6nJdcJu
BoaYJas68C483JJjUaD31M3c1bkuRosFWpZrBfoALnyv
D99njKWrHAJXpkjKxLYLAeYU4z3RuQh6hWGBiCHhxi7i
J7tpzrQHfWhW69Q6GBzW778ProJ3hRLyvaHMmGqsB5ZQ
g1PMk3xmxdb631cFkrCzJuGxehRBqs6P3o7nGAvhnNU
AGFbBosmenBH9fm9SkKcz8m6borRbKKJxUfQnGsHDcsn
HVMMKeksGXZih5aQpv1KYHCBhaGCRuXqE3bkQCURYehU
HnmRSitzQk249TVjTxVmWakNUrdMfEnG6TBpv7V9pQWB
4ywReyHX6hEr3GyP4d1AXpkam8HmFYZeKz8f3xpbqYSN
A5VNoX7JDuzi6BTRkAvEBf2dBiFK67yqbQC7ChUatExj
23h8frAqQwRiRRHF2mE8H6vztEZJ6DTYfJfRxr4d64HV
DjN8mCngZHnhDtKYwFjgQk4cjr6oUanE1GCS9AmBwTcq
5J4wTB2dfBLSJJa5g988N4LM8TNhV71EjjifWZej2oj6
8HACL7BXeJBS1aYYUizQFoxwYBVEiDMxD8QsrNkZwDx4
85QLSQAKhWYcKFQJ8xxfi4gkb82Cx6ito8CXLPCTGLn4
7XX43NrjTHTNPd5LfNmHKKLPhVSjgAV4FKRdXGDUVwgh
7GguEzb2ZgeH1TFukx9VdBKJdpRPsRom56WgFYK6zD9e
A3h5TLA7ij3VnsZJmt2g2piVXDenuMskXLUQ7LgcQZTi
GZSciBrAKKhHMZi6yphPKtvhoTssmKJUaCxxDTaKX8md
4X8f5P9V4Jvzr4sU2H6rTKYjJeDFEX2nxbcB3vmG2q2u
DFUXe5EHxAQLoLiUquuH75rFf5h5kbfK8s2PDrKVs6cK
73taguWJkNYo5mTfA52dxgwoTLN8PP6AFyCVHbSSfycg
B3WuWwm46HAaBT1xRHh8ymoiVrftJ3vuDYVaK5vqoX57
ApdcetQhQqrwvwC6BisX7y1iJJLbhtziPbXwko1KrpTQ
3Q89n76Lx7K6yyVagH4BTRfPJ8BLb5K8wB6LwvDK72WW
AkmPdA5DQsJgLsTiskRyeGbYzGxJ9GxAE6AA2vp9wmtx
9CNuKaJ7zzDvUHQpGmjB4akDYBFv6HXqhmxAQYx9Qw7w
6t4DUWiaewuxfuCo7RpAyw3avcPWp8zvpeogpBfuLFEL
ENZihoaRFsY1ZHzWibR3uMMVhMT2xAcm1JaAgoLxGSte
Bi9U9KvmgQK9kLww3N3UJvPy5A1mJnRqyzm4BXDePHvF
BXP1a8cCBdGNrpTjfncqeSxy8hhX6Szs4P76MF8eDQUU
F4ypfFqdWvR4zFCsbBLv6j29huFjpgRmmCKkVN9M8khT
FmohRubtmmrwWe8xJ3R6WTp5BYyVv6do9V2tfnFijUtg
9xnPxwedXTbwYeDegP4kXQ6LWyNKCwhaJuuF7ST7bxRi
HnH5gaa84M7phQFwNMp2PHVQqdeVCoXKFGu1nyTwKpuB
E3M3VWPsiUp89uxzLw4Cg1hZMbqk5PfHqRQqQ6qj4dX
7WBa9Du2N6KJTNnQdpWG4djRKzkZZcCan6ksEPaX8Vpo
9fwhHftD1KM3C5QQ3TJZBD13MVL4r9s3BsuR7rzbVZcu
Fgfdx9QFvwQwZ4QdUGPUBES7YR5S6itZPVJSUYKh5VEE
HPVVmzTKkwMWbpNyT9PQoeowBEWaXca2g9NYUawYuBQ6
2AkGKesooNq1Hn39xMsZnJkjQaNuqyaXaHfmPHgmpmPL
481L7qoCfkWqutamEcj7AYDS6P3azawF5WhshWRAPVdi
fBcykPptSKYJTCRZ8JTSN53gGo5HyQ9SaJXSpfRMERR
2nBqswD2KW4Vh63ZbFmtdRHT4n649bvS8n2XAueToi3P
59XziNjAka8dh9yDnquThAFu5jdHbhbyanx6h14ZpZTj
DBKiRUXMctVodDxNyQFrr2JJ1rZF6kr4muQDSKhMpBqu
6oSJJuGZSz1UgeJDMMHGMz81NbbXWsBn4P5Jrn3YQJo4'''
        for x in whales.split('\n'):
            if x.strip(' '):
                await user_map.must_get(x.strip(' '))
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write("loaded usermap in", int(elapsed_time), ' seconds')
    return user_map

def comb_asset_liab(a_l_tup):
    return a_l_tup[0] - a_l_tup[1]

def get_collateral_composition(x: DriftUser, margin_category, n):
    # ua = x.get_user_account()
    net_v = {i: comb_asset_liab(x.get_spot_market_asset_and_liability_value(i, margin_category))/QUOTE_PRECISION for i in range(n)}
    return net_v 

def get_perp_liab_composition(x: DriftUser, margin_category, n):
    # ua = x.get_user_account()
    net_p = {i: x.get_perp_market_liability(i, margin_category, signed=True)/QUOTE_PRECISION for i in range(n)}
    return net_p 

async def get_usermap_df(_drift_client, user_map_settings, mode, oracle_distor=.1, only_one_index=None, cov_matrix=None):
    def do_dict(x: DriftUser, margin_category: MarginCategory, oracle_cache=None):
        if oracle_cache is not None:
            x.drift_client.account_subscriber.cache = oracle_cache
        levs0 = {
        'leverage': x.get_leverage() / MARGIN_PRECISION, 
        'perp_liability': x.get_perp_market_liability(None, margin_category) / QUOTE_PRECISION,
        'spot_asset': x.get_spot_market_asset_value(None, margin_category) / QUOTE_PRECISION,
        'spot_liability': x.get_spot_market_liability(None, margin_category) / QUOTE_PRECISION,
        'upnl': x.get_unrealized_pnl(True) / QUOTE_PRECISION,
        'funding_upnl': x.get_unrealized_funding_pnl() / QUOTE_PRECISION,
        'net_v': get_collateral_composition(x, margin_category, spot_n),
        'net_p': get_perp_liab_composition(x, margin_category, perp_n),
        }
        levs0['net_usd_value'] = levs0['spot_asset'] + levs0['upnl'] - levs0['spot_liability']
        return levs0
    perp_n = 22
    spot_n = 10
    user_map_result = await load_user_map(_drift_client, user_map_settings)
    
    user_keys = list(user_map_result.user_map.keys())
    user_vals = list(user_map_result.values())

    stable_oracles = [str(x.oracle) for x in mainnet_spot_market_configs if 'USD' in x.symbol]

    if mode == 'margins':
        levs_none = list(do_dict(x, None) for x in user_vals)
        levs_init = list(do_dict(x, MarginCategory.INITIAL) for x in user_vals)
        levs_maint = list(do_dict(x, MarginCategory.MAINTENANCE) for x in user_vals)
        # print(levs_none[0].keys(), levs_init[0].keys(), levs_maint[0].keys())
        return (levs_none, levs_init, levs_maint), user_keys
    else:
        new_oracles_dat_up = {}
        oracle_distort_up = max(1 + oracle_distor, 1)

        new_oracles_dat_down = {}
        oracle_distort_down = max(1 - oracle_distor, 0)
        await _drift_client.account_subscriber.update_cache()
        cache_up = copy.deepcopy(_drift_client.account_subscriber.cache)
        cache_down = copy.deepcopy(_drift_client.account_subscriber.cache)
        for i,(key, val) in enumerate(_drift_client.account_subscriber.cache['oracle_price_data'].items()):
            new_oracles_dat_up[key] = copy.deepcopy(val)
            new_oracles_dat_down[key] = copy.deepcopy(val)
            if cov_matrix is not None and key in stable_oracles:
                continue
            if only_one_index is None or only_one_index == key:
                new_oracles_dat_up[key].data.price *= oracle_distort_up
                new_oracles_dat_down[key].data.price *= oracle_distort_down

        cache_up['oracle_price_data'] = new_oracles_dat_up
        cache_down['oracle_price_data'] = new_oracles_dat_down

        levs_none = list(do_dict(x, None, None) for x in user_vals)
        levs_up = list(do_dict(x, None, cache_up) for x in user_vals)
        levs_down = list(do_dict(x, None, cache_down) for x in user_vals)
        # print(levs_none[0].keys(), levs_init[0].keys(), levs_maint[0].keys())
        return (levs_none, levs_up, levs_down), user_keys

@st.cache_data
def cached_get_usermap_df(_drift_client, user_map_settings, mode, oracle_distort, only_one_index, cov_matrix):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(get_usermap_df(_drift_client, user_map_settings, mode, oracle_distort, only_one_index, cov_matrix))


def usermap_page(drift_client: DriftClient, env):
    s1, s11, s2, s3, s4 = st.columns(5)
    # await drift_client.account_subscriber.update_cache();
    perp_market_inspect = s1.selectbox('perp market:', list(range(22)))
    user_map_settings = s11.radio('user map settings:', ['all', 'active', 'idle', 'whales'], index=3)
    mode = s2.radio('mode:', ['oracle_distort', 'margin_cat'])
    mr_var = None
    cov_matrix = None
    only_one_index = None
    oracle_distort = None
    if mode == 'margin':
        kk = s3.radio('margin ratio:', [None, 'initial', 'maint'])
    else:    
        oracle_distort = s4.slider(
            'Select an oracle distort ',
            0.0, 1.0, .1, step=.05)

        only_one_index = s4.selectbox('only single oracle:', 
                                      ([None] 
                                    #    + list(drift_client.account_subscriber.cache['oracle_price_data'].keys())
                                       ),
                                      index=0
                                      )

        cov_matrix = s4.radio('distort settings:', [None, 'ignore stables'], index=1)
        


    (levs_none, levs_init, levs_maint), user_keys = cached_get_usermap_df(drift_client, user_map_settings, mode, oracle_distort, only_one_index, cov_matrix)

    if mode == 'margin':
        mr_var = s3.radio('margin ratio:', [None, 'initial', 'maint'])
        if mr_var is None:
            the_lev = levs_none
        elif mr_var == 'initial':
            the_lev = levs_init
        else:
            the_lev = levs_maint
    else:
        mr_var = s3.radio('oracle distortion:', [None, 'up', 'down'])
        if mr_var is None:
            the_lev = levs_none
        elif mr_var == 'up':
            the_lev = levs_init
        else:
            the_lev = levs_maint


    df = pd.DataFrame(the_lev)
    df.index = user_keys
    df = df.reset_index()
    
    tabs = st.tabs(['summary', 'leverage', 'spot0', 'scatter', 'raw tables'])
    with tabs[0]:
        s1,s2,s3 = st.columns(3)
        s1.metric('total users:', len(df))
        s2.metric('total usd value:', f'${df["net_usd_value"].sum():,.2f}')
        s3.metric('bankruptcies:', f'${df[df["net_usd_value"] < 0]["net_usd_value"].sum():,.2f}')
        st.plotly_chart(df['net_usd_value'].sort_values().reset_index(drop=True).plot())

    def get_rattt(row):
        # st.write(row)
        df1 = pd.Series([val/row['spot_asset'] * (row['perp_liability']+row['spot_liability']) 
                        if val > 0 else 0 for key,val in row['net_v'].items()]
                        )
        df1.index = ['spot_'+str(x)+'_all' for x in df1.index]

        df2 = pd.Series([val/(row['spot_asset']) * (row['perp_liability']) 
                        if val > 0 else 0 for key,val in row['net_v'].items()]
                        )
        df2.index = ['spot_'+str(x)+'_all_perp' for x in df2.index]

        df3 = pd.Series([val/(row['spot_asset']) * (row['spot_liability']) 
                        if val > 0 else 0 for key,val in row['net_v'].items()]
                        )
        df3.index = ['spot_'+str(x)+'_all_spot' for x in df3.index]
        
        df4 = pd.Series([val/(row['spot_asset']) * (row['net_p'][perp_market_inspect]) 
                        if val > 0 and row['net_p'][0] > 0 else 0 for key,val in row['net_v'].items()]
                        )
        df4.index = ['spot_'+str(x)+'_perp_'+str(perp_market_inspect)+'_long' for x in df4.index]

        df5 = pd.Series([val/(row['spot_asset']) * (row['net_p'][perp_market_inspect]) 
                        if val > 0 and row['net_p'][perp_market_inspect] < 0 else 0 for key,val in row['net_v'].items()]
                        )
        df5.index = ['spot_'+str(x)+'_perp_'+str(perp_market_inspect)+'_short' for x in df5.index]
        
        dffin = pd.concat([
            df1,
            df2,
            df3,
            df4,
            df5,
        ])
        return dffin
    df = pd.concat([df, df.apply(get_rattt, axis=1)],axis=1)
    res = pd.DataFrame({('spot'+str(i)): (df["spot_"+str(i)+'_all'].sum(), 
                                        df["spot_"+str(i)+'_all_spot'].sum(),
                                        df["spot_"+str(i)+'_all_perp'].sum() ,
                                        df["spot_"+str(i)+'_perp_'+str(perp_market_inspect)+'_long'].sum(),
                                        df["spot_"+str(i)+'_perp_'+str(perp_market_inspect)+'_short'].sum())
                                        for i in range(10)},
                                        
                    
                    index=['all_liabilities', 'all_spot', 'all_perp', 'perp_0_long', 'perp_0_short'])



    with tabs[1]:
        st.plotly_chart(df['leverage'].plot(kind='hist'))
        st.write(df.describe())
        lev_one_market = (df['net_p'].apply(lambda x: abs(x[perp_market_inspect]) if x[perp_market_inspect] != 0 else np.nan)/(df['spot_asset']-df['spot_liability']+df['upnl']))
        st.write(lev_one_market.describe())
    with tabs[2]:
        spot_market_inspect = st.selectbox('spot market:', list(range(10)))
        n = 0
        for num,val in enumerate(df.columns):
            if "spot_"+str(spot_market_inspect)+'_all' == val:
                n = num
        col = st.selectbox('column:', df.columns, index=n)
    with tabs[3]:

        
        # df[].plot
        # import plotly.graph_objects as go
        # fig = go.Figure()
        import plotly.express as px
        fig = px.scatter(df, x='leverage', y='spot_asset', size=col, hover_data=['index', 'leverage', col, 'spot_asset', 'spot_liability'])

        # Customize the layout if needed
        fig.update_layout(title='Bubble Plot of Size and SpotIi',
                        xaxis_title='Size',
                        yaxis_title='SpotIi')
        st.plotly_chart(fig)
    with tabs[4]:
        st.dataframe(res)
        st.dataframe(df)
