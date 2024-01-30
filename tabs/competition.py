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
from anchorpy.provider import Signature
from anchorpy import ProgramAccount

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



async def competitions(ch: DriftClient, env):
    dochainload = st.radio('do chain load:', [True, False], index=1, horizontal=True)

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
        txns = await transaction_history_for_account(provider.connection, Pubkey.from_string(pid), None, None, 1000)
        st.write(f'last {len(txns)} transactions')

        df = pd.DataFrame(txns)
        df['datetime'] = pd.to_datetime((df['blockTime'].astype(int)*1e9).astype(int))
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
            tts[idx].json(([field]), expanded=False)
            tts[idx].dataframe(pd.DataFrame(idl[field]).astype(str))

    target_key = Pubkey.from_string('EDVhTaKcxfnkNxcnVCVrgdRuSact8p6ywwt1KJWjyX8Q')
    if not dochainload:
        # just load a single competition account
        act = await program.account['Competition'].fetch(target_key)
        accounts = [[act], []]
        # return 0
    else:
        accounts = [(await program.account[act['name']].all()) for act in idl['accounts']]
    vault_df = pd.DataFrame()
    vault_dep_def = pd.DataFrame()
    with tabs[1]:
        for acts in accounts:
            res = []
            for x in acts:
                dd = None
                if isinstance(x, ProgramAccount):
                    key = x.public_key
                    dd = x.account.__dict__
                else:
                    key = target_key
                    dd = x.__dict__

                if 'name' in dd:
                   dd['name'] = bytes(dd['name']).decode('utf-8').strip(' ')
                ser = pd.Series(dd, name=key)
                res.append(ser)
            if len(res):
                res = pd.concat(res,axis=1)
                vault_df = res
                vault_df = vault_df.T#.sort_values(by='next_expiry_ts', ascending=False).T
                st.write(len(res))
                if len(vault_df) < 3:
                    st.dataframe(vault_df.T)
                else:
                    st.dataframe(vault_df)
   
    with tabs[0]:
        s1, s2, s3 = st.columns(3)
        s1.metric('number of '+idl['accounts'][0]['name']+'(s):', len(accounts[0]), str(len(accounts[1]))+' '+idl['accounts'][1]['name']+'(s)')
    
        if preset_programs == 'drift-competitions':
            draw_tabs = st.tabs(['current', 'past'])
            with draw_tabs[0]:
                if dochainload:
                    derived_snap_name = 'latest_snapshot(wb)'
                    vault_df[derived_snap_name] = 0
                    vault_df['last trade ts'] = 0
                    try:
                        user_stats_pubkeys = [Pubkey.from_string(x) for x in vault_df['user_stats'].values]
                        every_user_stats = await ch.program.account['UserStats'].fetch_multiple(
                            user_stats_pubkeys
                            )
                        # mode = st.radio('mode:', ['current state', 'extrapolated'], horizontal=True)
                        vault_df[derived_snap_name] = [x.fees.total_fee_paid//100 for x in every_user_stats]
                        vault_df['last trade ts'] = [pd.to_datetime(x.last_taker_volume30d_ts*1e9) for x in every_user_stats]
                    except Exception as e:
                        st.warning('couldnt load user stats')
                        st.warning('error:'+str(e))

                comp_acct = accounts[0][0]
                if isinstance(comp_acct, ProgramAccount):
                    comp_acct = comp_acct.account
                vsum = vault_df.sum()

                spot = await get_spot_market_account(ch.program, 0)
                if_vault = get_insurance_fund_vault_public_key(ch.program_id, 0)
                try:
                    v_amount = await load_token_balance(provider.connection, if_vault)
                except Exception as e:
                    st.warning('failed to load IF:', e)
                    v_amount = 0

                pol_vault =  (1-spot.insurance_fund.user_shares/spot.insurance_fund.total_shares)*v_amount/1e6
                max_prize = np.round((pol_vault - comp_acct.sponsor_info.min_sponsor_amount/1e6) * comp_acct.sponsor_info.max_sponsor_fraction/1e6, 2)
                
                max_prize = max(0, max_prize)
                TENK = 10000
                FIFTYK = 50000
                FIVEK = 5000
                ONEK = 1000
                prizes = [
                    np.round(max(min(ONEK, max_prize / 10), min(max_prize / 25, TENK)), 2),
                    np.round(max(min(FIVEK, max_prize / 2), min(max_prize / 12, FIFTYK)), 2),
                    max_prize
                ]
                
                # prizes = [np.round(min(1000, max_prize/10),2), np.round(min(5000, max_prize/2),2), max_prize]
                # prize_ev = 2000

                odds = [sum(prizes)/x for x in prizes]
                odds_rounded = [np.ceil(odds[0]), np.ceil(odds[1]), np.floor(odds[2])]
                # st.write('odds:', odds_rounded)
                prize_ev = sum([odds_rounded[i]/sum(odds_rounded)*prizes[i] for i in range(len(prizes))])
                s3.metric('expected prize value', f'${prize_ev:,.2f}', 
                          'prize buckets: $' + ', $'.join([str(x) for x in prizes]), help=
                        '''
    odds of each bucket (out of {'''+str(sum(odds_rounded))+'''}):''' + str(odds_rounded) + '''

    ''')
                takervolnom = '~ taker volume ($)'
                st.write('round number:', comp_acct.round_number, ' | number of winners:', comp_acct.number_of_winners, ' | round end (UTC):', 
                        pd.to_datetime(comp_acct.next_round_expiry_ts*1e9))
                st.markdown('### leaderboard:')

                # prize_ev = s3.number_input('expected prize ($):', step=1.0, min_value=0.0, value=2000.0, max_value=1e9)
                # vv = s3.radio('prize choice:', ['custom', 1000, 5000, 10000], horizontal=True)
                # ddf = pd.read_csv('https://drift-historical-data-v2.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/market/SOL-PERP/tradeRecords/2023/20230205')
                # st.write(ddf)
                # st.write(ddf.describe())
                if dochainload:
                    vault_df['entries'] = (vault_df[derived_snap_name] - vault_df['previous_snapshot_score']).max(0) + vault_df['bonus_score'] 
                    vault_df['entries'] =  vault_df['entries'].apply(lambda x: min(x, comp_acct.max_entries_per_competitor))
                    vault_df['EV ($)'] = vault_df['entries']/vault_df['entries'].sum() * prize_ev
                    vault_df['chance of a win'] = 1 - ((1- vault_df['entries']/vault_df['entries'].sum()) ** comp_acct.number_of_winners)
                    vault_df[takervolnom] = (vault_df[derived_snap_name] - vault_df['previous_snapshot_score'])/10

                    s2.metric('total entries:', f"{vault_df['entries'].sum():,.0f}",
                            f"{vsum['bonus_score']} from bonus_score")
                    d1, = st.columns(1)        
                    
                    vault_df1 = vault_df.reset_index().set_index('authority')
                    vault_df1 = vault_df1[['entries', 'EV ($)', 'chance of a win', 'unclaimed_winnings', takervolnom, 'last trade ts']].sort_values(by='entries', ascending=False)
                    vault_df1 = vault_df1.reset_index()
                    vault_df1.index.name = 'rank'
                    vault_df1.index += 1

                    d1.dataframe(vault_df1, use_container_width=True)

                    st.markdown('### odds calculator:')
                    do0, do1, do2 = st.columns([1,2,5])
                    up_to_n_entries = do1.number_input('entries:', min_value=0, value=1000)

                    direct = do0.radio('direction:', ['less than', 'greater than', 'precisely'])
                    dd = 1
                    dddf = pd.DataFrame([1])
                    if direct == 'less than':
                        dddf =  vault_df[vault_df.entries <= up_to_n_entries].entries
                        dd = dddf.sum()
                        st.write(dd)
                    elif direct == 'greater than':
                        dddf = vault_df[vault_df.entries >= up_to_n_entries].entries
                        dd = dddf.sum()
                    else:
                        dd = up_to_n_entries
                    win_type = do1.radio('win type:', ['top 3', 'any win'])
                    if win_type == 'top 3':
                        n = min(3, comp_acct.number_of_winners)
                    else:
                        n = comp_acct.number_of_winners
                    d1 = dd/vault_df['entries'].sum()
                    chance_of_a_win = 1 - ((1- d1) ** n)

                    pplnom = 'anyone of '+str(len(dddf)) if direct != 'someone' else direct
                    if chance_of_a_win * 100 > 0.02:
                        do2.write(f'{pplnom} who has {direct} {up_to_n_entries} tickets has a {chance_of_a_win*100:,.2f}% of {win_type}!')
                        do2.write(f'odds: 1 in {int(np.ceil(1/chance_of_a_win))}')

                    else:
                        do2.write(f'{pplnom} who has {direct} {up_to_n_entries} tickets has a {chance_of_a_win*100:,.9f}% of {win_type}!')
                        do2.write(f'odds: 1 in {int(np.ceil(1/chance_of_a_win))}')

                    st.write('never traded on drift:', (vault_df[derived_snap_name]<=1e-6).sum(), '/', len(vault_df))

                    st.write('never traded this round:', ((vault_df[derived_snap_name] - vault_df['previous_snapshot_score'])<=1e-6).sum(), '/', len(vault_df))
                    # st.write((vault_df[derived_snap_name]))
            with draw_tabs[1]:
                try:
                    api_result = requests.get('https://mainnet-beta.api.drift.trade/sweepstakes/results').json()
                except:
                    api_result = {}

                if 'data' in api_result:
                    dfs = []
                    # st.write(api_result['data']['competitionHistory'])
                    for x in api_result['data']['competitionHistory']:
                        dfs.append(pd.Series(x))
                    df = pd.concat(dfs, axis=1).T
                    df_cols_old = list(df.columns)
                    df['prize'] = df['summaryEvent'].apply(lambda x: float(x['prizeValue']))/1e6
                    df['startDate'] = df['startTs'].apply(lambda x: pd.to_datetime(int(x*1e9)))
                    df['endDate'] = df['endTs'].apply(lambda x: pd.to_datetime(int(x*1e9)))

                    st.metric('total prize', f'${df["prize"].astype(float).sum():,.2f}')
            
                    st.write(df[['endDate', 'prize']+df_cols_old])


                return 0 
                winner_txs = [x.strip(' ') for x in '''2D2PwCxXAoJSnycuzoCwu2ayER9MkTYnEuCYhe6JxdzLdwtQNcrUrpVwJH1xxrX4jNTnS7Cd6WNhF8DR6fGSTz5c
                5WfmX3foRoaapwPjKMsDnc7iCcmDBhX1Zd3YvSjX3yq88EgReuEz3vKTZ8TonyxiQadwpna3M56maSa5dA5LvYx4
                41xCXx5CWav7RuJ2RVTAqJvRvzmK65KvUBa2VfQ1xEgnpjGJSBzxLeQAuRmEsNTgn848P54adWFr57KMNNeUT8VT
                neHwH2TgvoHe1trFTUfT8RY7xARJRSf4A1S8w6GKmBRRSHLMaWM3CR5roWpf9EDMdCgkoCraPxCBnasTw2MQtuR
                3u1X9hrPm9LLkXu6gqkV82uYbyc1MK4Mhxxnnx5PYT3XnYPiRwuUVnFdEuz1QkXqDxYwtGdmifZGebMN81qaST9N
                2aXeaGPW2CXb3KdHwZHvpshyMhpEnvJ9WbkuEqMQAELjKDor3w1JYrJsybxfkgzS9rhXM6LgJYpyynuoWWkUFSTZ
                4jWmnibKdthi82EnRAcETUW8k8Jmd5jYaRTDUXAMSZCTHz2hgDRicFDPcQ9TojKsac4pEpTKfXdbmFEpdVuYYQC3
                4dzbm8FqGc7EpUbouVyc7rBEyea5PHDbJFhJtxFFiGw3nsMqf8gfwM9rckguULoyTWMBauBctycEPUYfmxg2DnH5
                moAMKo1xFHehamizKNWtv2BgwJMSDdBVVpBTFi6cQfPFmNSvSCGVHmgYXbktYL7Agkq2oXoLRtmYQzia1aRnncZ
                3Z6TmwebYm55k4gmLrceXqbHt3Ki4YoYjQyx2RVZGizSHQeY5Kb61Kixns18qqh1yx5jfQaQhjZDRPmGz42FpeMq
                kBszZ6DfiHAV1Hybq1Pjk9r6mYBY5bov94w7ihdfXtX1RuBcrJNgnEmyJbYen3ncxCGxTPiP3Eq6iKD5vLkGZ7B
                XYuVwMHxEZkmK7qTLnvwzPooCbpyvceb5ztQL3BHLG1gBVVn8iQPQ7NS4ibxBGEMCEGgCewaDjcktu8Sn1425ZZ
                4PM4qaLKAgZvo4nrKSPQAaUBSvvptevgKZ6U8QrtirveUdXFm2iwhD6M3JoxnMCAWRfVonx2Kd8SAHwNxXLECaJX
                2phGSWMqNVqCqqC2SbZBsTNBUaK4LF18GsSpFZV712x8wSH5azStCigUNZzTu5uDdL52xWNFZK79rGzCCzLxUMDu
                4xGjnp7vgZc8P2mwHKk9SzJFopvurTBeHkq5SmTy26Ywvit97teusaDsh4dmCcdgQJR3u6eMwvNKipvxqJM3yyaN
                3bgwJGffTLf7H47pa6DeboN48k3fYJrcmqocpEFdsct1DKH4n9Jim21wFk1NQ3cnR82CVECFZj1DoiT93h8Ur4Nd
                i1DVeGXLa5GXMRu7sp79t2Ea1g32XaqgDYWiWx1sQLmScFkpnrbzo9V3SMz2qe9Wsv1qmNZZrQvWjrEEtiPwpjw
                3GwxAkcuNGe5iS5fouRTC4j1KZFS5k22JaPWFg7jJzZieJbUaJbKBLPMzJBNEnsmMRcTcPTqBr684aKKd4uEta77
                3BeEUc3agJSLPcjeKerCgH4DMog2AQy6XSikxbLn35BL27GRnGRnT2k52wDAw7ZPD2HxHMAnBScJjBqHAvcZZGdW
                5694k5RGnrxGcVYknyiftv8KwhUDgRNGqdmSwXy396B33d59ChSggETqW8SSC46DaXruY41tZckDGepXn2eAR9Mp
                3czmwbiXr6MQL7KHRftq8f1HPCYpqgLwpDEyocQYqjHsboP9uTv6SZohdUmuetepbAGFDFbLe6PpTJ6vuCHPvFf5
                3izpRnpd9UxdDtWmG6u1hK9d3rYiu5Tt6SHvmgerHyWSDnM4XiYm9bqW6bTYkF8PBHrMkooUUZhkwoiiGZfLTeKz
                3UR2Ur7xdekfpob9J2mkaBk1Zs1AxNJBESjgUfAmQwzcZNt4cheRe9gD3AxLrz2AgcfP7gM1FFR9wSunN3vu8mZZ
                5XS61zvuwbatHofDxUJmb6irxdzWQKrbKmRJg5e3aK3H8ChTu6SEjeyr4EJnqZAzKFNufkPKaXFVcbzGoWntWKnq
                4aonrUVTaJkPfcfBpGPnXgVnwLGUGpT6BVnCnBSUiqBLqWCSqJyZeswWkn43EZeneWWXSUB2UNmAYexhAyDZcikJ
                3AhzmFgww1SYhtmE9Cf1FtFLC3tPaFBo7y6oHHd5RbCdz8PdgFFg9HWBFhw3q4y3i86sSh6KfTeUz6cFDo2vMwbe
                dENP6BqZ9Ff3vipmkVmF7nDVUKFxqrXFmmBVxjvYSTxwGfukUdfLFkJWTvarEVsVBPePbrv2gXb1QeziyptpcCr
                3Z3yBkWhFRoxmT13HjYFRG62uNVDraRpmiPqUePF7P7ctwoaygD72GnD7qRqrpbrRSdzcnwKe68UBDGgw1UxrmgH
                4kZKKFQkCXpm5kufzBsYwAAH8q2KE7V8JyobKHAi7UWHBNoHLg2yCg2BgnuevspGmJ8WZEU2rDUrQAxzDPY3M1kq
                4oZchYfaTgknqT3jGhm4yjsb4bh5w6am1Z7ppyU2BNnvJNpsu4jprdHxXEj3hXvQU7oiXLZgk9FfxZQCtcVPwUxo
                4RFAj6tFY1ZaKMdytyYjH57LqScchZ4SP3VDdsV46Ppv24qvqso6PiANJXTZ9Krc57yoakwHm3s8raACVh7XW5Uu
                3mDQCC3GRpbnrPHeZHcEpPRuBHg2BaiRFL35A4sxpSMwoAETegE7HSbQDGJxW3aaAHkk3rbpKimJgWiCRnZnRxC4'''.strip('').split('\n')]
                parser = EventParser(program.program_id, program.coder)
                all_sigs = df['signature'].values.tolist()
                c1, c2 = st.columns(2)
                do_check = c1.radio('see historical round result:', [None, 0], index=0, horizontal=True)
                # signature = c2.selectbox('tx signatures:', winner_txs, key='jkdfla')
                # st.write(df[df.signature==signature])
                if do_check is not None:
                    theset = winner_txs
                    idxes = [i for i,x in enumerate(all_sigs) if x in winner_txs]
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
                    logs = []
                    i = 0
                    for tx, sig in zip(txs, sigs):
                        def call_b(evt): 
                            logs.append(evt)
                            # i += 1
                        # likely rate limited
                        # if 'result' not in tx: 
                        #     st.write(tx['error'])
                        #     break 
                        # st.write(tx['result']['meta']['logMessages'], expanded=True)
                        parser.parse_logs(tx['result']['meta']['logMessages'], call_b)
                    # st.write(logs)
                    dd = pd.DataFrame([x[-1] for x in logs])
                    dd = dd[['winnerPlacement', 'competitorAuthority', 'prizeValue']]
                    dd['prizeValue']/=1e6
                    dd['winnerPlacement']+=1
                    dd = dd.set_index('winnerPlacement').sort_index()

                    st.markdown('### winners')
                    st.dataframe(dd, height=1150)
                        # st.json(transaction_got, expanded=False)

                    st.json(winner_txs, expanded=False)
