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

import requests
from aiocache import Cache
from aiocache import cached
from driftpy.types import InsuranceFundStakeAccount, SpotMarketAccount
from driftpy.addresses import * 
# using time module
import time
import plotly.express as px
from datafetch.transaction_fetch import transaction_history_for_account, load_token_balance
from datafetch.s3_fetch import load_if_s3_data
  
async def insurance_fund_page(ch: DriftClient, env):
    is_devnet = env == 'devnet'
    try:
        all_stakers = await ch.program.account['InsuranceFundStake'].all()
    except:
        all_stakers = []

    authorities = set()
    dfs = []
    for x in all_stakers: 
        key = str(x.public_key)
        staker: InsuranceFundStake = x.account

        if staker.authority not in authorities:
            authorities.add(staker.authority)

        data = staker.__dict__
        data.pop('padding')
        data['key'] = key
        dfs.append(data)

    state = await get_state_account(ch.program)
    s1, s2 = st.columns(2)
    dd = s1.selectbox('time period:', ['current month', 'all time'])

    tabs = st.tabs(['summary', 'balance', 'stakers', 'revenue flow', 'bankruptcies'])
    current_time = datetime.datetime.now()
    current_month = current_time.month
    current_year = current_time.year
    total_if_value = 0
    total_month_gain = 0

    # root_url = 'https://raw.githubusercontent.com/0xbigz/drift-v2-flat-data'
    root_url = 'https://raw.githubusercontent.com/ansatz-research/drift-v2-flat-data/my-branch-to-push/'
    p_url = root_url+'/data/perp_markets.csv'
    perp_df = pd.DataFrame()
    s_url = root_url+'/data/spot_markets.csv'
    spot_df = pd.DataFrame()
    try:
        perp_df = pd.read_csv(p_url, index_col=[0]).T
    except:
        st.warning('cannot load perp data: '+p_url)

    try:
        spot_df = pd.read_csv(s_url, index_col=[0]).T
    except:
        st.warning('cannot load spot data: '+s_url)

    name_rot, spot_asts = load_if_s3_data(current_year, current_month, dd, is_devnet)
    # st.dataframe(rot)
    # bankruptcies = rot[rot['amount']<0]
    # st.dataframe(bankruptcies)
    if_values = {}
    cur_spot = s2.selectbox('spot assets:', spot_asts)

    with tabs[1]:
        st.markdown('''[USDC Insurance Vault](https://solscan.io/account/2CqkQvYxp9Mq4PqLvAQ1eryYxebUh4Liyn5YMDtXsYci) 
                    | [Stake Example (Python)](https://github.com/0xbigz/driftpy-examples/blob/master/if_stake.py) 
                    | [Grafana Dashboard](https://metrics.drift.trade/d/hQYFylo4k/insurance-fund-balance?orgId=1&from=now-24h&to=now)''')
        bbs = st.columns(1)
        ccs = st.columns(1)

        conn = ch.program.provider.connection
        ts = time.time()
        spot_markets = []
        protocol_balances = []
        
        for i in range(state.number_of_spot_markets):
            spot = await get_spot_market_account(ch.program, i)
            total_n_shares = spot.insurance_fund.total_shares
            user_n_shares = spot.insurance_fund.user_shares


            unstaking_period = spot.insurance_fund.unstaking_period

            if spot.insurance_fund.total_factor:
                factor_for_protocol = 1 - spot.insurance_fund.user_factor/spot.insurance_fund.total_factor
            else:
                factor_for_protocol = 0 
            protocol_n_shares = total_n_shares - user_n_shares
            spot_markets.append(spot)

            # get_token_amount(spot.revenue_pool.scaled_balance, spot, )
            if_vault = get_insurance_fund_vault_public_key(ch.program_id, i)
            try:
                v_amount = await load_token_balance(conn, if_vault)
            except:
                v_amount = 0

            protocol_balance = v_amount * protocol_n_shares / (max(total_n_shares,1))
            protocol_balances.append(protocol_balance/10**spot.decimals)

            for staker_df in dfs: 
                if staker_df['market_index'] == i:
                    n_shares = staker_df['if_shares']
                    if total_n_shares != 0:
                        balance = v_amount * n_shares / total_n_shares
                        staker_df['$ balance'] = balance / (10 ** spot.decimals)

            name = str(''.join(map(chr, spot.name))).strip(' ')
            symbol = '$' if i==0 else ''
            if name == cur_spot:
                bbs[0].metric(f'{name} (marketIndex={i}) insurance vault balance:',
            f'{symbol}{v_amount/(10**spot.decimals):,.2f}',
            f'{symbol}{protocol_balance/10**spot.decimals:,.2f} protocol owned'
            ) 

            rev_pool_tokens = get_token_amount(
                            spot.revenue_pool.scaled_balance,
                            spot, 
                            'SpotBalanceType.Deposit()'
                        )

            #capped at 1000% APR
            next_payment = min(rev_pool_tokens/10, (v_amount*10/365/24))
            if v_amount > 0:
                staker_apr = (next_payment*24*365 * (1-factor_for_protocol))/v_amount
            else:
                staker_apr = 0

            total_if_value += (v_amount/10**spot.decimals) * spot.historical_oracle_data.last_oracle_price/1e6
            if_values[i] = (v_amount/10**spot.decimals)
            if name in name_rot:
                total_month_gain += (name_rot[name]['amount'].sum()) * spot.historical_oracle_data.last_oracle_price/1e6

            if name == cur_spot:
                ccs[0].metric('revenue pool', f'{symbol}{rev_pool_tokens/10**spot.decimals:,.2f}', 
                f'{symbol}{next_payment/10**spot.decimals:,.2f} next est. hr payment ('+str(np.round(factor_for_protocol*100, 2))+ '% protocol | '+str(np.round(staker_apr*100, 2))+'% staker APR)'
                )
                ccs[0].write('')

                # st.write(f'{name} (marketIndex={i}) insurance vault balance: {v_amount/QUOTE_PRECISION:,.2f} (protocol owned: {protocol_balance/QUOTE_PRECISION:,.2f})')
                ccs[0].write(f'{name} (marketIndex={i}) time since last settle: {np.round((ts - spot.insurance_fund.last_revenue_settle_ts)/(60*60), 2)} hours')
        
            
                # full_if_url = url_market_prefix+name.strip()+"/insurance-fund-records/2023/"+str(current_month)
                # st.write(full_if_url)
                try:
                    ccs[0].plotly_chart(name_rot[name]['amount'].cumsum().plot())
                except:
                    ccs[0].write('cannot load full_if_url')

    with tabs[0]:
        z1, z2, z3 = st.columns([1,1,2])
        z1.metric('Total Insurance:', 
        f'${(np.round(total_if_value, 2)):,.2f}',
        f'${(np.round(total_month_gain, 2)):,.2f} revenue '+dd,
        )

        z2.metric("Total Stakes", str(len(all_stakers)),str(len(authorities))+" Unique Stakers")

        # st.write(perp_df.columns)
        perp_if_at_risk = perp_df['market.insurance_claim.quote_max_insurance'].astype(float)
        perp_if_paid = perp_df['market.insurance_claim.quote_settled_insurance'].astype(float)
        
        z3.metric('Remaining Perp Coverage:', 
        f'${(np.round(perp_if_at_risk.sum() - perp_if_paid.sum(), 2)):,.2f}',
        f'- ${(np.round(perp_if_paid.sum(), 2)):,.2f} paid since inception',
        delta_color="normal"
        )
        with z3.expander('details'):
            # print(str(perp_df.columns.tolist()))
            tshow = perp_df[['market.insurance_claim.quote_settled_insurance', 
            'market.insurance_claim.quote_max_insurance', 'market.amm.total_liquidation_fee',
            'market.amm.total_fee_withdrawn']].astype(float)
            tshow.columns = ['settled_insurance', 'max_insurance', 'liq_fees', 'withdrawn_fees']
            st.dataframe(tshow)
        
        rot = name_rot.get(cur_spot, None)
        if rot is not None:
            apr_df = (rot['amount']/rot['insuranceVaultAmountBefore']*100*365.25*24/2).sort_index()#.rolling(24).mean()
            apr_fig = apr_df.plot(log_y=True)
            apr_fig.update_layout( 
                                title=cur_spot+' Revenue Emission',
                                xaxis_title="date",
                                yaxis_title="APR %",
                            )
            ifperf = ((rot['insuranceVaultAmountBefore']+rot['amount'])/rot['totalIfSharesAfter']).sort_index()
            ifperf.name = 'ifShare price'
            st.plotly_chart(apr_fig, True)
            st.plotly_chart(ifperf.plot(log_y=True))
            st.plotly_chart((ifperf.resample('1W').last().pct_change()*100).plot(kind='bar'))
        

        # st.dataframe(perp_df)

    with tabs[2]: 
        dcol1, dcol2 = st.columns(2)
        stakers = pd.DataFrame(data=dfs)

        total_shares = [spot_markets[r['market_index']].insurance_fund.total_shares for i, r in stakers.iterrows()]
        precisions = [(10 ** spot_markets[r['market_index']].decimals) for i, r in stakers.iterrows()]
        stakers['cost_basis'] /= precisions 
        stakers['if_shares'] /= precisions
        stakers['last_withdraw_request_shares'] /= precisions
        stakers['last_withdraw_request_shares'] = stakers['last_withdraw_request_shares'].replace(0, np.nan)
        stakers['last_withdraw_request_value'] /= precisions
        stakers['last_withdraw_request_value'] = stakers['last_withdraw_request_value'].replace(0, np.nan)
        stakers['total_shares'] = total_shares
        stakers['total_shares'] /= precisions
        stakers['available_time_to_withdraw'] = stakers['last_withdraw_request_ts'].apply(lambda x: pd.to_datetime((x + unstaking_period) * 1e9) if x!=0 else x)
        stakers['last_withdraw_request_ts'] = stakers['last_withdraw_request_ts'].apply(lambda x: pd.to_datetime(x * 1e9) if x!=0 else x)
        stakers['own %'] = stakers['if_shares']/stakers['total_shares'] * 100
        stakers['authority'] = stakers['authority'].astype(str)
        # print(stakers.columns)    
        stakers['$ balance'] = stakers['$ balance'].astype(float)
        i = dcol1.selectbox('market index:', sorted(list(stakers['market_index'].unique())))

        st.dataframe(stakers[stakers.market_index==i][['authority', 'market_index', '$ balance', 'if_shares', 'total_shares', 'own %', 'cost_basis', 'last_withdraw_request_shares', 'if_base',
        'last_withdraw_request_value',
        'last_withdraw_request_ts', 'available_time_to_withdraw', 'last_valid_ts',  'key',
        ]].sort_values('$ balance', ascending=False).reset_index(drop=True))

        df = stakers[['$ balance', 'authority', 'own %', 'market_index']]
        # z1, z2 = st.columns(2)
        # pie1, z2 = st.columns(2)

        df2download2 = stakers.to_csv(escapechar='"').encode('utf-8')
        st.download_button(
            label="share stakers statistics [bytes="+str(len(df))+"]",
            data=df2download2,
            file_name='if_stakers_full.csv',
            mime='text/csv',
        )

        df = df[df.market_index==i]
        df = pd.concat([df, pd.DataFrame([[protocol_balances[i], 'protocol', 100 - df['own %'].sum(), i]], index=[0], columns=df.columns)])
        
        other = df.sort_values('own %', ascending=False).iloc[10:].sum()
        other.loc['authority'] = 'OTHER'
        other = pd.DataFrame(other).T
        # print(other)
        df1 = pd.concat([
            df.sort_values('own %', ascending=False).iloc[:10],
            other,
        ])
        fig = px.pie(df1, values='own %', names='authority',
                    title='MarketIndex='+str(i)+' IF breakdown',
                    hover_data=['$ balance'], labels={'$ balance':'balance'})
        dcol2.plotly_chart(fig)
        dcol1.metric("Total Insurance (Market Index ="+str(i)+")", f'{if_values[i]:,.6f}',str(len(df.authority.unique()))+" Unique Stakers")
        stakers_market_index = stakers[stakers.market_index==i]
        stakers_frac = stakers_market_index["$ balance"]*stakers_market_index["last_withdraw_request_shares"]/stakers_market_index["if_shares"]
        dcol1.metric("Amount in Unstaking Cooldown:", 
                     f'{stakers_market_index["last_withdraw_request_value"].sum():,.2f}',
                     f'{(stakers_frac - stakers_market_index["last_withdraw_request_value"]).sum():,.2f} bonus for remaining stakers',
                     )
        with st.expander('request withdraw ts'):
            tpl = stakers_market_index.replace(0, np.nan).dropna(subset=['last_withdraw_request_ts']).set_index('last_withdraw_request_ts')
            st.plotly_chart(tpl['last_withdraw_request_value'].sort_index().cumsum().plot())

        selected = st.selectbox('authority:', sorted(stakers.authority.unique()))
        url_market_prefix = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/authority/'
        full_if_url = url_market_prefix+selected+"/insurance-fund-stake-records/2023/3"
        try:
            st.write(full_if_url)
            rot = pd.read_csv(full_if_url)
            rot = rot.set_index('ts')
            rot.index = pd.to_datetime((rot.index * 1e9).astype(int))
            st.dataframe(rot)
        except:
            st.warning('cannot read: ' + full_if_url)


        for x in all_stakers:
            if str(x.account.authority)==selected:
                with st.expander(str(x.public_key)):
                    st.write(str(x.account.__dict__))

    with tabs[3]:
        def flat_file_list_container_to_name(x):
            return str(''.join(map(chr, [int(y.strip()) for y in x.split('    ')[1:]]))).strip()

        import plotly.graph_objects as go
        if len(perp_df):
            # st.write(perp_df['market.name'].values[0].split('    ')[1:])
            perp_names = perp_df['market.name'].apply(flat_file_list_container_to_name)
            spot_pools =  spot_df['spot_market.name'].apply(flat_file_list_container_to_name)
            vv = perp_df['market.amm.fee_pool.scaled_balance'].astype(float).tolist()
            vv += spot_df['spot_market.spot_fee_pool.scaled_balance'].astype(float).tolist()
            vv += spot_df['spot_market.revenue_pool.scaled_balance'].astype(float).tolist()[:1]
            usdc_if_value = if_values[0]
            vv += [usdc_if_value] #todo
            fee_pools = perp_names.tolist() + spot_pools.tolist()
            usdc_rev_pool_idx = len(fee_pools)
            if_idx = len(fee_pools) + 1
            all_pools = fee_pools + ['fee pool']
            all_pool_indexes = range(0, len(all_pools)+1)
            perp_indexes = range(0, len(perp_names))
            # st.write(perp_names.tolist())
            # st.write(vv)
            fig = go.Figure(go.Sankey(
                arrangement = "snap",
                node = {
                    "label": all_pools + ['usdc revenue pool'] + ['usdc insurance fund'],
                    # "x": [0.2, 0.1, 0.5, 0.7, 0.3, 0.5],
                    # "y": [0.7, 0.5, 0.2, 0.4, 0.2, 0.3],
                    'pad':10},  # 10 Pixels
                link = {
                    'arrowlen': 20,
                    # "color": 'grey',
                    "source": [x for x in all_pool_indexes],
                    "target": [usdc_rev_pool_idx if x < len(all_pool_indexes)-2 else x+1 for x in all_pool_indexes ],
                    "value": [vv[x] for x in all_pool_indexes]}))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        df = None

        mi = st.selectbox('market', [0, 1])
        # try:
        br = []
        for x in [0, 1, 2]:
            rr = requests.get('https://mainnet-beta.api.drift.trade/bankruptcies/?pageIndex='+str(x)+'&pageSize=1000').json()
            br1 = rr['data']['bankruptcies']
            br.append(br1)
        br = [element for sublist in br for element in sublist]
        indexes = [x['ts'] for x in br]
        perpsdf = pd.DataFrame([x['perpBankruptcy'] for x in br], index=indexes)
        perpsdf['marketType'] = 'perp'
        spotsdf = pd.DataFrame([x['spotBankruptcy'] for x in br], index=indexes)
        spotsdf['marketType'] = 'spot'
        spotsdf = spotsdf[['marketType', 'marketIndex', 'ifPayment', 'borrowAmount']]
        spotsdf['borrowAmount'] = spotsdf['borrowAmount'].astype(float) * -1
        spotsdf.columns = ['marketType', 'marketIndex', 'ifPayment', 'pnl']
        df = pd.concat(
            [perpsdf[['marketType', 'marketIndex', 'ifPayment', 'pnl']], 
                       spotsdf[['marketType', 'marketIndex', 'ifPayment', 'pnl']],
                    ],
                       axis=0
                       
                       ).sort_index()
        
        if mi == 0:
            df['ifPayment'] = df['ifPayment'].astype(float)/(QUOTE_PRECISION)
            df['pnl'] = df['pnl'].astype(float)/(QUOTE_PRECISION)
            df = df[((df.marketType=='perp') | (df.marketIndex==0))]
            df = df[((df.ifPayment !=0) & (df.pnl != 0))]
        else:
            df['ifPayment'] = df['ifPayment'].astype(float)/(1e9)
            df['pnl'] = df['pnl'].astype(float)/(1e9)
            df = df[((df.marketType=='spot') & (df.marketIndex==1))]
            df = df[((df.ifPayment !=0) & (df.pnl != 0))]

        # .merge(
        #     pd.DataFrame([x['perpBankruptcy'] for x in br], index=indexes),
        #     # how='outer'
        #     )


        # except:
        #     st.warning('cannot load bankruptcies')

        # df = df.set_index('ts')
        df.index = pd.to_datetime((df.index.astype(int) * 1e9).astype(int))

        tt = df[['ifPayment','pnl']].cumsum()
        tt['socialLoss'] = -(tt['pnl']+tt['ifPayment'])
        st.plotly_chart(tt[['ifPayment','socialLoss']].plot())
        st.dataframe(df)


        