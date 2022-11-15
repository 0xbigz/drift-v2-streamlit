
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
from driftpy.clearing_house_user import ClearingHouseUser
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
import matplotlib.pyplot as plt 
from driftpy.clearing_house_user import get_token_amount
from driftpy.types import SpotBalanceType

async def show_pid_positions(url: str, clearing_house: ClearingHouse):
    ch = clearing_house
    state = await get_state_account(ch.program)

    with st.expander('state'):
        st.json(state.__dict__)

    all_users = await ch.program.account['User'].all()

    authorities = set()
    dfs = {}
    spotdfs = {}
    # healths = []
    from driftpy.types import User
    kp = Keypair()
    ch = ClearingHouse(ch.program, kp)

    chu = ClearingHouseUser(ch, authority=all_users[0].account.authority, use_cache=True)
    await chu.set_cache()
    cache = chu.CACHE

    for x in all_users:
        key = str(x.public_key)
        account: User = x.account

        chu = ClearingHouseUser(ch, authority=account.authority, subaccount_id=account.sub_account_id, use_cache=True)
        cache['user'] = account # update cache to look at the correct user account
        await chu.set_cache(cache)
        leverage = await chu.get_leverage()

        # mr = await chu.get_margin_requirement('Maintenance')
        # tc = await chu.get_total_collateral('Maintenance')
        # health = (tc-mr)
        # healths.append(health)

        if account.authority not in authorities:
            authorities.add(account.authority)

        dfs[key] = []
        name = str(''.join(map(chr, account.name)))

        for idx, pos in enumerate(x.account.perp_positions):
            dd = pos.__dict__
            dd['position_index'] = idx
            dd['authority'] = str(x.account.authority)
            dd['name'] = name
            dd['leverage'] = leverage / 10_000
            dfs[key].append(pd.Series(dd))
        dfs[key] = pd.concat(dfs[key],axis=1) 
        
        spotdfs[key] = []
        for idx, pos in enumerate(x.account.spot_positions):
            dd = pos.__dict__
            dd['position_index'] = idx
            dd['authority'] = str(x.account.authority)
            dd['name'] = name
            spotdfs[key].append(pd.Series(dd))
        spotdfs[key] = pd.concat(spotdfs[key],axis=1)    
    
    st.text('Number of Driftoors: ' + str(len(all_users)))
    st.text('Number of Unique Driftoors: ' + str(len(authorities)))

    perps = pd.concat(dfs,axis=1).T
    perps.index = perps.index.set_names(['public_key', 'idx2'])
    perps = perps.reset_index()

    spots = pd.concat(spotdfs,axis=1).T
    spots.index = spots.index.set_names(['public_key', 'idx2'])
    spots = spots.reset_index()
    
    markettype = st.radio("MarketType", ('Perp', 'Spot'))
    if markettype == 'Perp':
        num_markets = state.number_of_markets
    else:
        num_markets = state.number_of_spot_markets

    tabs = st.tabs([str(x) for x in range(num_markets)])
    for market_index, tab in enumerate(tabs):
        market_index = int(market_index)
        with tab:
            if markettype == 'Perp':
                market = await get_perp_market_account(ch.program, market_index)
                market_name = ''.join(map(chr, market.name));

                with st.expander(markettype+" market market_index="+str(market_index)+' '+market_name):
                    mdf = serialize_perp_market_2(market).T
                    st.dataframe(mdf)

                df1 = perps[((perps.base_asset_amount!=0) 
                            | (perps.quote_asset_amount!=0)
                            |  (perps.lp_shares!=0)
                             |  (perps.open_orders!=0)
                            ) 
                            & (perps.market_index==market_index)
                        ].sort_values('base_asset_amount', ascending=False).reset_index(drop=True)

                st.text('user long %:')
                pct_long = market.amm.base_asset_amount_long / (market.amm.base_asset_amount_long - market.amm.base_asset_amount_short + 1e-10) 
                my_bar = st.progress(pct_long)
                
                # st.text('user long % sentiment:')
                # sentiment = 0
                # if len(df1):
                #     sentiment = df1['base_asset_amount'].pipe(np.sign).sum()/len(df1) + .5
                # my_bar = st.progress(sentiment)

                df1['base_asset_amount'] /= 1e9
                df1['remainder_base_asset_amount'] /= 1e9
                df1['lp_shares'] /= 1e9
                df1['quote_asset_amount'] /= 1e6
                df1['quote_entry_amount'] /= 1e6
                df1['quote_break_even_amount'] /= 1e6

                df1['entry_price'] = -df1['quote_entry_amount']/df1['base_asset_amount'].apply(lambda x: 1 if x==0 else x)
                df1['breakeven_price'] = -df1['quote_break_even_amount']/df1['base_asset_amount'].apply(lambda x: 1 if x==0 else x)
                df1['cost_basis'] = -df1['quote_asset_amount']/df1['base_asset_amount'].apply(lambda x: -1 if x==0 else x)

                toshow = df1[[
                    'authority', 
                    'name', 
                    'open_orders', 
                    'leverage',
                    'base_asset_amount', 
                    'lp_shares',
                    'remainder_base_asset_amount',
                    'entry_price', 
                    'breakeven_price', 
                    'cost_basis',
                    'public_key', 
                ]]
                st.text('User Perp Positions ('+ str(len(df1)) +')')
                st.dataframe(toshow)
            else:
                market = await get_spot_market_account(ch.program, market_index)
                market_name = ''.join(map(chr, market.name));

                with st.expander(markettype+" market market_index="+str(market_index)+' '+market_name):
                    mdf = serialize_spot_market(market).T
                    st.table(mdf)

                conn = ch.program.provider.connection
                ivault_pk = market.insurance_fund.vault
                svault_pk = market.vault
                iv_amount = int((await conn.get_token_account_balance(ivault_pk))['result']['value']['amount'])
                sv_amount = int((await conn.get_token_account_balance(svault_pk))['result']['value']['amount'])

                st.text('insurance/spot vault balances:'+ str(iv_amount/1e6)+'/'+str(sv_amount/1e6))

                df1 = spots[(spots.scaled_balance!=0) & (spots.market_index==market_index)
                        ].sort_values('scaled_balance', ascending=False).reset_index(drop=True)\
                            .drop(['idx2', 'padding'], axis=1)
                for col in ['scaled_balance']:
                    df1[col] /= 1e9
                for col in ['cumulative_deposits']:
                    df1[col] /= (10 ** market.decimals)

                st.text('User Spot Positions ('+ str(len(df1)) +')')
                st.dataframe(df1[['public_key', 'balance_type', 'scaled_balance', 'cumulative_deposits', 'open_orders', 'authority']])

                total_cumm_deposits = df1['cumulative_deposits'].sum()

                sb = df1[df1['balance_type'].map(str) == 'SpotBalanceType.Deposit()']
                deposits = sb['scaled_balance'].values 
                total_deposits = 0
                for depo in deposits: 
                    token_amount = get_token_amount(
                        depo * 1e9, 
                        market, 
                        'SpotBalanceType.Deposit()'
                    )
                    total_deposits += token_amount
                        
                sb = df1[df1['balance_type'].map(str) == 'SpotBalanceType.Borrow()']
                borrows = sb['scaled_balance'].values 
                total_borrows = 0
                for borr in borrows: 
                    token_amount = get_token_amount(
                        borr * 1e9, 
                        market, 
                        'SpotBalanceType.Borrow()'
                    )
                    total_borrows += token_amount

                total_deposits /= (10 ** market.decimals)
                total_borrows /= (10 ** market.decimals)
                
                st.text(f'total deposits (token amount): {total_deposits:,.2f}')
                st.text(f'total borrows (token amount): {total_borrows:,.2f}')
                st.text(f'total cummulative deposits: {total_cumm_deposits:,.2f}')

                fig1, ax1 = plt.subplots()
                fig1.set_size_inches(15.5, 5.5)
                ax1.pie([total_deposits, total_borrows], labels=['deposits', 'borrows'], autopct='%1.5f%%',
                        startangle=90)
                ax1.axis('equal')  
                st.pyplot(fig1)

        authority = st.text_input('public_key:') 
        auth_df = df1[df1['public_key'] == authority]
        if len(auth_df) != 0: 
            columns = list(auth_df.columns)
            if markettype == 'Perp':
                columns.pop(columns.index('idx2'))
            df = auth_df[columns].iloc[0]
            json_df = df.to_json()
            st.json(json_df)
        else: 
            st.markdown('public key not found...')