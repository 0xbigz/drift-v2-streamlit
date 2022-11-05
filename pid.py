
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


async def show_pid_positions(pid='', url='https://api.devnet.solana.com'):
    config = configs['devnet']

    # print(config)
    # random key 
    with open("DRFTL7fm2cA13zHTSHnTKpt58uq5E49yr2vUxuonEtYd.json", 'r') as f: secret = json.load(f) 
    kp = Keypair.from_secret_key(bytes(secret))

    wallet = Wallet(kp)
    connection = AsyncClient(url)
    provider = Provider(connection, wallet)
    ch = ClearingHouse.from_config(config, provider)

    with st.expander('pid='+str(    ch.program_id) + " config"):
        # print(str(config))
        st.text(str(config))

    state = await get_state_account(ch.program)

    with st.expander('state'):
        st.text(str(state))

    all_users = await ch.program.account['User'].all()

    st.text('Number of Driftoors:' + str(len(all_users)))

    dfs = {}
    spotdfs = {}
    from driftpy.types import User
    for x in all_users:
        key = str(x.public_key)
        account: User = x.account

        dfs[key] = []
        name = str(''.join(map(chr, account.name)))

        for idx, pos in enumerate(x.account.perp_positions):
            dd = pos.__dict__
            dd['position_index'] = idx
            dd['authority'] = str(x.account.authority)
            dd['name'] = name
            dfs[key].append(pd.Series(dd))
        dfs[key] = pd.concat(dfs[key],axis=1) 
        
        spotdfs[key] = []
        for idx, pos in enumerate(x.account.spot_positions):
            dd = pos.__dict__
            dd['position_index'] = idx
            dd['authority'] = str(x.account.authority)
            spotdfs[key].append(pd.Series(dd))
        spotdfs[key] = pd.concat(spotdfs[key],axis=1)    
    perps = pd.concat(dfs,axis=1).T
    print(perps)
    perps.index = perps.index.set_names(['public_key', 'idx2'])
    perps = perps.reset_index()

    spots = pd.concat(spotdfs,axis=1).T
    spots.index = spots.index.set_names(['public_key', 'idx2'])
    spots = spots.reset_index()
    
    markettype = st.radio(
    "MarketType",
    ('Perp', 'Spot'))
    # print(state)
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
                df1['lp_shares'] /= 1e9
                df1['quote_asset_amount'] /= 1e6
                df1['quote_entry_amount'] /= 1e6
                df1['quote_break_even_amount'] /= 1e6

                df1['entry_price'] = -df1['quote_entry_amount']/df1['base_asset_amount'].apply(lambda x: 1 if x==0 else x)
                df1['breakeven_price'] = -df1['quote_break_even_amount']/df1['base_asset_amount'].apply(lambda x: 1 if x==0 else x)
                df1['cost_basis'] = -df1['quote_asset_amount']/df1['base_asset_amount'].apply(lambda x: 1 if x==0 else x)

                toshow = df1[[
                    'public_key', 'name', 'open_orders', 'lp_shares', 'base_asset_amount', 
                    'entry_price', 'breakeven_price', 'cost_basis',
                    'authority', 
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