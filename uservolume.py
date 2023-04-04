
import sys
from tokenize import tabsize
import driftpy
import pandas as pd 
import numpy as np 
from driftpy.math.oracle import *

pd.options.plotting.backend = "plotly"

# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.clearing_house import ClearingHouse
from driftpy.clearing_house_user import ClearingHouseUser
from driftpy.accounts import get_perp_market_account, get_spot_market_account, get_user_account, get_state_account
from driftpy.constants.numeric_constants import * 
from driftpy.clearing_house_user import get_token_amount
import os
import json
import streamlit as st
from driftpy.types import MarginRequirementType
from driftpy.constants.banks import devnet_banks, Bank
from driftpy.constants.markets import devnet_markets, Market
from driftpy.addresses import *
from dataclasses import dataclass
from solana.publickey import PublicKey
from helpers import serialize_perp_market_2, serialize_spot_market
from anchorpy import EventParser
import asyncio
import time
from enum import Enum
from driftpy.math.margin import MarginCategory, calculate_asset_weight
import datetime

def find_value(df, search_column, search_string, target_column):
    """
    Returns the value of a target column in a Pandas DataFrame where a search string is found in a search column.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to search.
    search_column (str): The name of the column to search for the search string.
    search_string (str): The string to search for in the search column.
    target_column (str): The name of the column to return the value from.
    
    Returns:
    The value of the target column in the row where the search string is found in the search column. If the search
    string is not found in the search column, returns None.
    """
    # Create a boolean mask for the rows where the search string is found in the search column.
    mask = df[search_column].str.contains(search_string, na=False)
    
    # Use the mask to filter the DataFrame and select the target column for the matching row.
    target_value = df.loc[mask, target_column].values
    
    # If there are no matching rows, return None. Otherwise, return the first matching value.
    return target_value[0] if len(target_value) > 0 else None

async def get_user_stats(clearing_house: ClearingHouse):
    ch = clearing_house
    all_user_stats = await ch.program.account['UserStats'].all()
    user_stats_df = pd.DataFrame([x.account.__dict__ for x in all_user_stats])
    return user_stats_df

async def show_user_volume(clearing_house: ClearingHouse):
    url = 'https://drift-historical-data.s3.eu-west-1.amazonaws.com/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH/'
    url += 'market/%s/trades/%s/%s/%s'
    mol1, molselect, mol0, mol2, _ = st.columns([3, 3, 3, 3, 10])
    market_name = mol1.selectbox('market', ['SOL-PERP', 'BTC-PERP', 'ETH-PERP', '1MBONK-PERP', 'MATIC-PERP', 'SOL'])
    range_selected = molselect.selectbox('range select:', ['daily', 'weekly'], 0)

    dates = []
    lastest_date = pd.to_datetime(datetime.datetime.now(), utc=True)
    if range_selected == 'daily':
        date = mol0.date_input('select approx. date:', lastest_date, min_value=datetime.datetime(2022,11,4), max_value=lastest_date) #(datetime.datetime.now(tzInfo)))
        dates = [date]
    elif range_selected == 'weekly':
        start_date = mol0.date_input('start date:', lastest_date - datetime.timedelta(days=7), min_value=datetime.datetime(2022,11,4), max_value=lastest_date) #(datetime.datetime.now(tzInfo)))
        end_date = mol2.date_input('end date:', lastest_date, min_value=datetime.datetime(2022,11,4), max_value=lastest_date) #(datetime.datetime.now(tzInfo)))
        dates = pd.date_range(start_date, end_date)
    dfs = []
    data_urls = []
    for date in dates:
        (year, month, day) = (date.strftime("%Y"), str(date.month), str(date.day))
        data_url = url % (market_name, year, month, day)
        data_urls.append(data_url)
        dfs.append(pd.read_csv(data_url))
    dfs = pd.concat(dfs)
    dd, _ = st.columns([6,5])
    with dd.expander('data sources'):
        st.write(data_urls)


    tabs = st.tabs(['Volume', 'Referrals'])
    with tabs[0]:
        maker_volume = dfs.groupby('maker')['quoteAssetAmountFilled'].sum()
        taker_volume = dfs.groupby('taker')['quoteAssetAmountFilled'].sum()
        df = pd.concat({'maker volume': maker_volume, 'taker volume': taker_volume},axis=1)
        df = df.sort_values('maker volume', ascending=False)
        st.dataframe(df)
    with tabs[1]:
        opt = st.radio('users:', ['by trader', 'by referrer'])
        if 'referrerReward' in dfs.columns:
            reward_volume = pd.DataFrame(dfs[dfs.referrerReward>0].groupby('taker')['quoteAssetAmountFilled'].sum())
            reward_volume.columns = ['taker volume']
            if opt == 'by referrer':
                user_stats_df = await get_user_stats(clearing_house)
                user_stats_df = user_stats_df[user_stats_df.referrer.astype(str)!='11111111111111111111111111111111']

                def get_subaccounts(authority, number_of_sub_accounts_created):
                    userAccountKeys = []
                    for sub_id in range(number_of_sub_accounts_created):    
                        user_account_pk = get_user_account_public_key(clearing_house.program_id,
                        authority,
                        sub_id)
                        userAccountKeys.append(str(user_account_pk))
                    return ','.join(userAccountKeys)

                user_stats_df['subAccounts'] = user_stats_df.apply(lambda x: get_subaccounts(x['authority'], x['number_of_sub_accounts_created'])
                , axis=1)
                reward_volume = reward_volume.reset_index()
                st.dataframe(user_stats_df[['authority', 'subAccounts', 'referrer']])
            st.dataframe(reward_volume)
        else:
            st.code('no referrerRewards were seen in this market')



    
    